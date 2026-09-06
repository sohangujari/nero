import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
import threading
import uuid

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from nero import __version__, routines, ui
from nero._runtime import check_python_version
from nero.config.manager import ConfigError, ConfigManager
from nero.config.schema import NeroConfig, STTConfig, TTSConfig
from nero.core.approvals import ApprovalQueue, default_approvals_path, queue_confirm
from nero.core.audit_log import AuditLog, default_audit_path
from nero.core.chat_loop import ChatLoop
from nero.mcp import MCPConnection, MCPError, load_servers
from nero.dashboard import run_dashboard
from nero.hardware.detector import (
    HardwareSpecs,
    detect_hardware,
    recommend_model,
    recommend_voice,
)
from nero.llm import ollama, openai_compat, providers
from nero.llm.client import LLMClient
from nero.memory.embeddings import Embedder
from nero.memory.facts import FactStore, default_facts_path
from nero.memory.history_store import HistoryStore, default_history_path
from nero.memory.notes import NoteIndex, default_notes_index_path
from nero.security import denylisted
from nero.skills.registry import build_registry
from nero.voice import audio_io
from nero.voice.audio_io import (
    AudioSource,
    Player,
    record_until_enter,
    record_until_silence,
)
from nero.voice.errors import TTSLoadError, VoiceDependencyError
from nero.voice.models import ensure_vad_model
from nero.voice.stt import STT_MODELS, FasterWhisperSTT
from nero.voice.tts import VOICE_CATALOG, build_tts
from nero.voice.vad import VoiceActivityDetector
from nero.telegram import PairingStore, TelegramBot, TelegramError, incoming, serve
from nero.voice.voice_loop import VoiceLoop

app = typer.Typer(add_completion=False, invoke_without_command=True)
config_app = typer.Typer(invoke_without_command=True, help="View and edit Nero Agent's configuration.")
app.add_typer(config_app, name="config")
facts_app = typer.Typer(invoke_without_command=True, help="View and manage facts Nero Agent remembers about you.")
app.add_typer(facts_app, name="facts")
notes_app = typer.Typer(help="Search and (re)index your notes directory.")
app.add_typer(notes_app, name="notes")
routine_app = typer.Typer(invoke_without_command=True, help="Manage scheduled routines (launchd).")
app.add_typer(routine_app, name="routine")
telegram_app = typer.Typer(invoke_without_command=True, help="Talk to Nero Agent from Telegram on your phone.")
app.add_typer(telegram_app, name="telegram")
approvals_app = typer.Typer(
    invoke_without_command=True, help="Review actions routines queued for approval."
)
app.add_typer(approvals_app, name="approvals")
console = Console()
logger = logging.getLogger("nero.cli")


def _enable_debug_logging() -> None:
    # DEBUG(hang): timestamps + thread name make it obvious which stage stalls.
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d [%(threadName)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("nero").setLevel(logging.DEBUG)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"nero {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Print the installed version and exit.",
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Verbose logging to stderr (tool-call plumbing, per-turn history).",
    ),
) -> None:
    """Nero Agent — your personal AI assistant.

    Run with no arguments for the universal session: the terminal chat, plus
    the Telegram bridge if you have paired a phone. `nero chat` is the terminal
    on its own, `nero talk` is voice, `nero telegram` is the bridge on its own.
    """
    if debug:
        _enable_debug_logging()
    warning = check_python_version()
    if warning and not getattr(sys, "frozen", False):
        # Frozen binaries bundle 3.12, so only warn on the pip/pipx path.
        Console(stderr=True).print(f"[yellow dim]{warning}[/yellow dim]")
    if ctx.invoked_subcommand is None:
        _run_chat(with_telegram=True)


def _load_or_exit(manager: ConfigManager) -> NeroConfig:
    try:
        return manager.load()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _apply_detection(
    manager: ConfigManager, config: NeroConfig | None = None
) -> tuple[NeroConfig, HardwareSpecs, str]:
    """Run detection, write results into the hardware block, save, return everything."""
    specs = detect_hardware()
    recommendation = recommend_model(specs)
    config = config if config is not None else _load_or_exit(manager)
    config.hardware.detected_ram_gb = specs.ram_gb
    config.hardware.detected_cpu_cores = specs.cpu_cores
    config.hardware.recommended_local_model = recommendation
    # Populate voice defaults from the hardware tier, but only when the user
    # hasn't explicitly changed them (mirrors recommended_local_model, without
    # clobbering a deliberate choice on a re-`detect`).
    stt_model, tts_engine = recommend_voice(specs)
    if config.voice.stt.model == STTConfig().model:
        config.voice.stt.model = stt_model
    if config.voice.tts.engine == TTSConfig().engine:
        config.voice.tts.engine = tts_engine
    manager.save(config)
    return config, specs, recommendation


@app.command()
def detect() -> None:
    """Re-run hardware detection and refresh the local-model recommendation."""
    manager = ConfigManager()
    config, specs, recommendation = _apply_detection(manager)
    table = Table(title="hardware", show_header=False, min_width=45)
    table.add_row("RAM", f"{specs.ram_gb:g} GB")
    table.add_row("CPU cores", str(specs.cpu_cores))
    table.add_row("OS", specs.os)
    table.add_row("Ollama running", "yes" if specs.has_ollama else "no")
    table.add_row("Recommended local model", recommendation)
    console.print(table)
    console.print(
        f"Provider unchanged ([bold]{config.llm.provider}[/bold]) — "
        "switch with [bold]nero config[/bold]."
    )


@app.command()
def mcp() -> None:
    """List the configured MCP servers and the tools they expose."""
    config = _load_or_exit(ConfigManager())
    if not config.mcp.servers:
        console.print(
            "[dim]No MCP servers configured.[/dim] Add one under [bold]mcp.servers[/bold] "
            "in your config file."
        )
        return
    table = Table(title="mcp servers", show_header=True)
    table.add_column("server")
    table.add_column("status")
    table.add_column("trust", style="dim")
    table.add_column("tools")
    for name, server in config.mcp.servers.items():
        if not server.enabled:
            table.add_row(name, "[dim]disabled[/dim]", "", "")
            continue
        connection = MCPConnection(
            name=name, command=server.command, args=server.args,
            env=server.env, timeout=server.timeout_seconds,
        )
        try:
            connection.start()
            tools = connection.list_tools()
        except MCPError as exc:
            table.add_row(name, f"[red]{escape(str(exc))}[/red]", "", "")
            continue
        finally:
            connection.close()
        table.add_row(
            name,
            "[green]ok[/green]",
            "trusted" if server.trusted else "confirms",
            f"{len(tools)}: " + escape(", ".join(tool.get("name", "?") for tool in tools)),
        )
    console.print(table)


@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="How many entries to show."),
) -> None:
    """Show what Nero Agent has actually done — a log of recent skill invocations."""
    entries = AuditLog(default_audit_path()).recent(limit)
    if not entries:
        console.print("[dim]No skill invocations recorded yet.[/dim]")
        return
    table = Table(title=f"nero history — last {len(entries)}", show_header=True)
    table.add_column("when", style="dim", no_wrap=True)
    table.add_column("skill")
    table.add_column("arguments")
    table.add_column("result")
    table.add_column("via", style="dim")
    # recent() is newest-first; reverse so the table reads like a log, oldest at top.
    for entry in reversed(entries):
        table.add_row(
            entry.timestamp.strftime("%Y-%m-%d %H:%M"),
            entry.skill_name,
            escape(json.dumps(entry.arguments, ensure_ascii=False)),
            escape(entry.result_summary),
            entry.provider,
        )
    console.print(table)


@app.command()
def forget() -> None:
    """Clear Nero Agent's remembered conversation history (not the audit log)."""
    store = HistoryStore(default_history_path(), session_id="-")
    if not store.recent(limit=1):
        console.print("[dim]Conversation history is already empty.[/dim]")
        return
    if not typer.confirm("Clear all remembered conversation history?", default=False):
        console.print("[dim]Kept. Nothing was cleared.[/dim]")
        return
    removed = store.clear()
    console.print(f"[green]Cleared[/green] {removed} stored messages.")


def _print_pending_approvals_notice() -> None:
    """One yellow line at chat startup when routines have queued destructive
    calls for review. Never blocks: a corrupt/unreadable approvals db just
    means no notice, not a startup failure."""
    try:
        pending = ApprovalQueue(default_approvals_path()).pending()
    except Exception:  # noqa: BLE001 — a notice must never block chat startup
        return
    if pending:
        console.print(
            f"[yellow]{len(pending)} action(s) from routines are waiting for your "
            "approval — review with nero approvals[/yellow]"
        )


@routine_app.callback()
def routine_main(ctx: typer.Context) -> None:
    """Without a subcommand, lists configured routines."""
    if ctx.invoked_subcommand is None:
        routine_list()


@routine_app.command("list")
def routine_list() -> None:
    """List configured routines: schedule, enabled, and whether installed."""
    config = _load_or_exit(ConfigManager())
    if not config.routines.routines:
        console.print(
            "[dim]No routines configured.[/dim] Add one under "
            "[bold]routines.routines[/bold] in your config file."
        )
        return
    agents_dir = routines.default_agents_dir()
    table = Table(title="nero routines", show_header=True)
    table.add_column("name")
    table.add_column("schedule")
    table.add_column("enabled")
    table.add_column("installed")
    for name, routine in config.routines.routines.items():
        table.add_row(
            name,
            routine.schedule,
            "yes" if routine.enabled else "no",
            "yes" if routines.is_installed(name, agents_dir) else "no",
        )
    console.print(table)


@routine_app.command("run")
def routine_run(name: str) -> None:
    """Headless single turn: send the routine's prompt, print the reply.

    No history persistence (a routine is not a conversation), no MCP servers.
    Any destructive skill call is refused and queued for approval — see
    nero.core.approvals.queue_confirm.
    """
    manager = ConfigManager()
    config = _load_or_exit(manager)
    routine = config.routines.routines.get(name)
    if routine is None:
        console.print(f"[red]No routine named {name!r}.[/red]")
        raise typer.Exit(1)
    if not routine.enabled:
        console.print(f"[red]Routine {name!r} is disabled.[/red]")
        raise typer.Exit(1)

    api_key = _provider_preflight(manager, config)
    queue = ApprovalQueue(default_approvals_path())
    registry = build_registry(
        config,
        audit=AuditLog(default_audit_path()),
        confirm=queue_confirm(queue, name),
    )
    client = LLMClient(
        config=config.llm,
        assistant_name=config.assistant.name,
        registry=registry,
        api_key=api_key,
    )
    messages = [{"role": "user", "content": routine.prompt}]
    client.send(messages, on_text=lambda text: console.print(text, end=""))
    console.print()


@routine_app.command("install")
def routine_install(name: str) -> None:
    """Write and load the launchd agent for a configured routine."""
    config = _load_or_exit(ConfigManager())
    routine = config.routines.routines.get(name)
    if routine is None:
        console.print(f"[red]No routine named {name!r}.[/red]")
        raise typer.Exit(1)
    try:
        executable = routines.resolve_executable()
        message = routines.install_routine(
            name, routine, executable, routines.default_agents_dir()
        )
    except routines.RoutineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(message)


@routine_app.command("uninstall")
def routine_uninstall(name: str) -> None:
    """Unload and remove the launchd agent for a routine."""
    console.print(routines.uninstall_routine(name, routines.default_agents_dir()))


@approvals_app.callback()
def approvals_main(ctx: typer.Context) -> None:
    """Without a subcommand, lists actions routines queued for approval."""
    if ctx.invoked_subcommand is None:
        pending = ApprovalQueue(default_approvals_path()).pending()
        if not pending:
            console.print("[dim]No actions waiting for approval.[/dim]")
            return
        table = Table(title="nero approvals", show_header=True)
        table.add_column("id")
        table.add_column("routine")
        table.add_column("skill")
        table.add_column("arguments")
        table.add_column("when", style="dim")
        for item in pending:
            table.add_row(
                str(item.id),
                item.routine,
                item.skill,
                escape(json.dumps(item.arguments, ensure_ascii=False)),
                item.requested_at.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(table)


@approvals_app.command("run")
def approvals_run(entry_id: int) -> None:
    """Show a pending request, ask for interactive confirmation, and — only if
    approved — execute the skill through a normal registry (so it is audited)
    and discard the entry. Non-TTY refuses and leaves the row pending: this
    must never auto-approve."""
    queue = ApprovalQueue(default_approvals_path())
    item = queue.get(entry_id)
    if item is None:
        console.print(f"[red]No pending approval with id {entry_id}.[/red]")
        raise typer.Exit(1)
    console.print(
        f"[yellow]Routine[/yellow] [bold]{item.routine}[/bold] wants to run "
        f"[bold]{item.skill}[/bold] with:"
    )
    console.print(json.dumps(item.arguments, indent=2, ensure_ascii=False, default=str))

    manager = ConfigManager()
    config = _load_or_exit(manager)
    approved = {"value": False}

    def confirm(skill_name: str, tier: str, arguments: dict) -> bool:
        approved["value"] = _confirm_skill(skill_name, tier, arguments, config.security, False)
        return approved["value"]

    registry = build_registry(config, audit=AuditLog(default_audit_path()), confirm=confirm)
    result = asyncio.run(registry.execute(item.skill, item.arguments, provider="approvals"))
    console.print(result)
    if approved["value"]:
        queue.discard(entry_id)
    else:
        console.print("[dim]Not approved. Left pending.[/dim]")


@approvals_app.command("discard")
def approvals_discard(entry_id: int) -> None:
    """Discard a pending approval without running it."""
    if ApprovalQueue(default_approvals_path()).discard(entry_id):
        console.print(f"[green]Discarded[/green] approval {entry_id}.")
    else:
        console.print(f"[dim]No pending approval with id {entry_id}.[/dim]")


@facts_app.callback()
def facts_main(ctx: typer.Context) -> None:
    """Without a subcommand, lists everything Nero Agent remembers about you."""
    if ctx.invoked_subcommand is None:
        facts = FactStore(default_facts_path()).all()
        if not facts:
            console.print("[dim]No facts remembered yet.[/dim]")
            return
        table = Table(title="nero facts", show_header=True)
        table.add_column("key")
        table.add_column("value")
        table.add_column("source", style="dim")
        table.add_column("updated", style="dim")
        for fact in facts:
            table.add_row(fact.key, escape(fact.value), fact.source or "-", fact.updated_at)
        console.print(table)


@facts_app.command("forget")
def facts_forget(key: str) -> None:
    """Forget one remembered fact by key."""
    if FactStore(default_facts_path()).forget(key):
        console.print(f"[green]Forgot[/green] {key}.")
    else:
        console.print(f"[dim]No fact stored under {key!r}.[/dim]")


def _notes_index(config: NeroConfig) -> NoteIndex | None:
    if not config.memory.notes_dir:
        console.print(
            "[yellow]No notes directory configured.[/yellow] Set one with "
            "[bold]nero config set memory.notes_dir <path>[/bold]."
        )
        return None
    return NoteIndex(default_notes_index_path(), config.memory.notes_dir, config.memory.notes_max_bytes)


@notes_app.command("index")
def notes_index_cmd() -> None:
    """Reindex the notes directory and report what changed."""
    index = _notes_index(_load_or_exit(ConfigManager()))
    if index is None:
        raise typer.Exit(1)
    added, updated, removed = index.reindex()
    console.print(f"[green]Indexed.[/green] added={added} updated={updated} removed={removed}")


@notes_app.command("search")
def notes_search_cmd(
    query: str, limit: int = typer.Option(5, "--limit", "-n", help="Max results.")
) -> None:
    """Search the indexed notes."""
    index = _notes_index(_load_or_exit(ConfigManager()))
    if index is None:
        raise typer.Exit(1)
    if index.is_empty():
        index.reindex()
    results = index.search(query, limit=limit)
    if not results:
        console.print(f"[dim]No notes match {query!r}.[/dim]")
        return
    table = Table(title=f"nero notes search — {query!r}", show_header=True)
    table.add_column("path")
    table.add_column("snippet")
    for path, snippet in results:
        table.add_row(escape(path), escape(snippet))
    console.print(table)


@telegram_app.callback()
def telegram_main(ctx: typer.Context) -> None:
    """Answer Telegram messages until you stop it (Ctrl+C)."""
    if ctx.invoked_subcommand is not None:
        return
    manager = ConfigManager()
    if not manager.exists():
        _first_time_setup(manager)
    config = _load_or_exit(manager)

    token = manager.get_telegram_token()
    if not token:
        console.print(
            "[yellow]No Telegram bot token stored.[/yellow] Run "
            "[bold]nero telegram setup[/bold] first."
        )
        raise typer.Exit(1)
    _print_pending_approvals_notice()
    api_key = _provider_preflight(manager, config)
    mcp_skills, mcp_connections = _load_mcp(config)
    registry = _build_registry(manager, config, extra_skills=mcp_skills)
    loop, mcp_connections = _build_chat_loop(manager, config, api_key, registry, mcp_connections)

    bot = TelegramBot(token)
    try:
        name = bot.username()
    except TelegramError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    allowed = set(config.telegram.allowed_chat_ids)
    console.print(
        f"[bold]{config.assistant.name}[/bold] is on Telegram as "
        f"[bold]@{name}[/bold], answering {len(allowed)} paired "
        f"chat{'s' if len(allowed) != 1 else ''}. Press Ctrl+C to stop."
    )
    if not allowed:
        console.print(
            f"[yellow]Nothing is paired yet.[/yellow] Open [bold]t.me/{name}[/bold], "
            "press Start, then run [bold]nero telegram approve <code>[/bold] with the "
            "code it sends you."
        )
    console.print(
        "[dim]Destructive skills stay refused here — there is no safe way to "
        "approve them from a phone.[/dim]\n"
    )
    try:
        serve(bot, allowed, loop.ask, pairings=PairingStore(),
              refresh=lambda: _allowed_chats(manager),
              on_event=lambda m: console.print(f"[dim]{escape(m)}[/dim]"))
    except KeyboardInterrupt:
        console.print("\n[dim]Telegram bridge stopped.[/dim]")
    except TelegramError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    finally:
        bot.close()
        for connection in mcp_connections:
            connection.close()


@telegram_app.command("install")
def telegram_install() -> None:
    """Keep the bridge running: at login, and again if it ever stops.

    Without this, `nero telegram` only answers while that terminal is open —
    close it and your phone goes quiet.
    """
    manager = ConfigManager()
    config = _load_or_exit(manager)
    if not manager.get_telegram_token():
        console.print(
            "[yellow]No Telegram bot token stored.[/yellow] Run "
            "[bold]nero telegram setup[/bold] first."
        )
        raise typer.Exit(1)
    if not config.telegram.allowed_chat_ids:
        console.print(
            "[yellow]Nothing is paired yet.[/yellow] Pair a chat first, or the "
            "service will start with nobody to answer."
        )
        raise typer.Exit(1)
    try:
        message = routines.install_bridge(
            routines.resolve_executable(), routines.default_agents_dir()
        )
    except routines.RoutineError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(message)
    console.print(
        "[dim]It is running now and will start again at login. "
        "Stop it with [bold]nero telegram uninstall[/bold].[/dim]"
    )


@telegram_app.command("uninstall")
def telegram_uninstall() -> None:
    """Stop the background bridge and remove its launchd agent."""
    console.print(routines.uninstall_bridge(routines.default_agents_dir()))


@telegram_app.command("approve")
def telegram_approve(
    code: str = typer.Argument(..., help="The six-digit code the bot sent you in Telegram."),
) -> None:
    """Pair the chat that was given this code.

    The code travels by phone and is typed here, so approving it is proof that
    whoever is at this terminal is also holding the paired device.
    """
    manager = ConfigManager()
    config = _load_or_exit(manager)
    chat_id = PairingStore().approve(code)
    if chat_id is None:
        console.print(
            "[red]No pending request matches that code.[/red] Codes expire after "
            "10 minutes — message the bot again for a fresh one."
        )
        raise typer.Exit(1)
    allowed = sorted(set(config.telegram.allowed_chat_ids) | {chat_id})
    manager.set_value("telegram.allowed_chat_ids", ",".join(str(i) for i in allowed))
    manager.set_value("telegram.enabled", "true")
    console.print(
        f"[green]Paired chat {chat_id}.[/green] "
        "A running [bold]nero telegram[/bold] picks it up within a few seconds."
    )


@telegram_app.command("pending")
def telegram_pending() -> None:
    """Chats waiting to be paired. Codes are shown in Telegram, never here."""
    waiting = PairingStore().pending()
    if not waiting:
        console.print("[dim]No chats are waiting to pair.[/dim]")
        return
    table = Table(title="pending pairings", show_header=True)
    table.add_column("Chat ID")
    table.add_column("Asked")
    for request in waiting:
        table.add_row(str(request.chat_id), request.age())
    console.print(table)
    console.print(
        "[dim]Approve with the code the bot sent to that chat: "
        "nero telegram approve <code>[/dim]"
    )


@telegram_app.command("setup")
def telegram_setup() -> None:
    """Store a bot token and pair the chat that is allowed to use it."""
    manager = ConfigManager()
    if not manager.exists():
        _first_time_setup(manager)
    if not _connect_telegram(manager):
        raise typer.Exit(1)


def _connect_telegram(manager: ConfigManager) -> bool:
    """Store a bot token and pair a chat. Returns False if nothing was set up.

    One routine, so `nero telegram setup` and the config menu can never drift
    into asking for different things. Never raises on a user mistake — the
    menu has to survive a cancelled step and redraw.
    """
    existing = manager.get_telegram_token()
    console.print(
        "Create a bot first: message [bold]@BotFather[/bold] on Telegram, send "
        "[bold]/newbot[/bold], and copy the token it gives you.\n"
        + ("[dim]A token is already stored — press Enter to keep it.[/dim]\n" if existing else "")
    )
    token = Prompt.ask("Bot token", password=True, default="", console=console).strip()
    if not token:
        if not existing:
            console.print("[yellow]Nothing entered — setup cancelled.[/yellow]")
            return False
        token = existing

    bot = TelegramBot(token)
    try:
        name = bot.username()
    except TelegramError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return False
    manager.set_telegram_token(token)
    console.print(f"[green]Connected as @{name}.[/green]")

    console.print(
        f"\nNow open [bold]t.me/{name}[/bold] in Telegram and press [bold]Start[/bold]. "
        "The bot will reply with a pairing code.\n[dim]Waiting… Ctrl+C to cancel.[/dim]"
    )
    pairings = PairingStore()
    try:
        chat_id = _await_pairing(bot, pairings)
    except KeyboardInterrupt:
        console.print("\n[dim]Pairing cancelled. The token was saved.[/dim]")
        return False
    except TelegramError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return False
    finally:
        bot.close()

    console.print(f"[green]Chat {chat_id} is asking to pair.[/green]")
    # Deliberately not shown here: typing the code Telegram displayed is the
    # only thing that proves the person at this terminal holds that phone.
    for attempt in range(3):
        entered = Prompt.ask("Enter the code shown in Telegram", console=console).strip()
        approved = pairings.approve(entered)
        if approved is not None:
            chat_id = approved
            break
        console.print("[red]That code does not match.[/red]" if attempt < 2 else "")
    else:
        console.print(
            "[yellow]Not paired.[/yellow] The token was saved — message the bot "
            "again and run [bold]nero telegram approve <code>[/bold]."
        )
        return False

    config = _load_or_exit(manager)
    allowed = sorted(set(config.telegram.allowed_chat_ids) | {chat_id})
    manager.set_value("telegram.allowed_chat_ids", ",".join(str(i) for i in allowed))
    manager.set_value("telegram.enabled", "true")
    console.print(
        f"[green]Paired chat {chat_id}.[/green] Start it with [bold]nero telegram[/bold]."
    )
    return True


def _allowed_chats(manager: ConfigManager) -> set[int]:
    """The paired chats, re-read from disk. Never raises: the bridge is a
    long-running server, and a config the user is mid-edit must not stop it."""
    try:
        return set(manager.load().telegram.allowed_chat_ids)
    except (ConfigError, OSError) as exc:
        logger.debug("could not re-read the allowlist: %s", exc)
        return set()


def _await_pairing(bot: TelegramBot, pairings: PairingStore) -> int:
    """Block until someone messages the bot, send them a code, return their id.

    Whoever arrives first only gets a *request*; it still has to be approved
    with the code they were sent, so being first is not enough to get paired.
    """
    from nero.telegram import PAIRING_REPLY

    offset = None
    while True:
        for update in bot.updates(offset):
            offset = update["update_id"] + 1
            received = incoming(update)
            if received is None:
                continue
            chat_id = received[0]
            bot.send(chat_id, PAIRING_REPLY.format(code=pairings.request(chat_id)))
            return chat_id


@app.command()
def dashboard(
    port: int = typer.Option(8642, "--port", help="Port to serve the local dashboard on."),
) -> None:
    """Serve a local read-only dashboard: history, skill audit, and config."""
    try:
        run_dashboard(port)
    except OSError as exc:
        console.print(f"[red]Could not start dashboard: {exc}[/red]")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        console.print("[dim]Dashboard stopped.[/dim]")


@app.command()
def talk(
    once: bool = typer.Option(False, "--once", help="Do a single voice exchange, then exit."),
    debug: bool = typer.Option(
        False, "--debug", help="Verbose per-stage voice pipeline logging (same as `nero --debug`)."
    ),
) -> None:
    """Talk to Nero Agent: recording stops on its own; talk over it to interrupt its reply."""
    if debug:  # DEBUG(hang): accept --debug after the subcommand too, not just before it.
        _enable_debug_logging()
    manager = ConfigManager()
    if not manager.exists():
        _first_time_setup(manager)
    config = _load_or_exit(manager)

    if not config.voice.enabled:
        console.print(
            "[yellow]Voice is disabled.[/yellow] Enable it with "
            "[bold]nero config[/bold] (or [bold]nero config set voice.enabled true[/bold])."
        )
        raise typer.Exit()

    api_key = _provider_preflight(manager, config)

    client = LLMClient(
        config=config.llm,
        assistant_name=config.assistant.name,
        registry=_build_registry(manager, config),
        api_key=api_key,
    )

    try:
        stt = FasterWhisperSTT(config.voice.stt.model, language=config.voice.stt.language)
    except VoiceDependencyError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    # Pre-flight: fetch model weights (with progress) and build the TTS engine ONCE,
    # before the loop starts. Doing this lazily inside make_player() meant a silent
    # ~300 MB download mid-turn — which looks exactly like a hang — and reloaded the
    # model on every single turn.
    try:
        _preflight_voice_models(config.voice.tts.engine)
        tts = build_tts(config.voice.tts.engine, config.voice.tts.voice_id)
    except VoiceDependencyError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    except TTSLoadError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc

    _prewarm(stt, tts)

    def make_player():
        return Player(tts, sample_rate=tts.SAMPLE_RATE)

    vad = _build_vad(config, console)
    # One microphone for the session: the recorder and the barge-in monitor
    # share it instead of opening (and contending for) the device per turn.
    source = AudioSource() if vad is not None else None
    if vad is not None:
        def record(prefix=None):
            return record_until_silence(
                console,
                vad,
                silence_ms=config.voice.vad.silence_ms,
                max_utterance_seconds=config.voice.vad.max_utterance_seconds,
                wait_for_speech_seconds=config.voice.vad.wait_for_speech_seconds,
                prefix=prefix,
                source=source,
            )
    else:
        def record(prefix=None):
            return record_until_enter(console, lambda: input())

    barge_in = config.voice.barge_in_active and vad is not None
    if (
        barge_in
        and not config.voice.force_barge_in
        and audio_io.output_is_builtin_speakers()
    ):
        barge_in = False
        console.print(
            "[dim]Barge-in is off: on built-in speakers Nero Agent would hear (and "
            "interrupt) itself. Headphones re-enable it, or force it with "
            "nero config set voice.force_barge_in true.[/dim]"
        )

    try:
        VoiceLoop(
            client=client,
            stt=stt,
            record=record,
            make_player=make_player,
            console=console,
            assistant_name=config.assistant.name,
            sample_rate=audio_io.RECORD_SAMPLE_RATE,
            once=once,
            history=_build_history(config),
            vad=vad,
            barge_in=barge_in,
            source=source,
            context_window=config.memory.compact_after_messages,
        ).run()
    finally:
        if source is not None:
            source.close()
        # We're exiting; a further Ctrl+C would only corrupt teardown output.
        _ignore_further_interrupts()


def _prewarm(*engines) -> None:
    """Pay each engine's cold-start cost before the loop starts.

    Kokoro's first synthesis measured 1383 ms against ~830 ms warm for the same
    sentence; CTranslate2 does comparable first-run setup. Paid here it is
    invisible next to the model loads that already happen; paid lazily it lands
    in the middle of the user's first turn, where it reads as a hang.

    Best-effort by definition: a warmup that fails costs nothing but the
    speed-up, so it must never keep `nero talk` from starting.
    """
    for engine in engines:
        try:
            engine.warmup()
        except Exception:  # noqa: BLE001 — an optimization is never a prerequisite
            logger.debug("warmup failed for %s", type(engine).__name__, exc_info=True)


def _build_history(config: NeroConfig) -> HistoryStore | None:
    """The one place conversation memory is constructed — both `nero` and
    `nero talk` call it, so text and voice can't drift. None when memory is off,
    so both loops treat "disabled" and "empty" identically (seed [], no append)."""
    if not config.memory.enabled:
        return None
    return HistoryStore(
        default_history_path(),
        session_id=uuid.uuid4().hex,
        max_turns=config.memory.max_history_turns,
        embedder=Embedder(enabled=config.memory.semantic_recall),
    )


def _build_vad(config: NeroConfig, console: Console) -> VoiceActivityDetector | None:
    """The one place the VAD is constructed — both `nero talk` paths use it.

    Returns None when VAD is off or unavailable. VAD is an enhancement, never a
    prerequisite: no failure here may cost the user a conversation, so every
    problem degrades to press-Enter-to-stop with a single printed line.
    """
    if not config.voice.vad.enabled:
        return None
    try:
        path = ensure_vad_model()
        return VoiceActivityDetector(path, threshold=config.voice.vad.threshold)
    except Exception as exc:  # noqa: BLE001 — download, load, or disk failure all degrade
        logger.debug("VAD unavailable: %s", exc, exc_info=True)
        console.print(
            "[yellow]Voice activity detection is unavailable "
            f"({exc}).[/yellow] Falling back to press Enter to stop recording."
        )
        return None


def _build_registry(manager: ConfigManager, config: NeroConfig, extra_skills=None):
    """The one place skills get constructed — both `nero` and `nero talk` call
    this, so the text and voice paths can never drift apart."""

    def remember_location(location: str) -> None:
        # Best-effort convenience only: a config-write failure (read-only
        # config dir, full disk -> OSError from save()'s write_text(), or a
        # bad value -> ConfigError) must never turn an already-successful
        # weather report into an error the user sees.
        with contextlib.suppress(ConfigError, OSError):
            manager.set_value("skills.weather.default_location", location)

    # The confirm callback's signature is fixed at (name, tier, arguments) —
    # it can't also take the registry it gates. Bind it via a one-element box
    # instead: `confirm` is only ever called during later dispatch, by which
    # point `registry_box` has been filled in below.
    registry_box: list = []

    def confirm(name: str, tier: str, arguments: dict) -> bool:
        return _confirm_skill(name, tier, arguments, config.security, registry_box[0].tainted)

    registry = build_registry(
        config,
        audit=AuditLog(default_audit_path()),
        on_location_resolved=remember_location,
        confirm=confirm,
        extra_skills=extra_skills,
    )
    registry_box.append(registry)
    return registry


def _confirm_skill(name: str, tier: str, arguments: dict, security, tainted: bool) -> bool:
    """Console confirmation prompt for a destructive skill call.

    Fails closed: no terminal (CliRunner, a pipe, a background job) means no
    prompt can be answered, so this returns False rather than hanging or
    auto-approving. A command argument matching `security.command_denylist`
    escalates to typing "yes" in full instead of a plain y/N.
    """
    if not console.is_terminal:
        return False
    console.print(
        f"[yellow]About to run[/yellow] [bold]{name}[/bold] "
        f"([bold]{tier}[/bold]) with:"
    )
    console.print(json.dumps(arguments, indent=2, ensure_ascii=False, default=str))
    if tainted:
        console.print(
            "[red]This turn ingested external content (web/file) — an injected "
            "instruction could be behind this call.[/red]"
        )
    matched = None
    for key, value in arguments.items():
        # git_command's "args" is a list of strings (e.g. ["reset", "--hard"])
        # rather than a single command string, and doesn't include the "git"
        # program name itself — reconstruct the same command string the skill
        # actually runs so it matches denylist entries like "git reset --hard".
        if (
            name == "git_command"
            and key == "args"
            and isinstance(value, list)
            and all(isinstance(item, str) for item in value)
        ):
            value = "git " + " ".join(value)
        if isinstance(value, str):
            matched = denylisted(value, security.command_denylist)
            if matched:
                break
    if matched:
        console.print(
            f"[red]Matches denylist pattern {matched!r}.[/red] Type [bold]yes[/bold] "
            "in full to proceed."
        )
        return Prompt.ask("Proceed?", default="", console=console).strip().lower() == "yes"
    return Confirm.ask("Proceed?", default=False, console=console)


@app.command()
def chat() -> None:
    """Chat in this terminal only — no Telegram bridge."""
    _run_chat(with_telegram=False)


def _run_chat(with_telegram: bool = False) -> None:
    manager = ConfigManager()
    if not manager.exists():
        _first_time_setup(manager)
    config = _load_or_exit(manager)
    _print_pending_approvals_notice()

    api_key = _provider_preflight(manager, config)
    mcp_skills, mcp_connections = _load_mcp(config)
    registry = _build_registry(manager, config, extra_skills=mcp_skills)

    loop, mcp_connections = _build_chat_loop(manager, config, api_key, registry, mcp_connections)
    bridge = _start_telegram_bridge(manager, config, loop) if with_telegram else None
    try:
        loop.run()
    finally:
        if bridge is not None:
            bridge.set()
        # A session must never leave orphaned server processes behind.
        for connection in mcp_connections:
            connection.close()


def _start_telegram_bridge(manager: ConfigManager, config: NeroConfig, loop):
    """Answer Telegram in the background of a terminal session, or None.

    Silent when Telegram isn't set up — `nero` must behave exactly as it always
    has for anyone who never touched it. The bridge shares `loop`, so a
    question asked from the phone and one typed here are the same conversation;
    ChatLoop.ask serializes them.
    """
    token = manager.get_telegram_token()
    if not token or not config.telegram.allowed_chat_ids:
        return None

    stop = threading.Event()

    def run() -> None:
        bot = TelegramBot(token)
        try:
            serve(
                bot,
                set(config.telegram.allowed_chat_ids),
                loop.ask,
                pairings=PairingStore(),
                refresh=lambda: _allowed_chats(manager),
                stop=stop,
                on_event=lambda m: logger.info("telegram: %s", m),
            )
        except Exception:  # noqa: BLE001 — a phone going quiet must not end the session
            logger.debug("telegram bridge stopped", exc_info=True)
        finally:
            bot.close()

    threading.Thread(target=run, daemon=True, name="nero-telegram").start()
    console.print(
        f"[dim]Also answering Telegram ({len(config.telegram.allowed_chat_ids)} paired). "
        "Run [/dim][bold]nero chat[/bold][dim] for the terminal alone.[/dim]"
    )
    return stop


def _build_chat_loop(manager, config, api_key, registry, mcp_connections):
    """The one place a ChatLoop is assembled — `nero` and `nero telegram` both
    call it, so a turn from a phone is the same turn as a turn in the terminal:
    same fallback chain, key rotation, memory and skills."""
    fallback_clients = _build_fallback_clients(manager, config, registry)
    coding_client = _resolve_coding_client(manager, config, registry)
    facts = [(fact.key, fact.value) for fact in FactStore(default_facts_path()).all()]

    client = LLMClient(
        config=config.llm,
        assistant_name=config.assistant.name,
        registry=registry,
        api_key=api_key,
        facts=facts,
    )
    loop = ChatLoop(
        client, console=console, assistant_name=config.assistant.name,
        history=_build_history(config), fallback_clients=fallback_clients,
        registry=registry, security=config.security,
        compact_after_messages=config.memory.compact_after_messages,
        route_by=config.llm.route_by, quality_rank=config.llm.quality_rank,
        health_check=config.llm.health_check,
        primary_api_keys=manager.get_api_keys(config.llm.provider),
        coding_client=coding_client,
    )
    return loop, mcp_connections


def _load_mcp(config: NeroConfig):
    """Spawn the configured MCP servers and wrap their tools as skills.

    Every failure is a warning, never a stop: a third-party server that won't
    start must not take the assistant down with it.
    """
    if not config.mcp.servers:
        return [], []
    builtin_names = build_registry(config).known_names()
    skills, connections, warnings = load_servers(config.mcp, builtin_names)
    for warning in warnings:
        console.print(f"[yellow]{escape(warning)}[/yellow]")
    return skills, connections


def _fallback_chain_entries(config: NeroConfig) -> tuple[list[tuple[str, str]], bool]:
    """The effective fallback chain: `llm.fallback_chain` wins if non-empty,
    else the scalar pair if both are set, else empty. The scalar pair is never
    rewritten into the chain (no migration) — this is a read-only resolution.

    Returns (entries, both_set) so the caller can print a warn-only note when
    both the chain and the scalar pair are configured.
    """
    chain = config.llm.fallback_chain
    scalar_set = bool(config.llm.fallback_provider and config.llm.fallback_model)
    if chain:
        return [tuple(entry.split("/", 1)) for entry in chain], scalar_set
    if scalar_set:
        return [(config.llm.fallback_provider, config.llm.fallback_model)], False
    return [], False


def _build_fallback_clients(
    manager: ConfigManager, config: NeroConfig, registry
) -> list[LLMClient]:
    """Resolve the effective fallback chain into live clients, in priority order.

    Filtering (blacklist/whitelist) and key/region resolution both happen once
    here at startup, not mid-turn — the design doc frames the dropped-entry
    note as firing "when the chain fires" (i.e. on a transient failure); doing
    it at startup instead is simpler and testable, and never blocks: a filtered
    or unresolvable entry is just dropped, chat still starts.
    """
    entries, both_set = _fallback_chain_entries(config)
    if both_set:
        console.print(
            "[yellow]Both llm.fallback_chain and llm.fallback_provider/"
            "llm.fallback_model are set — the chain wins.[/yellow]"
        )
    clients = []
    for provider, model in entries:
        if model in config.llm.model_blacklist:
            console.print(f"[dim]Skipping {provider}/{model}: blacklisted.[/dim]")
            continue
        if config.llm.model_whitelist and model not in config.llm.model_whitelist:
            console.print(f"[dim]Skipping {provider}/{model}: not on the whitelist.[/dim]")
            continue
        client, warning = _resolve_fallback_client(manager, config, registry, provider, model)
        if warning:
            console.print(
                f"[yellow]Fallback configured but {warning} — "
                f"{provider}/{model} disabled this session.[/yellow]"
            )
            continue
        clients.append(client)
    return clients


def _resolve_fallback_client(
    manager: ConfigManager, config: NeroConfig, registry, provider: str, model: str
) -> tuple[LLMClient | None, str | None]:
    """Build an LLMClient for the fallback provider+model, or explain why not.

    Non-exiting sibling of `_provider_preflight`: a missing fallback key or
    region must never block startup, only disable the fallback for this
    session. Resolves its own key from the fallback provider's keyring
    entry — never assumes "the" key means the primary provider's key.
    """
    if provider == "ollama":
        api_key = None
    elif provider == "bedrock":
        region = (
            config.llm.aws_region
            or os.environ.get("AWS_REGION_NAME")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        if not region:
            return None, "no AWS region configured"
        if not _bedrock_credentials_present():
            return None, "no AWS credentials found"
        api_key = None
    else:
        if provider in providers.CUSTOM_PROVIDERS and not config.llm.base_url:
            return None, "no endpoint URL configured"
        api_key = manager.get_api_key(provider)
        if not api_key and not providers.get(provider).key_optional:
            return None, f"no {provider} API key configured"

    fallback_config = config.llm.model_copy(update={"provider": provider, "model": model})
    client = LLMClient(
        config=fallback_config,
        assistant_name=config.assistant.name,
        registry=registry,
        api_key=api_key,
    )
    return client, None


def _resolve_coding_client(
    manager: ConfigManager, config: NeroConfig, registry
) -> LLMClient | None:
    """Resolve llm.coding_model ("provider/model") for the /code command, or
    None if unset or unresolvable.

    Non-exiting, like _resolve_fallback_client: a bad or missing coding_model
    must never block startup or error mid-conversation — /code just falls
    back to the primary model, with a dim notice printed at that point.
    """
    coding_model = config.llm.coding_model
    if not coding_model:
        return None
    provider, sep, model = coding_model.partition("/")
    if not sep:
        console.print(
            f'[yellow]llm.coding_model {coding_model!r} must be "provider/model" — '
            "/code will use the primary model.[/yellow]"
        )
        return None
    client, warning = _resolve_fallback_client(manager, config, registry, provider, model)
    if warning:
        console.print(
            f"[yellow]Coding model configured but {warning} — "
            "/code will use the primary model.[/yellow]"
        )
        return None
    return client


def _provider_preflight(manager: ConfigManager, config: NeroConfig) -> str | None:
    """Check the provider is usable and return its API key, or exit.

    The ollama branch is keyed on the provider name, not on keylessness: which
    providers happen to have a keyring entry is an accident of the provider
    table, not a routing decision. The pre-fix bug here was that a keyless
    `custom` endpoint's missing key was treated as fatal, ignoring that
    `custom` is `key_optional`.
    """
    provider = config.llm.provider
    if provider == "ollama":
        _ollama_preflight(config.llm.model)
        return None
    if provider == "bedrock":
        _bedrock_preflight(config)
        return None
    if provider in providers.CUSTOM_PROVIDERS and not config.llm.base_url:
        console.print(
            "[red]No Endpoint URL configured.[/red] Run [bold]nero config[/bold] "
            "and set the Endpoint URL."
        )
        raise typer.Exit(1)
    api_key = manager.get_api_key(provider)
    if not api_key and not providers.get(provider).key_optional:
        console.print(
            f"[red]No {provider} API key configured.[/red] "
            "Run [bold]nero config[/bold] and choose the API Key option."
        )
        raise typer.Exit(1)
    return api_key


def _bedrock_credentials_present() -> bool:
    """True if boto3's default chain resolves any credentials.

    Separate function so tests can stub the answer without importing boto3.
    The import is lazy — boto3 is only needed on the bedrock path.
    """
    try:
        import boto3

        return boto3.Session().get_credentials() is not None
    except Exception:  # noqa: BLE001 — any resolution failure means "not present"
        return False


def _bedrock_preflight(config: NeroConfig) -> None:
    """Fail fast with actionable messages before entering the chat loop.

    Region first: LiteLLM accepts it from config (aws_region_name) or the
    environment. Credentials second, via the ambient chain — Nero never
    stores AWS credentials itself.
    """
    region = (
        config.llm.aws_region
        or os.environ.get("AWS_REGION_NAME")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    if not region:
        console.print(
            "[red]No AWS region configured.[/red] Set it with "
            "[bold]nero config set llm.aws_region us-east-1[/bold] "
            "(or export AWS_REGION)."
        )
        raise typer.Exit(1)
    if not _bedrock_credentials_present():
        console.print(
            "[red]No AWS credentials found.[/red] Run [bold]aws configure[/bold], "
            "or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
        )
        raise typer.Exit(1)


def _ollama_preflight(model: str) -> None:
    """Fail fast with actionable messages before entering the chat loop."""
    if not ollama.reachable():
        console.print(
            "[red]Ollama isn't running.[/red] Start it with [bold]ollama serve[/bold], "
            "or switch providers with [bold]nero config[/bold]."
        )
        raise typer.Exit(1)
    if not ollama.has_model(model):
        if not typer.confirm(f"Model {model} isn't downloaded yet — pull it now?", default=True):
            console.print(
                f"Pick a different model with [bold]nero config[/bold], "
                f"or pull it later with [bold]ollama pull {model}[/bold]."
            )
            raise typer.Exit(1)
        if not ollama.pull_model(model):
            console.print(
                f"[red]Pull failed.[/red] Try [bold]ollama pull {model}[/bold] manually."
            )
            raise typer.Exit(1)


def _first_time_setup(manager: ConfigManager) -> None:
    console.print(
        Panel(
            "Welcome! Let's set Nero Agent up. This takes under a minute.",
            title="nero — first-time setup",
            border_style="cyan",
        )
    )
    config, specs, recommendation = _apply_detection(manager, NeroConfig())
    console.print(
        f"Detected: [bold]{specs.ram_gb:g} GB RAM[/bold], "
        f"[bold]{specs.cpu_cores} cores[/bold] — "
        f"recommended local model: [bold]{recommendation}[/bold]\n"
    )
    provider = ui.pick(
        "Provider",
        [(info.name, info.label) for info in providers.PROVIDERS],
        default="claude",
        console=console,
    ) or "claude"
    config.llm.provider = provider
    if provider == "ollama":
        config.llm.model = Prompt.ask("Local model", default=recommendation, console=console)
        manager.save(config)
        console.print("Ollama runs locally — no key needed.")
    elif provider in providers.CUSTOM_PROVIDERS:
        manager.save(config)  # provider only; the model is written by the helper
        _setup_custom_endpoint(manager, provider, None)
    elif provider == "bedrock":
        manager.save(config)  # provider only; region and model come from the helper
        _setup_bedrock(manager, None)
    else:
        config.llm.model = providers.get(provider).default_model
        manager.save(config)
        _pick_model(manager, provider, config.llm.model)
        api_key = typer.prompt(f"{provider} API key", hide_input=True).strip()
        manager.set_api_key(provider, api_key)
    console.print("[bold green]Nero Agent is ready.[/bold green]\n")


@config_app.callback()
def config_main(ctx: typer.Context) -> None:
    """Without a subcommand, opens the interactive config menu."""
    if ctx.invoked_subcommand is None:
        _interactive_menu()


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a config value by dotted path, e.g. `nero config set llm.provider ollama`."""
    manager = ConfigManager()
    try:
        manager.set_value(key, value)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            "Valid keys include: [bold]assistant.name[/bold], "
            f"[bold]llm.provider[/bold] ({'|'.join(providers.names())}), "
            "[bold]llm.model[/bold]"
        )
        raise typer.Exit(1) from exc
    console.print(f"[green]Saved:[/green] {key} = {value}")
    if key in ("llm.model", "llm.provider"):
        _warn_if_no_tool_support(manager)
        _warn_if_model_mismatched(manager)
    if key in ("llm.base_url", "llm.provider"):
        _warn_if_base_url_inert(manager)
    if key in ("llm.aws_region", "llm.provider"):
        _warn_if_aws_region_inert(manager)
    if key in ("llm.fallback_provider", "llm.fallback_model"):
        _warn_if_fallback_issue(manager)
    if key in ("llm.model", "llm.fallback_chain", "llm.model_blacklist", "llm.model_whitelist"):
        _warn_if_model_listed(manager)


@config_app.command("set-key")
def config_set_key(
    provider: str,
    slot: int = typer.Option(1, "--slot", min=1, help="Key slot to write (1 = base)."),
) -> None:
    """Store an API key for `provider`, optionally in rotation slot N (> 1)."""
    if provider not in providers.names():
        console.print(f"[red]Unknown provider: {provider}[/red]")
        raise typer.Exit(1)
    manager = ConfigManager()
    if not manager.provider_needs_key(provider):
        console.print(f"[red]{provider} does not use an API key.[/red]")
        raise typer.Exit(1)
    key = typer.prompt(f"{provider} API key (slot {slot})", hide_input=True).strip()
    if not key:
        console.print("[red]No key entered.[/red]")
        raise typer.Exit(1)
    manager.set_api_key(provider, key, slot=slot)
    console.print(f"[green]Saved key for {provider} (slot {slot}).[/green]")


def _warn_if_no_tool_support(manager: ConfigManager) -> None:
    """If the current model is an Ollama model that can't call tools, say so.

    Ollama reports this authoritatively via /api/show, so the warning is
    grounded, not a guess. Silent for cloud providers, and silent when Ollama
    can't answer (down, or model not pulled) — better quiet than a false alarm.
    """
    config = manager.load()
    if config.llm.provider != "ollama":
        return
    if ollama.supports_tools(config.llm.model) is False:
        console.print(
            f"[yellow]Heads up:[/yellow] [bold]{config.llm.model}[/bold] has no "
            "tool-calling support in Ollama, so skills like opening websites, "
            "checking the weather, or controlling music won't work with it. "
            "General conversation is fine. [bold]phi4-mini[/bold] or "
            "[bold]qwen3[/bold] are good choices if you want skills."
        )


def _warn_if_model_mismatched(manager: ConfigManager) -> None:
    """Say so when llm.model plainly belongs to a different provider.

    Only fires when the model sits in another provider's curated list — that's
    a fact, not a guess. Unknown models stay silent: the same "better quiet
    than a false alarm" rule as _warn_if_no_tool_support. This deliberately
    doesn't reach for catalog_models to settle more cases: that import is lazy
    so nero.llm.providers stays importable and testable without litellm
    installed, not so call sites like this one can dodge a startup cost —
    cli.py already pays the litellm import via LLMClient at module load.

    Warns only. Never rewrites the model: a script that sets a provider and
    then sets a model must not race against a silent correction.
    """
    config = manager.load()
    if config.llm.provider in providers.CUSTOM_PROVIDERS:
        return  # a custom endpoint can serve any model id; ownership means nothing there
    owners = [info.name for info in providers.PROVIDERS if config.llm.model in info.models]
    if not owners or config.llm.provider in owners:
        return
    console.print(
        f"[yellow]Heads up:[/yellow] [bold]{config.llm.model}[/bold] belongs to "
        f"[bold]{owners[0]}[/bold], not [bold]{config.llm.provider}[/bold]. "
        f"Set a model with [bold]nero config set llm.model <name>[/bold] "
        f"or pick one in [bold]nero config[/bold]."
    )


def _warn_if_base_url_inert(manager: ConfigManager) -> None:
    """Say so when a base URL is set that the current provider will not use.

    Warns only. The value is kept: switching to the custom provider later
    brings it back, and silently discarding it would be the side-effect
    mutation `_warn_if_model_mismatched` exists to avoid.
    """
    config = manager.load()
    if config.llm.base_url is None or config.llm.provider in providers.CUSTOM_PROVIDERS:
        return
    console.print(
        f"[yellow]Heads up:[/yellow] [bold]llm.base_url[/bold] applies only when "
        f"[bold]llm.provider[/bold] is a custom endpoint (custom, custom_anthropic); "
        f"it is currently [bold]{config.llm.provider}[/bold], so this endpoint "
        "will not be used."
    )


def _warn_if_aws_region_inert(manager: ConfigManager) -> None:
    """Say so when an AWS region is set that the current provider will not use.

    Warns only; the value is kept — the same rule llm.base_url follows.
    """
    config = manager.load()
    if config.llm.aws_region is None or config.llm.provider == "bedrock":
        return
    console.print(
        f"[yellow]Heads up:[/yellow] [bold]llm.aws_region[/bold] applies only "
        f"when [bold]llm.provider[/bold] is [bold]bedrock[/bold]; it is "
        f"currently [bold]{config.llm.provider}[/bold], so it will not be used."
    )


def _warn_if_fallback_issue(manager: ConfigManager) -> None:
    """Say so when the fallback pair is half-set or identical to the primary.

    Warns only; never mutates either field — the same rule llm.base_url and
    llm.aws_region follow.
    """
    config = manager.load()
    provider, model = config.llm.fallback_provider, config.llm.fallback_model
    if (provider is None) != (model is None):
        console.print(
            "[yellow]fallback needs both llm.fallback_provider and "
            "llm.fallback_model; currently inert[/yellow]"
        )
    elif provider is not None and (provider, model) == (config.llm.provider, config.llm.model):
        console.print("[yellow]fallback is the same as the primary model[/yellow]")


def _warn_if_model_listed(manager: ConfigManager) -> None:
    """Say when the current llm.model is blacklisted, or excluded by a
    non-empty whitelist. Warn-only, per the hard invariant that priority/lists
    constrain AUTOMATIC selection only — an explicit `config set llm.model`
    always wins and is set regardless.
    """
    config = manager.load()
    model = config.llm.model
    if model in config.llm.model_blacklist:
        console.print(
            f"[yellow]{model} is on llm.model_blacklist — automatic selection "
            "will avoid it, your explicit choice stands.[/yellow]"
        )
    elif config.llm.model_whitelist and model not in config.llm.model_whitelist:
        console.print(
            f"[yellow]{model} is not on llm.model_whitelist — automatic "
            "selection would skip it, your explicit choice stands.[/yellow]"
        )


@config_app.command("show")
def config_show() -> None:
    """Print the current configuration (API key masked)."""
    manager = ConfigManager()
    config = _load_or_exit(manager)
    console.print(_config_table(manager, config))


def _ignore_further_interrupts() -> None:
    """Stop reacting to Ctrl+C once we've committed to exiting.

    Without this, a second Ctrl+C (an impatient double-tap after "Bye!") lands
    inside interpreter teardown — typically numpy's NpzFile.__del__, since
    kokoro-onnx keeps voices-v1.0.bin open as an npz — producing a spurious
    "Exception ignored in: <function NpzFile.__del__>" traceback on a perfectly
    clean exit. Nothing is left to interrupt at that point, so ignore it.
    """
    with contextlib.suppress(ValueError, OSError):  # not the main thread / unsupported
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _preflight_voice_models(engine: str) -> None:
    """Download voice model weights up front, with a progress bar.

    Only Kokoro has fetchable weights today; other engines no-op.
    """
    if engine != "kokoro":
        return
    from nero.voice.models import ensure_kokoro_model, models_present

    if models_present():
        return
    console.print("[dim]First run: downloading voice models (~300 MB, one time)…[/dim]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        tasks: dict[str, int] = {}

        def on_progress(name: str, downloaded: int, total: int) -> None:
            if name not in tasks:
                tasks[name] = progress.add_task(name, total=total or None)
            progress.update(tasks[name], completed=downloaded)

        ensure_kokoro_model(on_progress=on_progress)
    console.print("[green]Voice models ready.[/green]")


def _voice_gender(voice_id: str) -> str:
    """Female/male for a voice id, catalog or not.

    A hand-set voice_id need not be one Nero curates — the config field is a
    plain string on purpose — so fall back to Kokoro's own naming, where the
    second letter of `<lang><gender>_name` is the gender.
    """
    for vid, _name, gender in VOICE_CATALOG:
        if vid == voice_id:
            return gender
    initial = voice_id[1:2]
    return {"f": "female", "m": "male"}.get(initial, "?")


def _key_display(manager: ConfigManager, provider: str, masked: bool = True) -> str:
    if not manager.provider_needs_key(provider):
        return "not needed"
    api_key = manager.get_api_key(provider)
    if not api_key:
        return "not set"
    label = manager.mask_api_key(api_key) if masked else "configured"
    extra_slots = len(manager.get_api_keys(provider)) - 1
    if extra_slots > 0:
        label += f" (+{extra_slots} more slot{'s' if extra_slots != 1 else ''})"
    return label


def _config_table(
    manager: ConfigManager, config: NeroConfig, session_cost: float = 0.0
) -> Table:
    table = Table(title="nero config", show_header=False, min_width=45)
    table.add_row("Assistant Name", config.assistant.name)
    table.add_row("LLM Provider", config.llm.provider)
    table.add_row("LLM Model", config.llm.model)
    if config.llm.provider in providers.CUSTOM_PROVIDERS:
        table.add_row("Endpoint URL", config.llm.base_url or "not set")
    elif config.llm.provider == "bedrock":
        table.add_row("AWS Region", config.llm.aws_region or "not set")
    table.add_row("Mode", config.mode)
    table.add_row(f"API Key ({config.llm.provider})", _key_display(manager, config.llm.provider))
    if config.llm.fallback_provider and config.llm.fallback_model:
        table.add_row("Fallback", f"{config.llm.fallback_provider}/{config.llm.fallback_model}")
    else:
        table.add_row("Fallback", "off")
    table.add_row("Fallback Chain", ", ".join(config.llm.fallback_chain) or "—")
    table.add_row("Model Blacklist", ", ".join(config.llm.model_blacklist) or "—")
    table.add_row("Model Whitelist", ", ".join(config.llm.model_whitelist) or "—")
    table.add_row("Route By", config.llm.route_by)
    table.add_row("Health Check", "yes" if config.llm.health_check else "no")
    table.add_row("Coding Model", config.llm.coding_model or "not set")
    if session_cost:
        table.add_row("Session Cost", f"${session_cost:.4f}")
    hardware = config.hardware
    if hardware.detected_ram_gb is not None:
        table.add_row("Detected Hardware", f"{hardware.detected_ram_gb:g} GB RAM, {hardware.detected_cpu_cores} cores")
        table.add_row("Recommended Local Model", hardware.recommended_local_model or "-")
    voice = config.voice
    table.add_row("Voice Enabled", "yes" if voice.enabled else "no")
    if voice.enabled:
        table.add_row("STT Model", voice.stt.model)
        table.add_row("TTS Engine", voice.tts.engine)
        table.add_row("Voice", f"{voice.tts.voice_id} ({_voice_gender(voice.tts.voice_id)})")
        table.add_row("VAD Auto-Stop", "yes" if voice.vad.enabled else "no")
        table.add_row("Barge-in", "yes" if voice.barge_in_active else "no")
    enabled = [name for name, on in config.skills.enabled.model_dump().items() if on]
    table.add_row("Skills Enabled", ", ".join(enabled) if enabled else "none")
    if config.skills.weather.default_location:
        table.add_row("Weather Location", config.skills.weather.default_location)
    memory = config.memory
    table.add_row("Memory", "yes" if memory.enabled else "no")
    if memory.enabled:
        table.add_row("History Turns", str(memory.max_history_turns))
    table.add_row("Telegram", _telegram_summary(manager, config).split("  (")[0])
    table.add_row("Security", _security_summary(config.security))
    return table


def _telegram_summary(manager: ConfigManager, config: NeroConfig) -> str:
    """One line describing whether the phone bridge is usable, and why not."""
    if not manager.get_telegram_token():
        return "not set up  (connect)"
    paired = len(config.telegram.allowed_chat_ids)
    if not paired:
        return "token stored, no chat paired  (pair)"
    return f"{paired} chat{'s' if paired != 1 else ''} paired  (add another)"


def _security_summary(security) -> str:
    limits = []
    if security.max_turns_per_session:
        limits.append(f"{security.max_turns_per_session} turns")
    if security.max_cost_usd_per_session:
        limits.append(f"${security.max_cost_usd_per_session:g}")
    limits_text = ", ".join(limits) if limits else "unlimited"
    return (
        f"{len(security.command_denylist)} denylist / "
        f"{len(security.command_allowlist)} allowlist / {limits_text}"
    )


def _interactive_menu() -> None:
    manager = ConfigManager()
    while True:
        config = _load_or_exit(manager)
        provider = config.llm.provider

        hardware = config.hardware
        if hardware.detected_ram_gb is not None:
            console.print(
                f"[dim]{hardware.detected_ram_gb:g} GB RAM, {hardware.detected_cpu_cores} cores"
                f" — recommended local model: {hardware.recommended_local_model or '-'}[/dim]"
            )
        else:
            console.print("[dim]Hardware not detected — run `nero detect`[/dim]")

        voice = config.voice
        memory = config.memory
        barge_in_hint = "" if voice.vad.enabled else "  (needs VAD auto-stop)"
        # Plain text, no rich markup: questionary renders labels verbatim, and
        # a bare [x] would be a style tag to the numbered fallback's table.
        rows = [
            ("1", "Assistant Name", config.assistant.name),
            ("2", "LLM Provider", f"{provider}  (change)"),
            ("3", "LLM Model", config.llm.model),
            ("4", f"API Key ({provider})", _key_display(manager, provider, masked=False)),
            ("5", "Voice Enabled", "yes" if voice.enabled else "no"),
            ("6", "STT Model", f"{voice.stt.model}  (auto)"),
            ("7", "TTS Engine", voice.tts.engine),
            ("8", "Voice", f"{voice.tts.voice_id} ({_voice_gender(voice.tts.voice_id)})"),
            ("9", "Mode", f"{config.mode}  (toggle)"),
            ("10", "Skills", _skills_summary(config)),
            ("11", "Weather Location", config.skills.weather.default_location or "not set"),
            ("12", "Memory", f"{'yes' if memory.enabled else 'no'}  (toggle)"),
            ("13", "History Turns", str(memory.max_history_turns)),
            (
                "14", "Barge-in",
                f"{'yes' if voice.barge_in_active else 'no'}  (toggle){barge_in_hint}",
            ),
            ("15", "VAD Auto-Stop", f"{'yes' if voice.vad.enabled else 'no'}  (toggle)"),
        ]
        if provider in providers.CUSTOM_PROVIDERS:
            rows.append(("16", "Endpoint URL", config.llm.base_url or "not set"))
        elif provider == "bedrock":
            rows.append(("16", "AWS Region", config.llm.aws_region or "not set"))
        rows.append(("17", "Telegram", _telegram_summary(manager, config)))
        # "" is the Done row: falsy, so it exits on the same branch Esc does,
        # and so does the blank answer the numbered fallback still accepts.
        choice = ui.pick(
            "nero config",
            [(value, f"{name:<20}  {text}") for value, name, text in rows] + [("", "Done")],
            console=console,
            prompt="Number (Enter to finish)",
        )
        if not choice:
            return
        if choice == "1":
            new_name = Prompt.ask("Assistant name", default=config.assistant.name, console=console)
            manager.set_value("assistant.name", new_name.strip() or config.assistant.name)
        elif choice == "2":
            new_provider = ui.pick(
                "LLM Provider",
                [(info.name, info.label) for info in providers.PROVIDERS],
                default=provider,
                console=console,
            )
            if new_provider and new_provider != provider:
                _switch_provider(manager, config, new_provider)
        elif choice == "3":
            _pick_model(manager, provider, config.llm.model)
            _warn_if_no_tool_support(manager)
        elif choice == "4":
            if not manager.provider_needs_key(provider):
                if provider == "bedrock":
                    console.print(
                        "Bedrock uses your AWS credentials — no API key stored here."
                    )
                else:
                    console.print(f"{provider} runs locally — no API key needed.")
                continue
            new_key = typer.prompt(f"{provider} API key", hide_input=True).strip()
            if new_key:
                manager.set_api_key(provider, new_key)
        elif choice == "5":
            manager.set_value("voice.enabled", str(not voice.enabled).lower())
        elif choice == "6":
            rows = [(model, f"{model}  ({note})") for model, note in STT_MODELS]
            rows.append((_CUSTOM_ROW, "Type a model name…"))
            picked = ui.pick("STT Model", rows, default=voice.stt.model, console=console)
            if picked == _CUSTOM_ROW:
                picked = Prompt.ask(
                    "STT model", default=voice.stt.model, console=console
                ).strip()
            if picked:
                manager.set_value("voice.stt.model", picked)
        elif choice == "7":
            new_engine = ui.pick(
                "TTS Engine", TTS_ENGINES, default=voice.tts.engine, console=console
            )
            if new_engine:
                manager.set_value("voice.tts.engine", new_engine)
        elif choice == "8":
            _pick_voice(manager, voice.tts.voice_id)
        elif choice == "9":
            manager.set_value("mode", "offline" if config.mode == "online" else "online")
        elif choice == "10":
            _skills_menu(manager, config)
        elif choice == "11":
            current = config.skills.weather.default_location or ""
            new_location = Prompt.ask(
                "Default weather location (blank to clear)", default=current, console=console
            ).strip()
            # set_value can't write null, so save the model directly.
            config.skills.weather.default_location = new_location or None
            manager.save(config)
        elif choice == "12":
            manager.set_value("memory.enabled", str(not memory.enabled).lower())
        elif choice == "13":
            new_turns = Prompt.ask(
                "How many exchanges to remember", default=str(memory.max_history_turns),
                console=console,
            ).strip()
            if new_turns:
                try:
                    manager.set_value("memory.max_history_turns", new_turns)
                except ConfigError:
                    console.print("[yellow]Enter a whole number of 0 or more.[/yellow]")
        elif choice == "14":
            manager.set_value("voice.barge_in", str(not voice.barge_in).lower())
        elif choice == "15":
            manager.set_value("voice.vad.enabled", str(not voice.vad.enabled).lower())
        elif choice == "16":
            if provider == "bedrock":
                new_region = Prompt.ask(
                    "AWS region", default=config.llm.aws_region or "us-east-1",
                    console=console,
                ).strip()
                if new_region:
                    manager.set_value("llm.aws_region", new_region)
            else:
                new_url = Prompt.ask(
                    "Endpoint base URL", default=config.llm.base_url or "", console=console
                ).strip()
                if new_url:
                    try:
                        manager.set_value("llm.base_url", new_url)
                    except ConfigError as exc:
                        console.print(f"[yellow]{exc}[/yellow]")
        elif choice == "17":
            _connect_telegram(manager)
        console.print("[green]Saved.[/green]\n")


def _pick_voice(manager: ConfigManager, current: str) -> None:
    picked = ui.pick(
        "Voice",
        [(vid, f"{name} ({gender})") for vid, name, gender in VOICE_CATALOG],
        default=current,
        console=console,
    )
    if picked:
        manager.set_value("voice.tts.voice_id", picked)


def _pick_custom_model(manager: ConfigManager, provider: str, current: str) -> None:
    """Free-text model entry, with an opt-in fetch from the endpoint itself.

    The LiteLLM catalog row is deliberately absent: a custom endpoint's models
    come from the endpoint, and offering OpenAI's catalog for a vLLM box would
    be worse than offering nothing.
    """
    fetch = (
        openai_compat.fetch_models
        if provider == "custom"
        else openai_compat.fetch_models_anthropic
    )
    base_url = manager.load().llm.base_url
    picked = ui.pick(
        "LLM Model — custom endpoint",
        [
            (_FETCH_ROW, "Fetch models from this endpoint…"),
            (_CUSTOM_ROW, "Type a model name…"),
        ],
        console=console,
    )
    if picked is None:
        return
    if picked == _FETCH_ROW:
        picked = _CUSTOM_ROW  # the fallback if anything below fails
        if not base_url:
            console.print("[yellow]Set the Endpoint URL first.[/yellow]")
        else:
            answered, models = fetch(base_url, manager.get_api_key(provider))
            if answered != base_url:
                console.print(
                    f"That server answered at [bold]{answered}[/bold], not "
                    f"[bold]{base_url}[/bold]."
                )
                if Confirm.ask("Store the corrected URL?", default=True, console=console):
                    manager.set_value("llm.base_url", answered)
            if models:
                picked = ui.pick(
                    "LLM Model", [(m, m) for m in models], default=current, console=console
                )
            else:
                console.print("[yellow]Could not list models from that endpoint.[/yellow]")
    if picked == _CUSTOM_ROW:
        picked = Prompt.ask("Model name", default=current, console=console).strip()
    if picked:
        manager.set_value("llm.model", picked)


# Sentinels for the model picker's non-model rows. "\0" can't collide with a
# real model name, so the caller can tell a chosen model from a chosen action.
_CATALOG_ROW = "\0catalog"
_CUSTOM_ROW = "\0custom"
_FETCH_ROW = "\0fetch"

# What the row 7 picker shows. This is presentation text; the Literal on
# TTSConfig is the source of truth, and a test asserts the two stay in step.
TTS_ENGINES: list[tuple[str, str]] = [
    ("kokoro", "Kokoro — local, fast, no network"),
    ("chatterbox", "Chatterbox — local, more expressive, heavier"),
    ("cloud", "Cloud — network required"),
]


def _pick_model(manager: ConfigManager, provider: str, current: str) -> None:
    """Curated shortlist first, the full LiteLLM catalog and free text behind it.

    The catalog is only fetched when its row is chosen — the lazy import keeps
    `nero.llm.providers` free of a litellm dependency at import time, which
    keeps it independently importable and testable.
    """
    if provider in providers.CUSTOM_PROVIDERS:
        _pick_custom_model(manager, provider, current)
        return
    info = providers.get(provider)
    if not info.models:
        # ollama: the model comes from hardware detection, not a cloud list.
        recommended = manager.load().hardware.recommended_local_model
        new_model = Prompt.ask(
            "Local model", default=current or recommended or "", console=console
        ).strip()
        if new_model:
            manager.set_value("llm.model", new_model)
        return

    rows = [(model, model) for model in info.models]
    rows.append((_CATALOG_ROW, "Show all models from LiteLLM's catalog…"))
    rows.append((_CUSTOM_ROW, "Type a model name…"))
    picked = ui.pick(f"LLM Model — {info.label}", rows, default=current, console=console)

    if picked == _CATALOG_ROW:
        catalog = providers.catalog_models(provider)
        if catalog:
            picked = ui.pick(
                f"LLM Model — {info.label} ({len(catalog)} from LiteLLM)",
                [(model, model) for model in catalog],
                default=current,
                console=console,
            )
        else:
            console.print(
                "[yellow]Couldn't read LiteLLM's catalog.[/yellow] Type the model name instead."
            )
            picked = _CUSTOM_ROW

    if picked == _CUSTOM_ROW:
        picked = Prompt.ask("Model name", default=current, console=console).strip()

    if picked and picked not in (_CATALOG_ROW, _CUSTOM_ROW):
        manager.set_value("llm.model", picked)


def _skills_summary(config: NeroConfig) -> str:
    toggles = config.skills.enabled.model_dump()
    enabled = [name for name, on in toggles.items() if on]
    return f"{len(enabled)}/{len(toggles)} enabled  (change)"


def _skills_menu(manager: ConfigManager, config: NeroConfig) -> None:
    toggles = config.skills.enabled.model_dump()
    # Ask the registry rather than hardcoding which skills need the network —
    # SkillMeta is the single source of truth.
    registry = build_registry(config)
    rows = []
    for name in toggles:
        skill = registry.get(name)
        needs_network = skill is not None and skill.meta.requires_network
        # Plain text: questionary does not parse rich markup.
        rows.append((name, f"{name} (needs network)" if needs_network else name))
    picked = ui.pick_many(
        "Skills", rows, {name for name, on in toggles.items() if on}, console=console
    )
    if picked is None:
        return
    for name, was_on in toggles.items():
        if (name in picked) != was_on:
            manager.set_value(f"skills.enabled.{name}", str(name in picked).lower())


def _setup_bedrock(manager: ConfigManager, current_region: str | None) -> None:
    """Prompt for region and model. Credentials come from the ambient AWS
    chain (aws configure, env vars, SSO) — Nero never stores them."""
    region = Prompt.ask(
        "AWS region", default=current_region or "us-east-1", console=console
    ).strip()
    if region:
        manager.set_value("llm.aws_region", region)
    default_model = providers.get("bedrock").default_model
    manager.set_value("llm.model", default_model)
    console.print(f"Model set to [bold]{default_model}[/bold].")
    _pick_model(manager, "bedrock", default_model)
    console.print(
        "Bedrock uses your AWS credentials (aws configure / env vars) — "
        "no key is stored by Nero Agent."
    )


def _setup_custom_endpoint(
    manager: ConfigManager, provider: str, current_url: str | None
) -> None:
    """Prompt for endpoint URL, model, and optional key.

    Shared by first-run setup and provider switching, which otherwise differ
    only in framing. The URL example forks per dialect because the canonical
    shapes are opposites: OpenAI-compatible bases end with /v1, Anthropic-
    compatible bases must not have it (LiteLLM appends /v1/messages itself).
    """
    example = (
        "e.g. http://localhost:1234/v1"
        if provider == "custom"
        else "e.g. https://api.moonshot.ai/anthropic — no /v1"
    )
    url = Prompt.ask(
        f"Endpoint base URL ({example})",
        default=current_url or "",
        console=console,
    ).strip()
    if url:
        try:
            manager.set_value("llm.base_url", url)
        except ConfigError as exc:
            # A malformed URL must not kill the menu, exactly as an out-of-range
            # memory.max_history_turns doesn't.
            console.print(f"[yellow]{exc}[/yellow]")
    _pick_custom_model(manager, provider, manager.load().llm.model)
    key = typer.prompt(
        "API key (blank if this endpoint needs none)",
        hide_input=True,
        default="",
        show_default=False,
    ).strip()
    if key:
        manager.set_api_key(provider, key)


def _switch_provider(manager: ConfigManager, config: NeroConfig, new_provider: str) -> None:
    if new_provider in providers.CUSTOM_PROVIDERS:
        manager.set_value("llm.provider", new_provider)
        _setup_custom_endpoint(manager, new_provider, config.llm.base_url)
        return
    if new_provider == "bedrock":
        manager.set_value("llm.provider", "bedrock")
        _setup_bedrock(manager, config.llm.aws_region)
        return
    manager.set_value("llm.provider", new_provider)
    if new_provider == "ollama":
        recommendation = config.hardware.recommended_local_model
        if not recommendation:
            _, _, recommendation = _apply_detection(manager)
        manager.set_value("llm.model", recommendation)
        console.print(
            f"Model set to [bold]{recommendation}[/bold] (hardware recommendation). "
            "No API key needed."
        )
        _warn_if_no_tool_support(manager)
        return
    default_model = providers.get(new_provider).default_model
    manager.set_value("llm.model", default_model)
    console.print(f"Model set to [bold]{default_model}[/bold].")
    _pick_model(manager, new_provider, default_model)
    if not manager.get_api_key(new_provider):
        new_key = typer.prompt(f"{new_provider} API key", hide_input=True).strip()
        if new_key:
            manager.set_api_key(new_provider, new_key)
