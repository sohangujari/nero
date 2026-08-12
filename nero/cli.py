import contextlib
import json
import logging
import signal
import sys
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
from rich.prompt import Prompt
from rich.table import Table

from nero import __version__
from nero._runtime import check_python_version
from nero.config.manager import ConfigError, ConfigManager
from nero.config.schema import NeroConfig, STTConfig, TTSConfig
from nero.core.audit_log import AuditLog, default_audit_path
from nero.core.chat_loop import ChatLoop
from nero.hardware.detector import (
    HardwareSpecs,
    detect_hardware,
    recommend_model,
    recommend_voice,
)
from nero.llm import ollama
from nero.llm.client import LLMClient
from nero.memory.history_store import HistoryStore, default_history_path
from nero.skills.registry import build_registry
from nero.voice import audio_io
from nero.voice.audio_io import Player, record_until_enter, record_until_silence
from nero.voice.errors import TTSLoadError, VoiceDependencyError
from nero.voice.models import ensure_vad_model
from nero.voice.stt import FasterWhisperSTT
from nero.voice.tts import VOICE_CATALOG, build_tts
from nero.voice.vad import VoiceActivityDetector
from nero.voice.voice_loop import VoiceLoop

PROVIDERS = ["claude", "openai", "gemini", "ollama"]
DEFAULT_MODELS = {
    "claude": "claude-sonnet-5",
    "openai": "gpt-5",
    "gemini": "gemini-2.5-pro",
}

app = typer.Typer(add_completion=False, invoke_without_command=True)
config_app = typer.Typer(invoke_without_command=True, help="View and edit Nero's configuration.")
app.add_typer(config_app, name="config")
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
    """Nero — your personal AI assistant in the terminal. Run with no arguments to chat."""
    if debug:
        _enable_debug_logging()
    warning = check_python_version()
    if warning and not getattr(sys, "frozen", False):
        # Frozen binaries bundle 3.12, so only warn on the pip/pipx path.
        Console(stderr=True).print(f"[yellow dim]{warning}[/yellow dim]")
    if ctx.invoked_subcommand is None:
        _run_chat()


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
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="How many entries to show."),
) -> None:
    """Show what Nero has actually done — a log of recent skill invocations."""
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
    """Clear Nero's remembered conversation history (not the audit log)."""
    store = HistoryStore(default_history_path(), session_id="-")
    if not store.recent(limit=1):
        console.print("[dim]Conversation history is already empty.[/dim]")
        return
    if not typer.confirm("Clear all remembered conversation history?", default=False):
        console.print("[dim]Kept. Nothing was cleared.[/dim]")
        return
    removed = store.clear()
    console.print(f"[green]Cleared[/green] {removed} stored messages.")


@app.command()
def talk(
    once: bool = typer.Option(False, "--once", help="Do a single voice exchange, then exit."),
    debug: bool = typer.Option(
        False, "--debug", help="Verbose per-stage voice pipeline logging (same as `nero --debug`)."
    ),
) -> None:
    """Talk to Nero: recording stops on its own; talk over it to interrupt its reply."""
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

    provider = config.llm.provider
    api_key = None
    if provider == "ollama":
        _ollama_preflight(config.llm.model)
    else:
        api_key = manager.get_api_key(provider)
        if not api_key:
            console.print(
                f"[red]No {provider} API key configured.[/red] "
                "Run [bold]nero config[/bold] and choose the API Key option."
            )
            raise typer.Exit(1)

    client = LLMClient(
        config=config.llm,
        assistant_name=config.assistant.name,
        registry=_build_registry(manager, config),
        api_key=api_key,
    )

    try:
        stt = FasterWhisperSTT(config.voice.stt.model)
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

    def make_player():
        return Player(tts, sample_rate=tts.SAMPLE_RATE)

    vad = _build_vad(config, console)
    if vad is not None:
        def record(prefix=None):
            return record_until_silence(
                console,
                vad,
                silence_ms=config.voice.vad.silence_ms,
                max_utterance_seconds=config.voice.vad.max_utterance_seconds,
                wait_for_speech_seconds=config.voice.vad.wait_for_speech_seconds,
                prefix=prefix,
            )
    else:
        def record(prefix=None):
            return record_until_enter(console, lambda: input())

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
            barge_in=config.voice.barge_in_active and vad is not None,
        ).run()
    finally:
        # We're exiting; a further Ctrl+C would only corrupt teardown output.
        _ignore_further_interrupts()


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


def _build_registry(manager: ConfigManager, config: NeroConfig):
    """The one place skills get constructed — both `nero` and `nero talk` call
    this, so the text and voice paths can never drift apart."""

    def remember_location(location: str) -> None:
        # Best-effort convenience only: a config-write failure (read-only
        # config dir, full disk -> OSError from save()'s write_text(), or a
        # bad value -> ConfigError) must never turn an already-successful
        # weather report into an error the user sees.
        with contextlib.suppress(ConfigError, OSError):
            manager.set_value("skills.weather.default_location", location)

    return build_registry(
        config,
        audit=AuditLog(default_audit_path()),
        on_location_resolved=remember_location,
    )


def _run_chat() -> None:
    manager = ConfigManager()
    if not manager.exists():
        _first_time_setup(manager)
    config = _load_or_exit(manager)
    provider = config.llm.provider

    api_key = None
    if provider == "ollama":
        _ollama_preflight(config.llm.model)
    else:
        api_key = manager.get_api_key(provider)
        if not api_key:
            console.print(
                f"[red]No {provider} API key configured.[/red] "
                "Run [bold]nero config[/bold] and choose the API Key option."
            )
            raise typer.Exit(1)

    client = LLMClient(
        config=config.llm,
        assistant_name=config.assistant.name,
        registry=_build_registry(manager, config),
        api_key=api_key,
    )
    ChatLoop(
        client, console=console, assistant_name=config.assistant.name,
        history=_build_history(config),
    ).run()


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
            "Welcome! Let's set Nero up. This takes under a minute.",
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
    provider = Prompt.ask("Provider", choices=PROVIDERS, default="claude", console=console)
    config.llm.provider = provider
    if provider == "ollama":
        config.llm.model = Prompt.ask("Local model", default=recommendation, console=console)
        console.print("Ollama runs locally — no key needed.")
    else:
        config.llm.model = DEFAULT_MODELS[provider]
        api_key = typer.prompt(f"{provider} API key", hide_input=True).strip()
        manager.set_api_key(provider, api_key)
    manager.save(config)
    console.print("[bold green]Nero is ready.[/bold green]\n")


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
            "[bold]llm.provider[/bold] (claude|openai|gemini|ollama), [bold]llm.model[/bold]"
        )
        raise typer.Exit(1) from exc
    console.print(f"[green]Saved:[/green] {key} = {value}")
    if key in ("llm.model", "llm.provider"):
        _warn_if_no_tool_support(manager)


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
    for vid, _name, gender in VOICE_CATALOG:
        if vid == voice_id:
            return gender
    return "?"


def _key_display(manager: ConfigManager, provider: str, masked: bool = True) -> str:
    if provider == "ollama":
        return "[dim]not needed[/dim]"
    api_key = manager.get_api_key(provider)
    if not api_key:
        return "[dim]not set[/dim]"
    return manager.mask_api_key(api_key) if masked else "[green]configured[/green]"


def _config_table(manager: ConfigManager, config: NeroConfig) -> Table:
    table = Table(title="nero config", show_header=False, min_width=45)
    table.add_row("Assistant Name", config.assistant.name)
    table.add_row("LLM Provider", config.llm.provider)
    table.add_row("LLM Model", config.llm.model)
    table.add_row("Mode", config.mode)
    table.add_row(f"API Key ({config.llm.provider})", _key_display(manager, config.llm.provider))
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
    return table


def _interactive_menu() -> None:
    manager = ConfigManager()
    while True:
        config = _load_or_exit(manager)
        provider = config.llm.provider

        body = Table(show_header=False, box=None, padding=(0, 2))
        hardware = config.hardware
        if hardware.detected_ram_gb is not None:
            body.add_row(
                "", "Detected",
                f"{hardware.detected_ram_gb:g} GB RAM, {hardware.detected_cpu_cores} cores",
            )
            body.add_row("", "Recommended local model", hardware.recommended_local_model or "-")
        else:
            body.add_row("", "Hardware", "[dim]not detected — run `nero detect`[/dim]")
        body.add_row("", "", "")
        body.add_row("1.", "Assistant Name", config.assistant.name)
        body.add_row("2.", "LLM Provider", f"{provider}  [dim]\\[change][/dim]")
        body.add_row("3.", "LLM Model", config.llm.model)
        body.add_row("4.", f"API Key ({provider})", _key_display(manager, provider, masked=False))
        voice = config.voice
        body.add_row("5.", "Voice Enabled", "yes" if voice.enabled else "no")
        body.add_row("6.", "STT Model", f"{voice.stt.model}  [dim]\\[auto][/dim]")
        body.add_row("7.", "TTS Engine", voice.tts.engine)
        body.add_row(
            "8.", "Voice",
            f"{voice.tts.voice_id} ({_voice_gender(voice.tts.voice_id)})  [dim]\\[change][/dim]",
        )
        body.add_row("9.", "Mode", f"{config.mode}  [dim]\\[toggle][/dim]")
        body.add_row("10.", "Skills", _skills_summary(config))
        body.add_row(
            "11.", "Weather Location",
            config.skills.weather.default_location or "[dim]not set[/dim]",
        )
        memory = config.memory
        body.add_row("12.", "Memory", f"{'yes' if memory.enabled else 'no'}  [dim]\\[toggle][/dim]")
        body.add_row("13.", "History Turns", str(memory.max_history_turns))
        barge_in_hint = (
            "  [dim](needs VAD auto-stop)[/dim]" if not voice.vad.enabled else ""
        )
        body.add_row(
            "14.", "Barge-in",
            f"{'yes' if voice.barge_in_active else 'no'}  [dim]\\[toggle][/dim]{barge_in_hint}",
        )
        body.add_row(
            "15.", "VAD Auto-Stop", f"{'yes' if voice.vad.enabled else 'no'}  [dim]\\[toggle][/dim]"
        )
        console.print(Panel(body, title="nero config", subtitle="Enter a number to edit, Enter to finish"))

        choice = Prompt.ask("Choice", default="", show_default=False, console=console).strip()
        if choice == "":
            return
        if choice == "1":
            new_name = Prompt.ask("Assistant name", default=config.assistant.name, console=console)
            manager.set_value("assistant.name", new_name.strip() or config.assistant.name)
        elif choice == "2":
            new_provider = Prompt.ask(
                "Provider", choices=PROVIDERS, default=provider, console=console
            )
            if new_provider != provider:
                _switch_provider(manager, config, new_provider)
        elif choice == "3":
            new_model = Prompt.ask("LLM model", default=config.llm.model, console=console)
            manager.set_value("llm.model", new_model.strip() or config.llm.model)
            _warn_if_no_tool_support(manager)
        elif choice == "4":
            if provider == "ollama":
                console.print("Ollama runs locally — no API key needed.")
                continue
            new_key = typer.prompt(f"{provider} API key", hide_input=True).strip()
            if new_key:
                manager.set_api_key(provider, new_key)
        elif choice == "5":
            manager.set_value("voice.enabled", str(not voice.enabled).lower())
        elif choice == "6":
            new_model = Prompt.ask("STT model", default=voice.stt.model, console=console)
            manager.set_value("voice.stt.model", new_model.strip() or voice.stt.model)
        elif choice == "7":
            new_engine = Prompt.ask(
                "TTS engine", choices=["kokoro", "chatterbox", "cloud"],
                default=voice.tts.engine, console=console,
            )
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
        else:
            console.print("[yellow]Pick 1–15, or press Enter to finish.[/yellow]")
            continue
        console.print("[green]Saved.[/green]\n")


def _pick_voice(manager: ConfigManager, current: str) -> None:
    table = Table(title="voices", show_header=True)
    table.add_column("#")
    table.add_column("id")
    table.add_column("name")
    table.add_column("gender")
    for i, (vid, name, gender) in enumerate(VOICE_CATALOG, start=1):
        marker = " (current)" if vid == current else ""
        table.add_row(str(i), vid, name + marker, gender)
    console.print(table)
    choice = Prompt.ask(
        "Voice number", choices=[str(i) for i in range(1, len(VOICE_CATALOG) + 1)],
        default="1", console=console,
    )
    voice_id = VOICE_CATALOG[int(choice) - 1][0]
    manager.set_value("voice.tts.voice_id", voice_id)


def _skills_summary(config: NeroConfig) -> str:
    toggles = config.skills.enabled.model_dump()
    enabled = [name for name, on in toggles.items() if on]
    return f"{len(enabled)}/{len(toggles)} enabled  [dim]\\[change][/dim]"


def _skills_menu(manager: ConfigManager, config: NeroConfig) -> None:
    toggles = config.skills.enabled.model_dump()
    names = list(toggles)
    # Ask the registry rather than hardcoding which skills need the network —
    # SkillMeta is the single source of truth.
    registry = build_registry(config)
    table = Table(title="skills", show_header=True)
    table.add_column("#")
    table.add_column("skill")
    table.add_column("enabled")
    table.add_column("needs network")
    for index, name in enumerate(names, start=1):
        skill = registry.get(name)
        needs_network = skill is not None and skill.meta.requires_network
        table.add_row(
            str(index), name,
            "yes" if toggles[name] else "no",
            "yes" if needs_network else "no",
        )
    console.print(table)
    choice = Prompt.ask(
        "Toggle which skill? (Enter to go back)", default="", show_default=False, console=console
    ).strip()
    if not choice:
        return
    if not choice.isdigit() or not 1 <= int(choice) <= len(names):
        console.print(f"[yellow]Pick 1–{len(names)}.[/yellow]")
        return
    name = names[int(choice) - 1]
    manager.set_value(f"skills.enabled.{name}", str(not toggles[name]).lower())


def _switch_provider(manager: ConfigManager, config: NeroConfig, new_provider: str) -> None:
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
    default_model = DEFAULT_MODELS[new_provider]
    manager.set_value("llm.model", default_model)
    console.print(f"Model set to [bold]{default_model}[/bold] — edit via option 3 if needed.")
    if not manager.get_api_key(new_provider):
        new_key = typer.prompt(f"{new_provider} API key", hide_input=True).strip()
        if new_key:
            manager.set_api_key(new_provider, new_key)
