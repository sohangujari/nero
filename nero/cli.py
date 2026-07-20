import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from nero import __version__
from nero.config.manager import ConfigError, ConfigManager
from nero.config.schema import NeroConfig
from nero.core.chat_loop import ChatLoop
from nero.hardware.detector import HardwareSpecs, detect_hardware, recommend_model
from nero.llm import ollama
from nero.llm.client import LLMClient
from nero.tools.open_app import OpenAppTool

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
        logging.basicConfig(format="%(name)s: %(message)s")
        logging.getLogger("nero").setLevel(logging.DEBUG)
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
        tools=[OpenAppTool()],
        api_key=api_key,
    )
    ChatLoop(client, console=console, assistant_name=config.assistant.name).run()


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


@config_app.command("show")
def config_show() -> None:
    """Print the current configuration (API key masked)."""
    manager = ConfigManager()
    config = _load_or_exit(manager)
    console.print(_config_table(manager, config))


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
    table.add_row(f"API Key ({config.llm.provider})", _key_display(manager, config.llm.provider))
    hardware = config.hardware
    if hardware.detected_ram_gb is not None:
        table.add_row("Detected Hardware", f"{hardware.detected_ram_gb:g} GB RAM, {hardware.detected_cpu_cores} cores")
        table.add_row("Recommended Local Model", hardware.recommended_local_model or "-")
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
        elif choice == "4":
            if provider == "ollama":
                console.print("Ollama runs locally — no API key needed.")
                continue
            new_key = typer.prompt(f"{provider} API key", hide_input=True).strip()
            if new_key:
                manager.set_api_key(provider, new_key)
        else:
            console.print("[yellow]Pick 1–4, or press Enter to finish.[/yellow]")
            continue
        console.print("[green]Saved.[/green]\n")


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
        return
    default_model = DEFAULT_MODELS[new_provider]
    manager.set_value("llm.model", default_model)
    console.print(f"Model set to [bold]{default_model}[/bold] — edit via option 3 if needed.")
    if not manager.get_api_key(new_provider):
        new_key = typer.prompt(f"{new_provider} API key", hide_input=True).strip()
        if new_key:
            manager.set_api_key(new_provider, new_key)
