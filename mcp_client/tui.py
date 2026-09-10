"""Full-screen Textual UI for manishcode.

Used for an interactive terminal session. Piped input, ``--plain`` and
``--debug`` keep the line-oriented REPL in :mod:`mcp_client.cli` instead.
"""

from __future__ import annotations

from pathlib import Path

from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Input, LoadingIndicator, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from .chat import ChatError, ChatSession, is_risky_tool
from .servers import ServerManager
from .ui import TuiUI, context_line

BANNER = r"""
 █▀▄▀█ █▀█ █▄ █ █ █▀ █ █ █▀▀ █▀█ █▀▄ █▀▀
 █ ▀ █ █▀█ █ ▀█ █ ▄█ █▀█ █▄▄ █▄█ █▄▀ ██▄
""".strip("\n")

COMMANDS: list[tuple[str, str]] = [
    ("/help", "show this list"),
    ("/new", "start a fresh conversation"),
    ("/tools", "list connected tools"),
    ("/model", "list models; /model NAME to switch | /model pull NAME to download"),
    ("/mcp", "list servers; /mcp connect NAME | /mcp disconnect NAME"),
    ("/context", "show token usage"),
    ("/save FILE", "write the conversation to a file"),
    ("/quit", "leave"),
]
_SUGGEST = SuggestFromList(
    ["/help", "/new", "/reset", "/clear", "/tools", "/model", "/model pull ",
     "/mcp", "/mcp connect ", "/mcp disconnect ", "/context", "/save ",
     "/quit", "/exit"],
    case_sensitive=False,
)


class ConfirmScreen(ModalScreen[bool]):
    """y/N modal shown before a side-effecting tool call runs."""

    BINDINGS = [
        Binding("y", "confirm(True)", "run"),
        Binding("n,escape", "confirm(False)", "skip"),
    ]

    def __init__(self, prompt: str) -> None:
        super().__init__()
        self._prompt = prompt

    def compose(self) -> ComposeResult:
        yield Static(
            Text.assemble(("confirm  ", "bold yellow"), (self._prompt, "")),
            id="confirm-box",
        )
        yield Static(Text("y  run      n  skip", style="dim"), id="confirm-hint")

    def action_confirm(self, ok: bool) -> None:
        self.dismiss(ok)


class ModelScreen(ModalScreen[str | None]):
    """Arrow-key picker for switching the active Ollama model."""

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, models: list[str], current: str) -> None:
        super().__init__()
        self._models = models
        self._current = current

    def compose(self) -> ComposeResult:
        with Container(id="model-box"):
            yield Static(Text("Select model", style="bold"))
            yield Static(Text("Enter to switch  ·  Esc to cancel  ·  "
                              "/model pull NAME to download", style="dim"))
            yield OptionList(*(
                Option(f"{'✓ ' if m == self._current else '  '}{m}", id=m)
                for m in self._models
            ), id="model-list")

    def on_mount(self) -> None:
        ol = self.query_one(OptionList)
        if self._current in self._models:
            ol.highlighted = self._models.index(self._current)
        ol.focus()

    def on_option_list_option_selected(
        self, ev: OptionList.OptionSelected
    ) -> None:
        self.dismiss(ev.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


def _make_confirm(app: "ManishcodeApp", mode: str):
    if mode == "none":
        return None

    async def confirm(name: str, tool_args: dict) -> bool:
        if mode == "risky" and not is_risky_tool(name):
            return True
        preview = str(tool_args)
        if len(preview) > 200:
            preview = preview[:200] + "…"
        return bool(await app.push_screen_wait(
            ConfirmScreen(f"run {name}({preview}) ?")))

    return confirm


class ManishcodeApp(App):
    CSS = """
    Screen { layout: vertical; background: $surface; }
    #log { height: 1fr; padding: 0 1; background: $surface; scrollbar-size-vertical: 1; }
    #footer { dock: bottom; height: auto; background: $panel; }
    #working { height: 1; display: none; color: $accent; background: $panel; }
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #prompt { border: round $primary; margin: 0 1; background: $surface; }
    #prompt:disabled { border: round $warning; }
    #prompt:focus { border: round $accent; }
    ConfirmScreen { align: center middle; background: $background 60%; }
    #confirm-box { width: 70%; max-width: 90; padding: 1 2; background: $panel;
                   border: round $warning; }
    #confirm-hint { width: 70%; max-width: 90; padding: 0 2; }
    ModelScreen { align: center middle; background: $background 60%; }
    #model-box { width: 70%; max-width: 90; height: auto; max-height: 80%;
                 padding: 1 2; background: $panel; border: round $accent; }
    #model-list { height: auto; max-height: 20; background: $panel; }
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "quit", priority=True),
        Binding("ctrl+q", "quit", "quit"),
    ]

    def __init__(self, *, session: ChatSession, manager: ServerManager,
                 all_specs: dict, context_limit: int, history_path, model: str,
                 confirm_mode: str) -> None:
        super().__init__()
        self.session = session
        self.session.ui = TuiUI(self)
        self.manager = manager
        self.all_specs = all_specs or {}
        self.context_limit = context_limit
        self.history_path = history_path
        self._confirm_mode = confirm_mode
        self._working = False
        self._phase = ""

    @property
    def model(self) -> str:
        return self.session.model

    # ---------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=False, highlight=False, min_width=20)
        with Container(id="footer"):
            yield LoadingIndicator(id="working")  # CSS keeps it hidden until a turn
            yield Static("", id="status")
            yield Input(id="prompt",
                        placeholder="Message the model, or / for commands",
                        suggester=_SUGGEST)

    def on_mount(self) -> None:
        self.session._confirm = _make_confirm(self, self._confirm_mode)
        log = self.query_one(RichLog)
        log.write(Text(BANNER, style="bold cyan"))
        n = len(self.manager.ollama_tools)
        log.write(Text(
            f"  {Path.cwd()}  ·  ctx {self.context_limit:,}  "
            f"·  / for commands, ctrl+c to quit", style="dim"))
        if n == 0:
            hint = ", ".join(self.all_specs) or "none in config"
            log.write(Text(f"  no tools loaded - /mcp connect NAME to add "
                           f"({hint})", style="dim"))
        log.write(Text(""))
        self._refresh_status()
        self.query_one(Input).focus()

    # ------------------------------------------------- hooks used by TuiUI
    def tui_write(self, renderable) -> None:
        self.query_one(RichLog).write(renderable)

    def tui_assistant(self, text: str) -> None:
        log = self.query_one(RichLog)
        log.write(Rule(style="green"))
        log.write(Markdown(text or "_(no response)_"))
        log.write(Text(""))

    def tui_set_working(self, working: bool) -> None:
        self._working = working
        if not working:
            self._phase = ""
        try:
            self.query_one("#working", LoadingIndicator).display = working
        except Exception:  # noqa: BLE001 - widget may not be mounted yet
            pass
        self._refresh_status()

    def tui_phase(self, label: str) -> None:
        self._phase = label
        self._refresh_status()

    # ---------------------------------------------------------- status bar
    def _refresh_status(self) -> None:
        conns = getattr(self.manager, "connections", [])
        servers = ", ".join(c.spec.name for c in conns) or "no servers"
        line = Text()
        line.append(f"{self.model}", style="bold")
        line.append(f"  ·  {len(self.manager.ollama_tools)} tools")
        line.append(f"  ·  {servers}")
        lim = self.context_limit
        if lim:
            used = getattr(self.session, "prompt_tokens", 0) or 0
            pct = used * 100 // lim
            line.append(
                f"  ·  ctx {used:,}/{lim:,} ({pct}%)",
                style="red" if pct >= 90 else "yellow" if pct >= 75 else "cyan")
        if self._working:
            line.append(f"  ·  {self._phase or 'working'}…", style="yellow")
        self.query_one("#status", Static).update(line)

    # --------------------------------------------------------------- input
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        self.query_one(Input).value = ""
        if not text or self._working:
            return
        if text.startswith("/"):
            # a worker context so commands may open modals (push_screen_wait)
            self.run_worker(self._command(text), name="command")
            return
        log = self.query_one(RichLog)
        log.write(Rule(style="cyan"))
        log.write(Text(text, style="bold cyan"))
        log.write(Text(""))
        self.query_one(Input).disabled = True
        self.tui_set_working(True)
        self.run_worker(self._turn(text), exclusive=True, name="turn")

    async def _turn(self, text: str) -> None:
        try:
            answer = await self.session.send(text)
            self.tui_assistant(answer)
        except ChatError as e:
            self.tui_write(Text(f"[chat error] {e}  —  keep going or /quit",
                                style="red"))
        finally:
            self.tui_set_working(False)
            inp = self.query_one(Input)
            inp.disabled = False
            inp.focus()
        try:
            dead = await self.manager.health_check()
        except Exception:  # noqa: BLE001
            dead = []
        if dead:
            self.tui_write(Text(
                f"[warn] server(s) not responding: {', '.join(dead)}",
                style="yellow"))

    # ------------------------------------------------------------ commands
    async def _command(self, text: str) -> None:
        log = self.query_one(RichLog)
        parts = text.split()
        cmd = parts[0].lower()
        if cmd in ("/quit", "/exit"):
            self.exit()
        elif cmd in ("/help", "/", "/?"):
            log.write(Text("commands", style="bold"))
            for name, help_text in COMMANDS:
                log.write(Text(f"  {name:<14} {help_text}", style="dim"))
        elif cmd in ("/new", "/reset", "/clear"):
            self.session.messages[:] = [
                m for m in self.session.messages
                if isinstance(m, dict) and m.get("role") == "system"]
            self.session.prompt_tokens = 0
            log.write(Text("(new conversation)", style="dim"))
            self._refresh_status()
        elif cmd == "/tools":
            log.write(Text(self.manager.describe_tools(), style="dim"))
        elif cmd == "/context":
            log.write(Text(context_line(self.session, self.context_limit),
                           style="dim"))
        elif cmd == "/save":
            from .history import save_history
            target = (Path(parts[1]).expanduser() if len(parts) > 1
                      else self.history_path)
            if not target:
                log.write(Text("usage: /save FILE", style="red"))
            else:
                save_history(target, self.session.messages, self.model)
                log.write(Text(f"saved to {target}", style="dim"))
        elif cmd == "/model":
            await self._model(parts)
        elif cmd == "/mcp":
            await self._mcp(parts)
        else:
            log.write(Text(f"unknown command {cmd!r} — /help for the list",
                           style="red"))

    async def _model(self, parts: list[str]) -> None:
        log = self.query_one(RichLog)
        if len(parts) == 1:
            try:
                names = self.session.list_models()
            except ChatError as e:
                log.write(Text(str(e), style="red"))
                return
            if not names:
                log.write(Text("no models installed — /model pull NAME",
                               style="red"))
                return
            choice = await self.push_screen_wait(
                ModelScreen(names, self.session.model))
            if not choice or choice == self.session.model:
                return
            try:
                self.session.set_model(choice)
            except ChatError as e:
                log.write(Text(f"  err  {e}", style="red"))
                return
            log.write(Text(f"  now using {self.session.model}", style="green"))
            self._refresh_status()
            return
        if parts[1].lower() == "pull" and len(parts) == 3:
            log.write(Text(f"pulling {parts[2]!r}… (may take a while)",
                           style="dim"))
            try:
                await self.session.pull_model(parts[2])
            except ChatError as e:
                log.write(Text(f"  err  {e}", style="red"))
                return
            log.write(Text(f"  ok  now using {self.session.model}",
                           style="green"))
            self._refresh_status()
            return
        if len(parts) == 2:
            try:
                self.session.set_model(parts[1])
            except ChatError as e:
                log.write(Text(f"  err  {e}", style="red"))
                return
            log.write(Text(f"  now using {self.session.model}", style="green"))
            self._refresh_status()
            return
        log.write(Text("usage: /model | /model NAME | /model pull NAME",
                       style="red"))

    async def _mcp(self, parts: list[str]) -> None:
        log = self.query_one(RichLog)
        action = parts[1].lower() if len(parts) > 1 else "list"
        if action in ("list", "ls", "status"):
            connected = {c.spec.name: c for c in self.manager.connections}
            names = sorted(set(self.all_specs) | set(connected))
            if not names:
                log.write(Text("no servers in config, none connected", style="dim"))
            for name in names:
                if name in connected:
                    c = connected[name]
                    tag = f"connected  {len(c.tool_names)} tool(s)"
                    if not c.alive:
                        tag += "  [DOWN]"
                elif self.all_specs.get(name) and self.all_specs[name].missing_env:
                    tag = f"available  (needs env: {', '.join(self.all_specs[name].missing_env)})"
                else:
                    tag = "available"
                log.write(Text(f"  {name:<16} {tag}", style="dim"))
            return
        if action == "connect" and len(parts) == 3:
            name = parts[2]
            spec = self.all_specs.get(name)
            if spec is None:
                log.write(Text(f"{name!r} not in config", style="red"))
                return
            if spec.missing_env:
                log.write(Text(
                    f"cannot connect {name!r}: env not set: {', '.join(spec.missing_env)}",
                    style="red"))
                return
            log.write(Text(f"connecting {name!r}…", style="dim"))
            try:
                n = await self.manager.connect(spec)
                log.write(Text(f"  ok  {name}: {n} tool(s) added", style="green"))
            except Exception as e:  # noqa: BLE001
                log.write(Text(f"  err  {e}", style="red"))
            self._refresh_status()
        elif action == "disconnect" and len(parts) == 3:
            try:
                await self.manager.disconnect(parts[2])
                log.write(Text(f"  disconnected {parts[2]!r}", style="dim"))
            except Exception as e:  # noqa: BLE001
                log.write(Text(f"  err  {e}", style="red"))
            self._refresh_status()
        elif action in ("connect", "disconnect"):
            connected = {c.spec.name for c in self.manager.connections}
            if action == "connect":
                choices = sorted(n for n in self.all_specs if n not in connected)
                hint = "connect" if choices else None
            else:
                choices = sorted(connected)
                hint = "disconnect" if choices else None
            if hint:
                log.write(Text(f"  /mcp {hint} NAME  —  "
                               + ", ".join(choices), style="dim"))
            else:
                log.write(Text(f"nothing to {action}", style="dim"))
        else:
            log.write(Text(
                "usage: /mcp [list] | /mcp connect NAME | /mcp disconnect NAME",
                style="red"))


async def run_tui(*, session: ChatSession, manager: ServerManager, all_specs: dict,
                  context_limit: int, history_path, model: str,
                  confirm_mode: str) -> None:
    app = ManishcodeApp(
        session=session, manager=manager, all_specs=all_specs,
        context_limit=context_limit, history_path=history_path, model=model,
        confirm_mode=confirm_mode,
    )
    await app.run_async()
