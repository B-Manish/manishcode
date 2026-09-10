"""Console output.

``PlainUI`` produces exactly the line-oriented output this tool has always
printed (so piped/non-TTY use and tests are unchanged). ``RichUI`` styles the
same events - coloured role rules, markdown answers, tidy tool blocks, a
"thinking" spinner - for an interactive terminal.
"""

from __future__ import annotations

import contextlib
import sys

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text

    _HAS_RICH = True
except ImportError:  # pragma: no cover - rich is a declared dependency
    _HAS_RICH = False


class PlainUI:
    """The original output: plain ``print`` calls, no styling."""

    def user_turn(self, text: str) -> None:
        pass

    def assistant(self, text: str) -> None:
        print(f"\n{text}\n")

    def tool_call(self, name: str, args: str) -> None:
        print(f"  [tool] {name}({args})")

    def tool_result(self, name: str, result: str) -> None:
        print(f"  [tool] {name} -> {result}")

    def tool_error(self, msg: str) -> None:
        print(f"  [tool error] {msg}")

    def tool_note(self, msg: str) -> None:
        print(f"  [tool] {msg}")

    def info(self, msg: str) -> None:
        print(msg)

    def warn(self, msg: str) -> None:
        print(msg)

    def rule(self, msg: str) -> None:
        print(msg)

    @contextlib.contextmanager
    def thinking(self, label: str = "thinking"):
        yield


class RichUI(PlainUI):
    """Styled output for an interactive terminal. Text is passed as ``Text``
    (never markup strings) so literal ``[tool]``-style brackets in messages
    aren't parsed as styles."""

    def __init__(self) -> None:
        self.console = Console()

    def user_turn(self, text: str) -> None:
        self.console.print()
        self.console.rule(Text("you", style="bold cyan"), align="left",
                          style="cyan", characters="-")

    def assistant(self, text: str) -> None:
        self.console.print()
        self.console.rule(Text("assistant", style="bold green"), align="left",
                          style="green", characters="-")
        self.console.print(Markdown(text or "_(no response)_"))
        self.console.print()

    def tool_call(self, name: str, args: str) -> None:
        self.console.print(Text.assemble(
            ("  > ", "yellow"), (name, "bold"), (f"({args})", "dim")))

    def tool_result(self, name: str, result: str) -> None:
        self.console.print(Text.assemble(("    ok  ", "green"), (result, "dim")))

    def tool_error(self, msg: str) -> None:
        self.console.print(Text(f"    err  {msg}", style="red"))

    def tool_note(self, msg: str) -> None:
        self.console.print(Text(f"    .. {msg}", style="dim"))

    def info(self, msg: str) -> None:
        self.console.print(Text(msg, style="dim"))

    def warn(self, msg: str) -> None:
        self.console.print(Text(msg, style="yellow"))

    def rule(self, msg: str) -> None:
        self.console.rule(Text(msg, style="dim"), style="dim", characters="-")

    @contextlib.contextmanager
    def thinking(self, label: str = "thinking"):
        with self.console.status(Text(f"{label}...", style="dim"), spinner="dots"):
            yield


class TuiUI(PlainUI):
    """Routes chat events into the Textual app's transcript. Methods are called
    from the event loop (ChatSession offloads the blocking Ollama call), so they
    can touch widgets directly."""

    def __init__(self, app) -> None:
        self.app = app

    def assistant(self, text: str) -> None:
        self.app.tui_assistant(text)

    def tool_call(self, name: str, args: str) -> None:
        self.app.tui_write(Text.assemble(
            ("  > ", "yellow"), (name, "bold"), (f"({args})", "dim")))

    def tool_result(self, name: str, result: str) -> None:
        self.app.tui_write(Text.assemble(("    ok  ", "green"), (result, "dim")))

    def tool_error(self, msg: str) -> None:
        self.app.tui_write(Text(f"    err  {msg}", style="red"))

    def tool_note(self, msg: str) -> None:
        self.app.tui_write(Text(f"    .. {msg}", style="dim"))

    def info(self, msg: str) -> None:
        self.app.tui_write(Text(msg, style="dim"))

    def warn(self, msg: str) -> None:
        self.app.tui_write(Text(msg, style="yellow"))

    def rule(self, msg: str) -> None:
        self.app.tui_write(Text(msg, style="dim"))

    @contextlib.contextmanager
    def thinking(self, label: str = "thinking"):
        # The app shows/hides the loading indicator for the whole turn; this
        # hook just nudges the status label between model / tool phases.
        self.app.tui_phase(label)
        try:
            yield
        finally:
            self.app.tui_phase("")


def context_line(session, limit: int) -> str:
    """One-line summary of how full the context window is, printed after a turn."""
    used = getattr(session, "prompt_tokens", 0)
    if not used:
        return "[context] no data yet"
    msg = f"[context] {used:,} prompt tokens in use"
    if getattr(session, "eval_tokens", 0):
        msg += f" (+{session.eval_tokens:,} generated last step)"
    if limit > 0:
        pct = used * 100 / limit
        msg += f"  ~{pct:.0f}% of {limit:,}"
        if pct >= 90:
            msg += "  [!] near limit - Ollama will start dropping oldest messages"
        elif pct >= 75:
            msg += "  [!] getting full; consider /reset"
    return msg


def make_ui(plain: bool = False) -> PlainUI:
    """RichUI on an interactive terminal; PlainUI when piped, dumb, or --plain."""
    if plain or not _HAS_RICH:
        return PlainUI()
    try:
        if not sys.stdout.isatty():
            return PlainUI()
    except (AttributeError, ValueError):
        return PlainUI()
    return RichUI()
