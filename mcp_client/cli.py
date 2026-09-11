"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .chat import (
    DEFAULT_MAX_TOOL_RESULT,
    DEFAULT_SYSTEM_PROMPT,
    ChatError,
    ChatSession,
    is_risky_tool,
)
from .config import (
    ConfigError,
    ServerSpec,
    drop_unavailable,
    find_config,
    load_config,
    select_servers,
    spec_from_cmdline,
)
from .debug import set_debug
from .history import load_history, save_history
from .servers import ServerManager
from .ui import context_line as _context_line
from .ui import make_ui

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion

    _HAS_PTK = True
except ImportError:  # fall back to plain input() if prompt_toolkit is missing
    _HAS_PTK = False


def _force_utf8_console() -> None:
    """Windows consoles default to cp1252; model output often contains emoji or
    other non-cp1252 characters. Re-encode stdout/stderr as UTF-8 and never let
    an un-encodable character crash the session."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


STARTER_CONFIG = """\
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    },
    "duckduckgo": {
      "command": "uvx",
      "args": ["duckduckgo-mcp-server"]
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@brave/brave-search-mcp-server", "--transport", "stdio"],
      "env": { "BRAVE_API_KEY": "${BRAVE_API_KEY}" }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}" }
    },
    "gmail": {
      "command": "npx",
      "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"]
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    }
  }
}
"""


def _cmd_init(argv: list[str]) -> int:
    """`manishcode init` - drop a starter config.json in the current folder."""
    if any(a in ("-h", "--help") for a in argv):
        print("usage: manishcode init [--force]\n\n"
              "Write a starter config.json (filesystem + duckduckgo) into the\n"
              "current folder. --force overwrites an existing one.")
        return 0
    force = "--force" in argv or "-f" in argv
    target = Path.cwd() / "config.json"
    if target.exists() and not force:
        print(f"{target} already exists. Use `manishcode init --force` to overwrite.")
        return 1
    try:
        target.write_text(STARTER_CONFIG, encoding="utf-8")
    except OSError as e:
        print(f"could not write {target}: {e}")
        return 1
    print(f"Wrote {target}")
    print("  A menu of common MCP servers. Nothing is connected until you run")
    print("  `manishcode --server NAME`, `--all`, or `/mcp connect NAME`.")
    print("  ('filesystem' is scoped to this folder; brave/github need an API key.)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="manishcode",
        description="Chat with a local Ollama model that can call MCP server tools.",
        epilog="run `manishcode init` first to create a starter config.json",
    )
    p.add_argument("--config", help="path to config.json (default: ./config.json)")
    p.add_argument(
        "--server",
        action="append",
        dest="servers",
        metavar="NAME",
        help="connect to this server from the config at startup (repeatable). "
        "By default nothing is connected - use this, --all, or /mcp connect.",
    )
    p.add_argument(
        "-a", "--all",
        action="store_true",
        dest="all_servers",
        help="connect to every server defined in the config at startup",
    )
    p.add_argument(
        "--server-cmd",
        action="append",
        dest="server_cmds",
        metavar="CMD",
        help="launch an MCP server by raw command instead of using the config "
        "(repeatable). e.g. --server-cmd \"npx -y @modelcontextprotocol/server-filesystem C:\\path\"",
    )
    p.add_argument("--model", default=None,
                   help="Ollama model. Default: the model already loaded in "
                        "`ollama ps`, else the largest installed model that "
                        "fits ~70%% of this machine's RAM.")
    p.add_argument(
        "--think",
        action="store_true",
        help="enable the model's thinking/reasoning mode",
    )
    p.add_argument("--debug", action="store_true", help="print full tool-call JSON")
    p.add_argument(
        "--plain",
        action="store_true",
        help="disable the styled (colour/markdown) output and use plain text",
    )
    p.add_argument(
        "--context",
        type=int,
        default=int(os.environ["OLLAMA_CONTEXT_LENGTH"])
        if os.environ.get("OLLAMA_CONTEXT_LENGTH") else None,
        metavar="TOKENS",
        help="override the context window used to show a 'context used' "
        "percentage after each turn. By default this is auto-detected from "
        "each model's own metadata and updates when you /model switch "
        "(default: $OLLAMA_CONTEXT_LENGTH if set, else auto-detect).",
    )
    p.add_argument(
        "--confirm-tools",
        choices=("risky", "all", "none"),
        default="risky",
        help="ask for y/N confirmation before a tool call runs: 'risky' (default) "
        "prompts only for side-effecting tools (send mail, trash, write/move "
        "files, ...); 'all' prompts for every call; 'none' never prompts",
    )
    p.add_argument(
        "--max-tool-result",
        type=int,
        default=DEFAULT_MAX_TOOL_RESULT,
        metavar="CHARS",
        help=f"trim any single tool result to this many chars before sending it "
        f"to the model, so one big file can't blow past the context window "
        f"(default: {DEFAULT_MAX_TOOL_RESULT}; 0 = unlimited)",
    )
    p.add_argument(
        "--history",
        metavar="FILE",
        help="load conversation from FILE at start and save back to it after every turn",
    )
    p.add_argument(
        "--list-servers",
        action="store_true",
        help="list servers defined in the config and exit",
    )
    p.add_argument(
        "--no-tools",
        action="store_true",
        help="pure chat: no servers AND no tool-oriented system prompt. "
        "(bare `manishcode` also starts no servers, but keeps the prompt so "
        "/mcp connect works well.)",
    )
    return p


def resolve_specs(args, all_specs: dict[str, ServerSpec]) -> list[ServerSpec]:
    """Which servers to start at launch. Bare `manishcode` starts none - the
    model gets tools only via --server, --all, --server-cmd, or /mcp connect."""
    if args.server_cmds:
        specs = [
            spec_from_cmdline(cmd, name=f"cmd{i+1}")
            for i, cmd in enumerate(args.server_cmds)
        ]
        if args.servers:
            print("[warn] --server is ignored when --server-cmd is used")
        return specs
    if args.all_servers:
        return drop_unavailable(list(all_specs.values()))
    if args.servers:
        return drop_unavailable(select_servers(all_specs, args.servers))
    return []


def _make_confirm(mode: str):
    """Build the confirm(name, args) callback the ChatSession calls before a
    tool runs. ``none`` -> no callback; ``all`` -> prompt every time;
    ``risky`` -> prompt only for side-effecting tools."""
    if mode == "none":
        return None

    async def confirm(name: str, tool_args: dict) -> bool:
        if mode == "risky" and not is_risky_tool(name):
            return True
        loop = asyncio.get_event_loop()
        preview = str(tool_args)
        if len(preview) > 300:
            preview = preview[:300] + "..."
        prompt = f"  [confirm] run {name}({preview}) ? [y/N] "
        answer = (await loop.run_in_executor(None, input, prompt)).strip().lower()
        return answer in ("y", "yes")

    return confirm


async def run(args) -> int:
    set_debug(args.debug)
    ui = make_ui(plain=args.plain or args.debug)

    if args.list_servers:
        cfg_path = find_config(args.config)
        if cfg_path is None:
            print("no config file found")
            return 1
        for name, spec in load_config(cfg_path).items():
            print(f"{name}: {spec.display_cmd}")
        return 0

    all_specs: dict[str, ServerSpec] = {}
    cfg_path = find_config(args.config)
    if cfg_path is None and not args.no_tools and not args.server_cmds:
        # No config yet: drop a starter one so `/mcp connect` has a catalogue.
        starter = Path.cwd() / "config.json"
        try:
            starter.write_text(STARTER_CONFIG, encoding="utf-8")
            cfg_path = starter
            print(f"No config found - wrote a starter {starter}")
        except OSError:
            pass
    if cfg_path is not None:
        try:
            all_specs = load_config(cfg_path)
        except ConfigError:
            pass

    specs = [] if args.no_tools else resolve_specs(args, all_specs)
    if specs:
        print(f"Starting {len(specs)} MCP server(s)...")
    elif not args.no_tools:
        catalogue = ", ".join(all_specs) or "none in config"
        print(f"No MCP servers started. Add tools with  /mcp connect NAME  "
              f"({catalogue}),\nor start with  --server NAME  /  --all.")

    history_path = Path(args.history).expanduser() if args.history else None
    initial_messages = []
    if history_path:
        try:
            initial_messages = load_history(history_path)
            if initial_messages:
                print(f"Loaded {len(initial_messages)} message(s) from {history_path}")
        except ValueError as e:
            print(f"[warn] {e}; starting fresh")

    confirm_cb = _make_confirm(args.confirm_tools)

    def on_change(messages):
        if history_path:
            try:
                save_history(history_path, messages, session.model)
            except OSError as e:
                print(f"[warn] could not save history: {e}")

    use_tui = not args.plain and not args.debug
    try:
        use_tui = use_tui and sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        use_tui = False
    if use_tui:
        try:
            from .tui import run_tui
        except ImportError:
            use_tui = False

    try:
        async with ServerManager(specs) as manager:
            if not use_tui:
                print(f"\n{len(manager.ollama_tools)} tool(s) available:")
                print(manager.describe_tools())
            try:
                session = ChatSession(
                    manager,
                    model=args.model,
                    think=args.think,
                    messages=initial_messages,
                    on_change=on_change if history_path else None,
                    max_tool_result=args.max_tool_result,
                    confirm=None if use_tui else confirm_cb,
                    system_prompt=None if args.no_tools else DEFAULT_SYSTEM_PROMPT,
                    ui=ui,
                    context_length=args.context,
                )
            except ChatError as e:
                print(f"\nERROR: {e}")
                return 1

            args.model = session.model  # resolve auto-pick for status + history

            if use_tui:
                await run_tui(
                    session=session, manager=manager, all_specs=all_specs,
                    history_path=history_path,
                    model=args.model, confirm_mode=args.confirm_tools,
                )
            else:
                ui.rule(
                    f"{args.model}  think={args.think}  "
                    f"confirm-tools={args.confirm_tools}  context={session.context_length:,}"
                )
                ui.info("Type your message, or / for the list of commands.")
                await chat_repl(session, manager, history_path, args.model,
                                all_specs=all_specs,
                                ui=ui)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return 1

    return 0


# name(s) -> (help text). First name is canonical; the rest are aliases.
SLASH_COMMANDS: list[tuple[tuple[str, ...], str]] = [
    (("/help", "/", "/?"), "show this list of commands"),
    (("/new", "/reset", "/clear"), "start a fresh conversation (clear history)"),
    (("/tools",), "list connected tools and which server owns them"),
    (("/model", "/model NAME"), "list installed models; /model NAME to switch | /model pull NAME to download"),
    (("/mcp",), "list config servers; /mcp connect NAME | /mcp disconnect NAME"),
    # (works in --no-tools too: start bare, then /mcp connect what you need)
    (("/context",), "show how many tokens the conversation is using"),
    (("/save", "/save FILE"), "write the conversation to FILE (or the --history file)"),
    (("/quit", "/exit"), "leave"),
]


def _print_commands() -> None:
    print("Commands:")
    for names, help_text in SLASH_COMMANDS:
        print(f"  {', '.join(names):<24} {help_text}")


# (name, meta) pairs shown in the as-you-type dropdown.
_COMPLETION_ITEMS = [
    ("/help", "show the command list"),
    ("/new", "start a fresh conversation"),
    ("/reset", "alias for /new"),
    ("/clear", "alias for /new"),
    ("/tools", "list connected tools"),
    ("/model", "list / switch / pull Ollama models"),
    ("/mcp", "list / connect / disconnect MCP servers"),
    ("/context", "show token usage"),
    ("/save", "write the conversation to a file"),
    ("/quit", "exit"),
    ("/exit", "alias for /quit"),
]

if _HAS_PTK:

    class _SlashCompleter(Completer):
        """Offer slash-command completions, but only while the line is a bare
        ``/word`` with no space yet."""

        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/") or " " in text:
                return
            for name, meta in _COMPLETION_ITEMS:
                if name.startswith(text.lower()):
                    yield Completion(
                        name, start_position=-len(text),
                        display=name, display_meta=meta,
                    )


def _make_reader():
    """Return an async ``read(prompt) -> str``. Uses prompt_toolkit (live
    command dropdown) on a real terminal; falls back to plain input() when
    prompt_toolkit is missing, stdin is piped, or the console can't host it."""
    loop = asyncio.get_event_loop()

    async def plain_read(prompt: str) -> str:
        return (await loop.run_in_executor(None, input, prompt)).strip()

    if not _HAS_PTK or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return plain_read

    ptk: PromptSession = PromptSession(
        completer=_SlashCompleter(), complete_while_typing=True
    )
    state = {"ok": True}

    async def ptk_read(prompt: str) -> str:
        if state["ok"]:
            try:
                return (await ptk.prompt_async(prompt)).strip()
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:  # noqa: BLE001 - console can't host prompt_toolkit
                state["ok"] = False
                print("[warn] rich prompt unavailable; using a plain prompt")
        return await plain_read(prompt)

    return ptk_read


def _print_mcp(manager, all_specs: dict) -> None:
    connected = {c.spec.name: c for c in getattr(manager, "connections", [])}
    names = sorted(set(all_specs) | set(connected))
    if not names:
        print("no servers in config, none connected")
        return
    for name in names:
        if name in connected:
            c = connected[name]
            tag = f"connected  {len(c.tool_names)} tool(s)"
            if not c.alive:
                tag += "  [DOWN]"
        elif name in all_specs and all_specs[name].missing_env:
            tag = f"available  (needs env: {', '.join(all_specs[name].missing_env)})"
        else:
            tag = "available"
        print(f"  {name:<16} {tag}")
    print("  /mcp connect NAME  |  /mcp disconnect NAME")


async def _handle_mcp(user_input: str, manager, all_specs: dict) -> None:
    parts = user_input.split()
    action = parts[1].lower() if len(parts) > 1 else "list"
    if action in ("list", "ls", "status"):
        _print_mcp(manager, all_specs)
        return
    if action == "connect" and len(parts) == 3:
        name = parts[2]
        spec = all_specs.get(name)
        if spec is None:
            print(f"{name!r} not in config (have: {', '.join(all_specs) or 'none'})")
            return
        if spec.missing_env:
            print(f"cannot connect {name!r}: env not set: {', '.join(spec.missing_env)}")
            return
        print(f"connecting {name!r}...")
        try:
            n = await manager.connect(spec)
            print(f"  [ok] {name}: {n} tool(s) added")
        except Exception as e:  # noqa: BLE001
            print(f"  [error] could not connect {name!r}: {e}")
    elif action == "disconnect" and len(parts) == 3:
        name = parts[2]
        try:
            await manager.disconnect(name)
            print(f"  disconnected {name!r}")
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {e}")
    elif action in ("connect", "disconnect"):
        connected = {c.spec.name for c in manager.connections}
        if action == "connect":
            choices = sorted(n for n in all_specs if n not in connected)
        else:
            choices = sorted(connected)
        if choices:
            print(f"  /mcp {action} NAME  ->  {', '.join(choices)}")
        else:
            print(f"  nothing to {action}")
    else:
        print("usage: /mcp [list]  |  /mcp connect NAME  |  /mcp disconnect NAME")


async def _handle_model(user_input: str, session: ChatSession) -> None:
    parts = user_input.split()
    if len(parts) == 1:
        for name in session.list_models():
            mark = "* " if name == session.model else "  "
            print(f"{mark}{name}")
        print("  /model NAME to switch  |  /model pull NAME to download")
        return
    if parts[1].lower() == "pull" and len(parts) == 3:
        print(f"pulling {parts[2]!r}... (may take a while)")
        try:
            await session.pull_model(parts[2])
        except ChatError as e:
            print(f"  [error] {e}")
            return
        print(f"  [ok] now using {session.model}")
        return
    if len(parts) == 2:
        try:
            session.set_model(parts[1])
        except ChatError as e:
            print(f"  [error] {e}")
            return
        print(f"  now using {session.model}")
        return
    print("usage: /model  |  /model NAME  |  /model pull NAME")


async def chat_repl(session: ChatSession, manager: ServerManager, history_path, model,
                    all_specs: dict | None = None, ui=None):
    all_specs = all_specs or {}
    ui = ui or make_ui(plain=True)
    read = _make_reader()
    while True:
        try:
            user_input = await read(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split(maxsplit=1)[0].lower()
            if cmd in ("/help", "/", "/?"):
                _print_commands()
                continue
            if cmd in ("/quit", "/exit"):
                break
            if cmd in ("/new", "/reset", "/clear"):
                keep = [m for m in session.messages
                        if isinstance(m, dict) and m.get("role") == "system"]
                session.messages[:] = keep
                print("(new conversation)")
                continue
            if cmd == "/tools":
                print(manager.describe_tools())
                continue
            if cmd == "/model":
                await _handle_model(user_input, session)
                continue
            if cmd == "/mcp":
                await _handle_mcp(user_input, manager, all_specs)
                continue
            if cmd == "/context":
                print(_context_line(session, session.context_length))
                continue
            if cmd == "/save":
                parts = user_input.split(maxsplit=1)
                target = Path(parts[1]).expanduser() if len(parts) > 1 else history_path
                if not target:
                    print("usage: /save FILE")
                    continue
                save_history(target, session.messages, session.model)
                print(f"saved to {target}")
                continue
            print(f"unknown command {cmd!r}. Type / for the list.")
            continue

        ui.user_turn(user_input)
        try:
            answer = await session.send(user_input)
        except ChatError as e:
            ui.warn(f"[chat error] {e}  -  you can keep going or /quit")
            continue

        ui.assistant(answer)
        ui.info(_context_line(session, session.context_length))

        dead = await manager.health_check()
        if dead:
            ui.warn(f"[warn] server(s) not responding: {', '.join(dead)}. "
                    "Their tools may fail until you restart.")


def main() -> None:
    _force_utf8_console()
    argv = sys.argv[1:]
    if argv and argv[0] == "init":
        sys.exit(_cmd_init(argv[1:]))
    args = build_parser().parse_args()
    try:
        rc = asyncio.run(run(args))
    except ConfigError as e:
        print(f"config error: {e}")
        rc = 2
    except KeyboardInterrupt:
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
