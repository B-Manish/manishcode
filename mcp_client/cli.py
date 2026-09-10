"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from .chat import (
    DEFAULT_MAX_TOOL_RESULT,
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


def _force_utf8_console() -> None:
    """Windows consoles default to cp1252; model output often contains emoji or
    other non-cp1252 characters. Re-encode stdout/stderr as UTF-8 and never let
    an un-encodable character crash the session."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcp-ollama",
        description="Chat with a local Ollama model that can call MCP server tools.",
    )
    p.add_argument("--config", help="path to config.json (default: ./config.json)")
    p.add_argument(
        "--server",
        action="append",
        dest="servers",
        metavar="NAME",
        help="name of a server from the config to connect to (repeatable; "
        "default: all servers in the config)",
    )
    p.add_argument(
        "--server-cmd",
        action="append",
        dest="server_cmds",
        metavar="CMD",
        help="launch an MCP server by raw command instead of using the config "
        "(repeatable). e.g. --server-cmd \"npx -y @modelcontextprotocol/server-filesystem C:\\path\"",
    )
    p.add_argument("--model", default="qwen3:8b", help="Ollama model (default: qwen3:8b)")
    p.add_argument(
        "--think",
        action="store_true",
        help="enable the model's thinking/reasoning mode",
    )
    p.add_argument("--debug", action="store_true", help="print full tool-call JSON")
    p.add_argument(
        "--context",
        type=int,
        default=int(os.environ.get("OLLAMA_CONTEXT_LENGTH") or 4096),
        metavar="TOKENS",
        help="effective Ollama context window, used only to show a 'context "
        "used' percentage after each turn (default: $OLLAMA_CONTEXT_LENGTH or "
        "4096). Set this to whatever you started `ollama serve` with.",
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
    return p


def resolve_specs(args) -> list[ServerSpec]:
    if args.server_cmds:
        specs = [
            spec_from_cmdline(cmd, name=f"cmd{i+1}")
            for i, cmd in enumerate(args.server_cmds)
        ]
        if args.servers:
            print("[warn] --server is ignored when --server-cmd is used")
        return specs

    cfg_path = find_config(args.config)
    if cfg_path is None:
        raise ConfigError(
            "no config file found (looked for ./config.json). "
            "Pass --config, or use --server-cmd."
        )
    all_specs = load_config(cfg_path)
    chosen = select_servers(all_specs, args.servers)
    available = drop_unavailable(chosen)
    if not available:
        raise ConfigError(
            "no servers left to start (all selected servers were skipped for "
            "missing environment variables)"
        )
    return available


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

    if args.list_servers:
        cfg_path = find_config(args.config)
        if cfg_path is None:
            print("no config file found")
            return 1
        for name, spec in load_config(cfg_path).items():
            print(f"{name}: {spec.display_cmd}")
        return 0

    specs = resolve_specs(args)
    print(f"Starting {len(specs)} MCP server(s)...")

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
                save_history(history_path, messages, args.model)
            except OSError as e:
                print(f"[warn] could not save history: {e}")

    try:
        async with ServerManager(specs) as manager:
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
                    confirm=confirm_cb,
                )
            except ChatError as e:
                print(f"\nERROR: {e}")
                return 1

            print(
                f"\nModel: {args.model}  think={args.think}  debug={args.debug}  "
                f"max-tool-result={args.max_tool_result or 'unlimited'}  "
                f"confirm-tools={args.confirm_tools}  context={args.context:,}\n"
                "Type your message. Commands: /quit, /reset, /tools, /context, /save FILE\n"
            )
            await chat_repl(session, manager, history_path, args.model,
                            context_limit=args.context)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        return 1

    return 0


def _context_line(session: ChatSession, limit: int) -> str:
    used = session.prompt_tokens
    if not used:
        return "[context] no data yet"
    msg = f"[context] {used:,} prompt tokens in use"
    if session.eval_tokens:
        msg += f" (+{session.eval_tokens:,} generated last step)"
    if limit > 0:
        pct = used * 100 / limit
        msg += f"  ~{pct:.0f}% of {limit:,}"
        if pct >= 90:
            msg += "  [!] near limit - Ollama will start dropping oldest messages"
        elif pct >= 75:
            msg += "  [!] getting full; consider /reset"
    return msg


async def chat_repl(session: ChatSession, manager: ServerManager, history_path, model,
                    context_limit: int = 0):
    loop = asyncio.get_event_loop()
    while True:
        try:
            user_input = (await loop.run_in_executor(None, input, ">>> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/reset":
            session.messages.clear()
            print("(history cleared)")
            continue
        if user_input == "/tools":
            print(manager.describe_tools())
            continue
        if user_input == "/context":
            print(_context_line(session, context_limit))
            continue
        if user_input.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            target = Path(parts[1]).expanduser() if len(parts) > 1 else history_path
            if not target:
                print("usage: /save FILE")
                continue
            save_history(target, session.messages, model)
            print(f"saved to {target}")
            continue

        try:
            answer = await session.send(user_input)
        except ChatError as e:
            print(f"\n[chat error] {e}\nYou can keep going or /quit.\n")
            continue

        print(f"\n{answer}\n")
        print(_context_line(session, context_limit))

        dead = await manager.health_check()
        if dead:
            print(f"[warn] server(s) not responding: {', '.join(dead)}. "
                  "Their tools may fail until you restart.\n")


def main() -> None:
    _force_utf8_console()
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
