"""
Smoke test for mcp_client: exercises every feature against the real filesystem +
memory MCP servers and your local Ollama.

Run:
    uv run python smoke_test.py

Requires: Node.js (for npx), Ollama running with the model pulled (default qwen3:8b,
override with:  uv run python smoke_test.py --model llama3.1:8b).

Each check prints [PASS] or [FAIL]; exit code is non-zero if anything failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from mcp_client.chat import ChatError, ChatSession
from mcp_client.cli import _force_utf8_console
from mcp_client.config import (
    ConfigError,
    ServerSpec,
    drop_unavailable,
    load_config,
    select_servers,
)
from mcp_client.history import load_history, save_history
from mcp_client.servers import ServerManager, ToolCallError

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def fs_spec(root: str, name: str = "fs") -> ServerSpec:
    return ServerSpec(name, "npx", ["-y", "@modelcontextprotocol/server-filesystem", root])


async def test_config(tmp: Path) -> None:
    print("\n== config file ==")
    try:
        load_config(tmp / "nope.json")
        check("missing config raises ConfigError", False)
    except ConfigError:
        check("missing config raises ConfigError", True)

    bad = tmp / "bad.json"
    bad.write_text("{not json")
    try:
        load_config(bad)
        check("invalid JSON raises ConfigError", False)
    except ConfigError:
        check("invalid JSON raises ConfigError", True)

    good = tmp / "config.json"
    good.write_text(json.dumps({"mcpServers": {
        "a": {"command": "npx", "args": ["x"]},
        "b": {"command": "npx", "args": ["y"], "env": {"K": "V"}},
    }}))
    specs = load_config(good)
    check("parses mcpServers", set(specs) == {"a", "b"}, str(list(specs)))
    check("env parsed", specs["b"].env == {"K": "V"})
    check("select one server", [s.name for s in select_servers(specs, ["b"])] == ["b"])
    try:
        select_servers(specs, ["ghost"])
        check("unknown server name raises", False)
    except ConfigError:
        check("unknown server name raises", True)

    import os as _os

    keyed = tmp / "keyed.json"
    keyed.write_text(json.dumps({"mcpServers": {
        "plain": {"command": "npx", "args": ["x"]},
        "needskey": {"command": "npx", "args": ["y"],
                     "env": {"API_KEY": "${SMOKE_TEST_MISSING_KEY}"}},
    }}))
    kspecs = load_config(keyed)
    check("server with unset ${VAR} reported missing",
          kspecs["needskey"].missing_env == ["SMOKE_TEST_MISSING_KEY"])
    avail = drop_unavailable(list(kspecs.values()), warn=lambda *_: None)
    check("drop_unavailable skips the keyless server",
          [s.name for s in avail] == ["plain"])
    _os.environ["SMOKE_TEST_MISSING_KEY"] = "sekret"
    try:
        check("${VAR} expands from env once set",
              kspecs["needskey"].missing_env == []
              and kspecs["needskey"].full_env()["API_KEY"] == "sekret")
    finally:
        del _os.environ["SMOKE_TEST_MISSING_KEY"]

    from mcp_client.chat import is_risky_tool
    check("side-effecting tools flagged risky",
          all(is_risky_tool(t) for t in
              ("send_email", "trash_message", "write_file", "move_file")))
    check("read-only tools not flagged risky",
          not any(is_risky_tool(t) for t in
                  ("search", "read_text_file", "list_directory", "get_thread")))


async def test_ui() -> None:
    print("\n== console UI ==")
    from mcp_client.ui import PlainUI, RichUI, make_ui

    check("make_ui returns PlainUI when piped (not a tty)",
          isinstance(make_ui(), PlainUI))
    check("make_ui(plain=True) is PlainUI", type(make_ui(plain=True)) is PlainUI)
    # RichUI must render literal bracket text without choking on it as markup
    try:
        RichUI().info("[context] 40 tokens [!] full")
        RichUI().assistant("**bold** and a list\n- a\n- b")
        check("RichUI renders bracketed / markdown text", True)
    except Exception as e:  # noqa: BLE001
        check("RichUI renders bracketed / markdown text", False, repr(e))


async def test_tui() -> None:
    print("\n== Textual TUI (headless) ==")
    from textual.widgets import RichLog

    from mcp_client.tui import ConfirmScreen, ManishcodeApp, _make_confirm

    class FakeConn:
        class spec:  # noqa: D106
            name = "duckduckgo"
        tool_names = ["search"]
        alive = True

    class FakeManager:
        ollama_tools = [{"function": {"name": "search"}}]
        connections = [FakeConn()]

        def describe_tools(self):
            return "  - search (duckduckgo)"

        async def health_check(self):
            return []

    class FakeSession:
        prompt_tokens = 0
        eval_tokens = 0
        messages = [{"role": "system", "content": "x"}]
        _confirm = None
        ui = None
        confirmed = None

        async def send(self, text):
            self.prompt_tokens = 900
            if self._confirm:
                self.confirmed = await self._confirm("write_file", {"p": "x"})
            return "**hi** from a fake model\n- a\n- b"

    sess = FakeSession()
    app = ManishcodeApp(session=sess, manager=FakeManager(), all_specs={},
                        context_limit=4096, history_path=None, model="m",
                        confirm_mode="risky")
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#prompt").value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            app.query_one("#prompt").value = "hello"
            await pilot.press("enter")
            await pilot.pause(0.3)
            # a risky tool inside the turn should raise the confirm modal
            check("confirm modal shown for risky tool",
                  isinstance(app.screen, ConfirmScreen))
            await pilot.press("y")
            await pilot.pause(0.3)
            check("modal answer reaches the session", sess.confirmed is True)
            log_text = " ".join(str(s) for s in app.query_one(RichLog).lines)
            check("assistant reply rendered", "fake model" in log_text)
            check("status shows context after a turn", "ctx 21%" in
                  str(app.query_one("#status").render()))
            app.query_one("#prompt").value = "/quit"
            await pilot.press("enter")
            await pilot.pause()
        check("TUI ran and exited cleanly", True)
    except Exception as e:  # noqa: BLE001
        check("TUI ran and exited cleanly", False, repr(e))


async def test_no_tools() -> None:
    print("\n== --no-tools / empty ServerManager ==")
    async with ServerManager([]) as mgr:
        check("empty spec list starts cleanly (no RuntimeError)", True)
        check("no tools", mgr.ollama_tools == [])
        check("health_check is a no-op", await mgr.health_check() == [])
        try:
            await mgr.call_tool("anything", {})
            check("call_tool on unknown tool raises", False)
        except ToolCallError:
            check("call_tool on unknown tool raises", True)


async def test_routing(tmp: Path) -> None:
    print("\n== multi-server routing ==")
    d1, d2 = tmp / "s1", tmp / "s2"
    d1.mkdir(); d2.mkdir()
    (d1 / "a.txt").write_text("ALPHA")
    (d2 / "b.txt").write_text("BRAVO")

    async with ServerManager([fs_spec(str(d1), "one"), fs_spec(str(d2), "two")]) as mgr:
        names = [t["function"]["name"] for t in mgr.ollama_tools]
        check("both servers connected", len(mgr.connections) == 2)
        check("tool-name collision renamed", any(n.startswith("two__") for n in names))

        r1 = await mgr.call_tool("read_text_file", {"path": str(d1 / "a.txt")})
        check("call routed to server 'one'", "ALPHA" in r1, r1)

        two_read = next(n for n in names if n.startswith("two__") and "read_text_file" in n)
        r2 = await mgr.call_tool(two_read, {"path": str(d2 / "b.txt")})
        check("call routed to server 'two'", "BRAVO" in r2, r2)

        try:
            await mgr.call_tool("read_text_file", {"path": str(d2 / "b.txt")})
            check("server isolation enforced", False)
        except ToolCallError:
            check("server isolation enforced", True)

        try:
            await mgr.call_tool("bogus_tool", {})
            check("unknown tool -> ToolCallError", False)
        except ToolCallError:
            check("unknown tool -> ToolCallError", True)


async def test_runtime_connect(tmp: Path) -> None:
    print("\n== runtime /mcp connect + disconnect ==")
    d1, d2 = tmp / "rc1", tmp / "rc2"
    d1.mkdir(); d2.mkdir()
    (d2 / "z.txt").write_text("ZULU")

    async with ServerManager([fs_spec(str(d1), "a")]) as mgr:
        check("starts with one server", len(mgr.connections) == 1)
        base_tools = len(mgr.ollama_tools)

        added = await mgr.connect(fs_spec(str(d2), "b"))
        check("connect() adds a server", len(mgr.connections) == 2 and added > 0)
        check("connect() merges tools", len(mgr.ollama_tools) == base_tools + added)

        # "b" collides with "a" on tool names, so its tools are exposed b__*
        b_read = "b__read_text_file"
        names = [t["function"]["name"] for t in mgr.ollama_tools]
        check("new server's tools exposed under its name", b_read in names)
        r = await mgr.call_tool(b_read, {"path": str(d2 / "z.txt")})
        check("new server's tool routes", "ZULU" in r, r)

        try:
            await mgr.connect(fs_spec(str(d1), "a"))
            check("connect() rejects duplicate name", False)
        except ValueError:
            check("connect() rejects duplicate name", True)

        await mgr.disconnect("b")
        check("disconnect() drops the server", len(mgr.connections) == 1)
        check("disconnect() drops its tools", len(mgr.ollama_tools) == base_tools)
        try:
            await mgr.call_tool(b_read, {"path": str(d2 / "z.txt")})
            check("disconnected tool is gone", False)
        except ToolCallError:
            check("disconnected tool is gone", True)


async def test_tool_result_cap(tmp: Path) -> None:
    print("\n== tool result cap ==")
    from mcp_client.chat import _cap_tool_result

    small, dropped = _cap_tool_result("x" * 100, 8000)
    check("short result untouched", small == "x" * 100 and dropped == 0)

    big, dropped = _cap_tool_result("y" * 20000, 8000)
    check("long result trimmed to limit + marker", dropped == 12000 and big.startswith("y" * 8000))
    check("marker explains the cut", "truncated 12000 of 20000 chars" in big)

    off, dropped = _cap_tool_result("z" * 20000, 0)
    check("limit 0 disables trimming", off == "z" * 20000 and dropped == 0)

    d = tmp / "cap"
    d.mkdir()
    (d / "big.txt").write_text("A" * 50000)
    async with ServerManager([fs_spec(str(d))]) as mgr:
        try:
            sess = ChatSession(mgr, model="qwen3:8b", max_tool_result=5000)
        except ChatError as e:
            check("cap: Ollama available", False, str(e))
            return
        await sess.send(f"Use read_text_file on {d / 'big.txt'} and say 'done'.")
        tool_msg = next(m for m in sess.messages if isinstance(m, dict) and m.get("role") == "tool")
        check("oversized tool result capped in context", len(tool_msg["content"]) < 6000,
              f"{len(tool_msg['content'])} chars")


async def test_bad_server() -> None:
    print("\n== bad server handling ==")
    try:
        async with ServerManager([ServerSpec("broken", "this-cmd-does-not-exist", [])]):
            check("all-servers-failed raises RuntimeError", False)
    except RuntimeError:
        check("all-servers-failed raises RuntimeError", True)
    except Exception as e:  # noqa: BLE001
        check("all-servers-failed raises RuntimeError", False, repr(e))


async def test_chat(tmp: Path, model: str) -> None:
    print(f"\n== chat + tools + history  (model: {model}) ==")
    d = tmp / "chat"
    d.mkdir()
    (d / "fact.txt").write_text("The secret animal is OTTER.")
    hist = tmp / "hist.json"

    async with ServerManager([fs_spec(str(d))]) as mgr:
        try:
            sess = ChatSession(mgr, model=model, think=False,
                               on_change=lambda m: save_history(hist, m, model))
        except ChatError as e:
            check("Ollama reachable + model present", False, str(e))
            return
        check("Ollama reachable + model present", True)

        ans = await sess.send(
            f"Use read_text_file to read {d / 'fact.txt'} and tell me the secret animal."
        )
        check("model called tool and used the result", "OTTER" in ans.upper(), ans[:120])
        check("history file written", hist.is_file())
        check("history round-trips", len(load_history(hist)) >= 3)

        ans2 = await sess.send(
            "Use read_text_file on C:/Windows/does-not-exist-xyz.txt . "
            "If it fails, just tell me it failed in one sentence."
        )
        check("failed tool call does not crash session", isinstance(ans2, str) and len(ans2) > 0)

        try:
            ChatSession(mgr, model="totally-not-a-real-model:0b")
            check("missing model raises ChatError", False)
        except ChatError:
            check("missing model raises ChatError", True)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--skip-ollama", action="store_true",
                    help="run only the checks that don't need Ollama")
    args = ap.parse_args()
    _force_utf8_console()

    with tempfile.TemporaryDirectory(prefix="mcp_smoke_") as td:
        tmp = Path(td)
        await test_config(tmp)
        await test_ui()
        await test_tui()
        await test_no_tools()
        await test_routing(tmp)
        await test_runtime_connect(tmp)
        await test_bad_server()
        if not args.skip_ollama:
            await test_tool_result_cap(tmp)
        if not args.skip_ollama:
            await test_chat(tmp, args.model)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 50}\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
