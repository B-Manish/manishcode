"""The interactive chat loop: user <-> Ollama <-> MCP tools."""

from __future__ import annotations

import asyncio
import functools

import ollama

from .debug import debug, is_debug
from .servers import ServerManager, ToolCallError
from .ui import PlainUI

MAX_TOOL_ROUNDS = 25  # guard against a model that loops forever on tools
DEFAULT_MAX_TOOL_RESULT = 8000  # chars of a tool result fed to the model (0 = unlimited)

# Local models tend to answer "I don't have real-time access" instead of reaching
# for a tool. This nudges them to actually call the tools they've been given.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to external tools, which may include "
    "web search, page fetching, file access, and more. You do NOT have built-in "
    "knowledge of current events, today's date-specific facts, weather, prices, "
    "or news.\n"
    "Rules for current or external information:\n"
    "1. Call a web search tool first. Ask for several results (e.g. max_results "
    "5), not just one.\n"
    "2. If the search snippets do not already contain the specific answer "
    "(a temperature, a number, a headline, a quote), call the page-fetch tool "
    "on the most relevant result URL to read the actual content before "
    "answering.\n"
    "3. Never reply that you 'cannot fetch live data' or tell the user to visit "
    "a link themselves - use the fetch tool to get the data. Only give up after "
    "the tools genuinely return nothing useful, and then say what you tried.\n"
    "4. Base the final answer on tool results and mention the source.\n"
    "If a Gmail tool is available: it returns messages newest-first. For 'recent' "
    "or 'last N' mail, pass query 'in:inbox' (or an empty query) with maxResults "
    "= N. Do NOT invent date syntax like 'after:last 5'. Real Gmail operators are "
    "from:, to:, subject:, is:unread, has:attachment, newer_than:7d, older_than:1m."
)


class ChatError(Exception):
    pass


def _cap_tool_result(text: str, limit: int) -> tuple[str, int]:
    """Trim an over-long tool result before it goes into the model's context.

    A single big file (a 33 KB README, a directory tree, an API dump) can blow
    past a small local context window; the model then silently loses the tool
    definitions and the question. We keep the head and tell the model plainly
    that it was cut and how to get the rest.
    """
    if limit <= 0 or len(text) <= limit:
        return text, 0
    dropped = len(text) - limit
    marker = (
        f"\n\n[... truncated {dropped} of {len(text)} chars. "
        f"Ask for a smaller slice: a specific file, a sub-path, fewer items, "
        f"or the fetch tool on a download URL. ...]"
    )
    return text[:limit] + marker, dropped


# Substrings that mark a tool as having a real-world side effect (sending mail,
# deleting/trashing, writing or moving files, ...). With --confirm-tools risky
# (the default) the user is asked before any such call runs.
RISKY_TOOL_SUBSTRINGS = (
    "send", "trash", "delete", "remove", "forward", "reply", "draft",
    "write", "edit", "append", "move", "rename", "create", "update",
    "modify", "archive", "label", "mark", "batch", "put", "post",
)


def is_risky_tool(name: str) -> bool:
    n = name.lower()
    return any(s in n for s in RISKY_TOOL_SUBSTRINGS)


def _check_ollama(client: ollama.Client, model: str) -> None:
    try:
        names = [m.model for m in client.list().models]
    except Exception as e:  # noqa: BLE001
        raise ChatError(
            f"cannot reach Ollama ({e}). Is `ollama serve` running?"
        ) from e
    if model not in names and f"{model}:latest" not in names:
        raise ChatError(
            f"model {model!r} not found in Ollama. Pull it with `ollama pull {model}`.\n"
            f"Available: {', '.join(names) or 'none'}"
        )


class ChatSession:
    def __init__(
        self,
        manager: ServerManager,
        model: str,
        think: bool = False,
        messages: list | None = None,
        on_change=None,
        max_tool_result: int = DEFAULT_MAX_TOOL_RESULT,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        confirm=None,
        ui=None,
    ):
        self.manager = manager
        self.model = model
        self.think = think
        self.ui = ui or PlainUI()
        # confirm(name, args) -> awaitable[bool]; return False to skip the call.
        self._confirm = confirm
        self.max_tool_result = max_tool_result
        self.messages: list = list(messages or [])
        if system_prompt and not any(
            isinstance(m, dict) and m.get("role") == "system" for m in self.messages
        ):
            self.messages.insert(0, {"role": "system", "content": system_prompt})
        self.client = ollama.Client()
        self._on_change = on_change
        # Token counts from the most recent Ollama call. prompt_tokens is the
        # whole conversation + system prompt + tool schemas that was fed in, i.e.
        # how much of the context window is currently in use.
        self.prompt_tokens = 0
        self.eval_tokens = 0
        _check_ollama(self.client, model)

    def _changed(self) -> None:
        if self._on_change:
            self._on_change(self.messages)

    async def _chat_once(self) -> dict:
        debug("ollama request", {
            "model": self.model,
            "think": self.think,
            "messages": self.messages,
            "tools": f"<{len(self.manager.ollama_tools)} tools, listed at startup>",
        })
        call = functools.partial(
            self.client.chat,
            model=self.model,
            messages=self.messages,
            tools=self.manager.ollama_tools,
            think=self.think,
        )
        try:
            # ollama's client is blocking; keep it off the event loop so a TUI
            # (or anything else awaiting) stays responsive while the model runs.
            resp = await asyncio.get_running_loop().run_in_executor(None, call)
        except Exception as e:  # noqa: BLE001
            raise ChatError(f"Ollama chat failed: {e}") from e
        data = resp.model_dump(exclude_none=True)
        debug("ollama response", data)
        if data.get("prompt_eval_count"):
            self.prompt_tokens = data["prompt_eval_count"]
        self.eval_tokens = data.get("eval_count", 0)
        msg = resp["message"]
        return msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else dict(msg)

    async def send(self, user_input: str) -> str:
        """Run one user turn to completion; return the model's final text."""
        self.messages.append({"role": "user", "content": user_input})
        self._changed()

        for _ in range(MAX_TOOL_ROUNDS):
            with self.ui.thinking():
                msg = await self._chat_once()
            self.messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                self._changed()
                return msg.get("content", "") or "(no response)"

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"] or {}
                if not is_debug():
                    self.ui.tool_call(name, _short(args))

                if self._confirm is not None:
                    try:
                        approved = await self._confirm(name, args)
                    except (EOFError, KeyboardInterrupt):
                        approved = False
                    if not approved:
                        self.ui.tool_note(f"{name} skipped (declined)")
                        self.messages.append({
                            "role": "tool",
                            "name": name,
                            "content": (
                                "The user declined this tool call. Do not retry it. "
                                "Ask the user how to proceed, or continue without it."
                            ),
                        })
                        continue

                try:
                    result_text = await self.manager.call_tool(name, args)
                    status = "ok"
                except ToolCallError as e:
                    result_text = f"ERROR: {e}"
                    status = "error"
                    self.ui.tool_error(str(e))

                capped, dropped = _cap_tool_result(result_text, self.max_tool_result)
                if dropped:
                    self.ui.tool_note(
                        f"{name} result trimmed "
                        f"{len(result_text)} -> {self.max_tool_result} chars "
                        f"(raise with --max-tool-result, 0 = off)"
                    )
                self.messages.append(
                    {"role": "tool", "name": name, "content": capped}
                )
                if not is_debug() and status == "ok":
                    self.ui.tool_result(name, _short(result_text))
            self._changed()

        note = "(stopped: too many tool rounds without a final answer)"
        self.messages.append({"role": "assistant", "content": note})
        self._changed()
        return note


def _short(value, limit: int = 120) -> str:
    s = value if isinstance(value, str) else str(value)
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "..."
