"""Connect to one or more MCP servers and route tool calls to the right one.

Each server runs inside its own asyncio task that owns the full lifetime of its
``stdio_client`` / ``ClientSession`` scope. This is deliberate: the MCP SDK's
context managers are built on anyio task groups and raise
"Attempted to exit cancel scope in a different task than it was entered in"
if several of them are opened and closed from a single ``AsyncExitStack``.
Keeping one scope per task sidesteps that entirely.

The main task talks to a server by submitting a coroutine factory onto that
server's job queue and awaiting the result via a future.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import ServerSpec
from .debug import debug


class ToolCallError(Exception):
    """Raised when a tool call cannot be completed (unknown tool, server down,
    server returned an error). Caught by the chat loop and fed back to the model."""


def _tool_schema(tool) -> dict:
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
    return schema or {"type": "object", "properties": {}}


def _result_text(result) -> str:
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        elif getattr(block, "type", None) == "resource":
            parts.append(str(getattr(block, "resource", block)))
    return "\n".join(parts)


def _is_error(result) -> bool:
    return bool(getattr(result, "is_error", None) or getattr(result, "isError", None))


def _stderr_tail(errlog, limit: int = 800) -> str:
    """Read back what the server printed to stderr. Textual's alt-screen hides
    real stderr, so a startup failure (missing creds, bad args, ...) otherwise
    only surfaces as anyio's generic 'unhandled errors in a TaskGroup'."""
    try:
        errlog.seek(0)
        text = errlog.read().strip()
    except Exception:  # noqa: BLE001
        return ""
    return text[-limit:] if text else ""


@dataclass
class Connection:
    spec: ServerSpec
    tool_names: list[str] = field(default_factory=list)
    alive: bool = True
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _task: asyncio.Task | None = None

    async def submit(self, make_coro):
        """Run ``make_coro(session)`` on this connection's worker task."""
        if not self.alive:
            raise ToolCallError(f"server {self.spec.name!r} is not running")
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((make_coro, fut))
        return await fut


class ServerManager:
    def __init__(self, specs: list[ServerSpec]):
        self._specs = specs
        self.connections: list[Connection] = []
        self._route: dict[str, tuple[Connection, str]] = {}
        self._ollama_tools: list[dict] = []
        self._tasks: list[asyncio.Task] = []
        self._shutdown = asyncio.Event()

    async def __aenter__(self) -> "ServerManager":
        ready = []
        for spec in self._specs:
            conn, started = self._spawn(spec)
            ready.append((conn, started))

        for conn, started in ready:
            try:
                tools = await started
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] could not start server {conn.spec.name!r}: {e}")
                continue
            self._register(conn, tools)

        # An empty spec list is an intentional "start bare, /mcp connect later"
        # session. Only treat it as an error when servers were asked for and
        # none came up.
        if self._specs and not self.connections:
            await self._stop()
            raise RuntimeError("no MCP servers connected")
        return self

    def _spawn(self, spec: ServerSpec) -> tuple[Connection, asyncio.Future]:
        conn = Connection(spec=spec)
        started: asyncio.Future = asyncio.get_running_loop().create_future()
        conn._task = asyncio.create_task(
            self._serve(conn, started), name=f"mcp:{spec.name}"
        )
        self._tasks.append(conn._task)
        return conn, started

    async def connect(self, spec: ServerSpec) -> int:
        """Start one more server at runtime and merge in its tools. Returns the
        tool count. Raises ValueError if a server of that name is already up."""
        if any(c.spec.name == spec.name for c in self.connections):
            raise ValueError(f"server {spec.name!r} is already connected")
        conn, started = self._spawn(spec)
        tools = await started  # propagates a startup failure to the caller
        self._register(conn, tools)
        return len(conn.tool_names)

    async def disconnect(self, name: str) -> None:
        """Stop one server and drop its tools. Raises ValueError if not found."""
        conn = next((c for c in self.connections if c.spec.name == name), None)
        if conn is None:
            raise ValueError(f"server {name!r} is not connected")
        conn._stop.set()
        conn.alive = False
        if conn._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(conn._task), timeout=5)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                conn._task.cancel()
        for exposed in conn.tool_names:
            self._route.pop(exposed, None)
        dropped = set(conn.tool_names)
        self._ollama_tools = [
            t for t in self._ollama_tools if t["function"]["name"] not in dropped
        ]
        self.connections.remove(conn)

    async def __aexit__(self, *exc):
        await self._stop()
        return False

    async def _stop(self) -> None:
        self._shutdown.set()
        for t in self._tasks:
            try:
                await asyncio.wait_for(asyncio.shield(t), timeout=10)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                t.cancel()

    async def _serve(self, conn: Connection, started: asyncio.Future) -> None:
        params = StdioServerParameters(
            command=conn.spec.command, args=conn.spec.args, env=conn.spec.full_env()
        )
        errlog = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        try:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    started.set_result(tools)
                    await self._pump(conn, session)
        except Exception as e:  # noqa: BLE001
            conn.alive = False
            if not started.done():
                tail = _stderr_tail(errlog)
                started.set_exception(RuntimeError(f"{e}\n{tail}") if tail else e)
            else:
                debug(f"server {conn.spec.name} exited", str(e))
        finally:
            conn.alive = False
            errlog.close()

    async def _pump(self, conn: Connection, session: ClientSession) -> None:
        stop_wait = asyncio.create_task(self._await_stop(conn))
        try:
            while True:
                get_job = asyncio.create_task(conn._queue.get())
                done, _ = await asyncio.wait(
                    {get_job, stop_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                if get_job not in done:
                    get_job.cancel()
                    return  # global shutdown or this server was disconnected
                make_coro, fut = get_job.result()
                try:
                    fut.set_result(await make_coro(session))
                except Exception as e:  # noqa: BLE001
                    if not fut.done():
                        fut.set_exception(e)
        finally:
            stop_wait.cancel()

    async def _await_stop(self, conn: Connection) -> None:
        waiters = [
            asyncio.create_task(self._shutdown.wait()),
            asyncio.create_task(conn._stop.wait()),
        ]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waiters:
                w.cancel()

    def _register(self, conn: Connection, tools) -> None:
        for tool in tools:
            exposed = tool.name
            if exposed in self._route:
                exposed = f"{conn.spec.name}__{tool.name}"
            self._route[exposed] = (conn, tool.name)
            conn.tool_names.append(exposed)
            self._ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": exposed,
                        "description": tool.description or "",
                        "parameters": _tool_schema(tool),
                    },
                }
            )
        self.connections.append(conn)
        print(f"  [ok] {conn.spec.name}: {len(conn.tool_names)} tool(s)")

    @property
    def ollama_tools(self) -> list[dict]:
        return self._ollama_tools

    def describe_tools(self) -> str:
        lines = []
        for conn in self.connections:
            status = "" if conn.alive else "  [DOWN]"
            for name in conn.tool_names:
                lines.append(f"  - {name}  ({conn.spec.name}){status}")
        return "\n".join(lines) or "  (none)"

    async def call_tool(self, name: str, arguments: dict) -> str:
        route = self._route.get(name)
        if route is None:
            raise ToolCallError(
                f"unknown tool {name!r}. Available: {', '.join(self._route) or 'none'}"
            )
        conn, real_name = route
        debug(f"tool call -> {conn.spec.name}", {"tool": real_name, "arguments": arguments})

        async def job(session):
            return await session.call_tool(real_name, arguments=arguments or {})

        try:
            result = await conn.submit(job)
        except ToolCallError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ToolCallError(
                f"server {conn.spec.name!r} failed to run {real_name!r}: {e}"
            ) from e

        text = _result_text(result) or "(tool returned no text content)"
        debug(f"tool result <- {conn.spec.name}", {"is_error": _is_error(result), "content": text})
        if _is_error(result):
            raise ToolCallError(f"{real_name!r} returned an error: {text}")
        return text

    async def health_check(self) -> list[str]:
        dead = []
        for conn in self.connections:
            if not conn.alive:
                dead.append(conn.spec.name)
                continue
            try:
                await asyncio.wait_for(conn.submit(lambda s: s.list_tools()), timeout=5)
            except Exception:  # noqa: BLE001
                dead.append(conn.spec.name)
        return dead
