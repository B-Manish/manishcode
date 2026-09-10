"""Load MCP server definitions from a JSON config file.

Config shape (config.json)::

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\me"],
          "env": { "SOME_TOKEN": "..." }
        }
      }
    }

The ``mcpServers`` key name matches the convention used by Claude Desktop and
other MCP hosts, so existing configs can often be reused as-is.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_NAMES = ("config.json", "mcp_config.json")

# ``${VAR}`` references inside an ``env`` value are filled in from the parent
# process environment at launch time. This lets a config ship a server that
# needs a secret (an API key) without hard-coding it: set the variable and the
# server is used, leave it unset and the server is skipped with a warning.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(Exception):
    pass


@dataclass
class ServerSpec:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    @property
    def display_cmd(self) -> str:
        return " ".join([self.command, *self.args])

    @property
    def missing_env(self) -> list[str]:
        """Names of ``${VAR}`` references in ``env`` that are not set in the
        parent environment. A non-empty list means this server can't run."""
        missing: list[str] = []
        for value in self.env.values():
            for m in _ENV_REF.finditer(str(value)):
                var = m.group(1)
                if var not in os.environ and var not in missing:
                    missing.append(var)
        return missing

    def full_env(self) -> dict[str, str]:
        """MCP servers are launched with a fresh env by default; merge in the
        parent environment so PATH etc. still work, then layer config env on top.
        ``${VAR}`` references in config values are expanded from the parent
        environment (unset -> empty string)."""
        merged = dict(os.environ)
        for k, v in self.env.items():
            merged[str(k)] = _ENV_REF.sub(
                lambda m: os.environ.get(m.group(1), ""), str(v)
            )
        return merged


def _spec_from_dict(name: str, raw: dict) -> ServerSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"server {name!r}: entry must be an object")
    command = raw.get("command")
    if not command:
        raise ConfigError(f"server {name!r}: missing 'command'")
    args = raw.get("args", [])
    if not isinstance(args, list):
        raise ConfigError(f"server {name!r}: 'args' must be a list")
    env = raw.get("env", {})
    if not isinstance(env, dict):
        raise ConfigError(f"server {name!r}: 'env' must be an object")
    return ServerSpec(
        name=name,
        command=str(command),
        args=[str(a) for a in args],
        env={str(k): str(v) for k, v in env.items()},
    )


def find_config(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        return p
    for name in DEFAULT_CONFIG_NAMES:
        p = Path.cwd() / name
        if p.is_file():
            return p
    return None


def load_config(path: Path) -> dict[str, ServerSpec]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(f"invalid JSON in {path}: {e}") from e
    servers_raw = raw.get("mcpServers") or raw.get("servers")
    if not isinstance(servers_raw, dict) or not servers_raw:
        raise ConfigError(
            f"{path}: expected a non-empty 'mcpServers' object"
        )
    return {name: _spec_from_dict(name, entry) for name, entry in servers_raw.items()}


def spec_from_cmdline(server_cmd: str, name: str | None = None) -> ServerSpec:
    """Build a ServerSpec from a raw shell command string (--server-cmd)."""
    parts = shlex.split(server_cmd, posix=False)
    if not parts:
        raise ConfigError("--server-cmd is empty")
    # shlex posix=False keeps quotes on Windows paths; strip them.
    parts = [p.strip('"') for p in parts]
    return ServerSpec(
        name=name or parts[0],
        command=parts[0],
        args=parts[1:],
    )


def select_servers(
    all_specs: dict[str, ServerSpec], names: list[str] | None
) -> list[ServerSpec]:
    if not names:
        return list(all_specs.values())
    chosen = []
    for n in names:
        if n not in all_specs:
            raise ConfigError(
                f"server {n!r} not in config (have: {', '.join(all_specs) or 'none'})"
            )
        chosen.append(all_specs[n])
    return chosen


def drop_unavailable(
    specs: list[ServerSpec], warn=print
) -> list[ServerSpec]:
    """Filter out servers whose required ``${VAR}`` env references are unset,
    warning about each one instead of letting the launch crash."""
    available = []
    for spec in specs:
        missing = spec.missing_env
        if missing:
            warn(
                f"[warn] skipping server {spec.name!r}: "
                f"environment variable(s) not set: {', '.join(missing)}"
            )
            continue
        available.append(spec)
    return available
