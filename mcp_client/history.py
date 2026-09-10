"""Save / load conversation history to a local JSON file."""

from __future__ import annotations

import json
from pathlib import Path


def _normalize(messages: list) -> list[dict]:
    """Ollama returns Message objects (pydantic-ish); make them plain dicts."""
    out = []
    for m in messages:
        if isinstance(m, dict):
            out.append(m)
        elif hasattr(m, "model_dump"):
            out.append(m.model_dump(exclude_none=True))
        else:
            out.append(dict(m))
    return out


def save_history(path: Path, messages: list, model: str) -> None:
    path = Path(path).expanduser()
    payload = {"model": model, "messages": _normalize(messages)}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_history(path: Path) -> list[dict]:
    path = Path(path).expanduser()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"corrupt history file {path}: {e}") from e
    messages = payload.get("messages", payload if isinstance(payload, list) else [])
    if not isinstance(messages, list):
        raise ValueError(f"history file {path} has no message list")
    return messages
