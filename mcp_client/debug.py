"""Tiny debug logger shared across modules.

Not using the stdlib ``logging`` module here because the output is meant to be
read inline in the chat session, interleaved with normal prompts, and we want
pretty-printed JSON rather than single-line records.
"""

import json
import sys

_ENABLED = False


def set_debug(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = enabled


def is_debug() -> bool:
    return _ENABLED


def debug(label: str, payload=None) -> None:
    """Print a labelled debug block to stderr when --debug is on."""
    if not _ENABLED:
        return
    print(f"\n\033[90m--- {label} ---", file=sys.stderr)
    if payload is not None:
        if isinstance(payload, str):
            print(payload, file=sys.stderr)
        else:
            try:
                print(json.dumps(payload, indent=2, default=str), file=sys.stderr)
            except (TypeError, ValueError):
                print(repr(payload), file=sys.stderr)
    print("\033[0m", file=sys.stderr, end="")
    sys.stderr.flush()
