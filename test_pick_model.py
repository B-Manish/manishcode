"""Self-check for chat.pick_model. Run: python test_pick_model.py"""
from types import SimpleNamespace as NS

import mcp_client.chat as chat


class FakeClient:
    def __init__(self, installed=(), loaded=()):
        self._installed = list(installed)
        self._loaded = list(loaded)

    def ps(self):
        return NS(models=[NS(model=m) for m in self._loaded])

    def list(self):
        return NS(models=[NS(model=m, size=s) for m, s in self._installed])


GB = 1024 ** 3


def main():
    chat._total_ram_bytes = lambda: 16 * GB
    chat._gpu_vram_bytes = lambda: None  # default: no GPU

    # already-loaded model wins outright
    c = FakeClient(installed=[("small:1b", GB), ("huge:70b", 40 * GB)],
                   loaded=["small:1b"])
    assert chat.pick_model(c) == "small:1b"

    # nothing loaded -> largest that fits 70% of 16GB (~11.2GB)
    c = FakeClient(installed=[("a", 2 * GB), ("b", 8 * GB), ("c", 40 * GB)])
    assert chat.pick_model(c) == "b"

    # GPU present -> size to VRAM (8GB) not RAM: 80% -> 6.4GB
    chat._gpu_vram_bytes = lambda: 8 * GB
    c = FakeClient(installed=[("a", 2 * GB), ("b", 7 * GB), ("c", 40 * GB)])
    assert chat.pick_model(c) == "a"
    chat._gpu_vram_bytes = lambda: None

    # none fit -> smallest
    c = FakeClient(installed=[("x", 30 * GB), ("y", 50 * GB)])
    assert chat.pick_model(c) == "x"

    # no models -> error
    try:
        chat.pick_model(FakeClient())
    except chat.ChatError:
        pass
    else:
        raise AssertionError("expected ChatError")

    print("ok")


if __name__ == "__main__":
    main()
