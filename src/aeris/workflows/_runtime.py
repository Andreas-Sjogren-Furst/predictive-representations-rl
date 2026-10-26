"""Process-wide settings for workflows that run torch."""

from __future__ import annotations


def use_single_cpu_thread() -> None:
    """Run torch on one CPU thread.

    AERIS evaluates tiny networks one observation at a time. Extra threads only
    add synchronization: on a shared 16-core node under load, one action took
    0.4 ms with one thread and 143 ms with torch's default 16.
    """
    import torch

    torch.set_num_threads(1)
