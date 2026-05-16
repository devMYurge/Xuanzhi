"""Device auto-detection shared across NLP modules.

Order of preference: CUDA → Apple MPS → CPU. Callers can override by
passing ``device="cuda" | "mps" | "cpu"`` directly.
"""

from __future__ import annotations


def resolve_device(device: str | None = None) -> str:
    """Return a torch device string. Imports torch lazily so non-NLP
    callers don't pay the cost.
    """
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
