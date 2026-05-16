"""Torch device auto-detection, shared by the nlp and cv layers.

Order of preference: CUDA -> Apple MPS -> CPU. Callers can override by
passing an explicit ``device`` string.
"""

from __future__ import annotations


def resolve_device(device: str | None = None) -> str:
    """Return a torch device string. ``torch`` is imported lazily so
    non-ML callers don't pay the import cost.
    """
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
