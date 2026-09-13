"""Device selection: MPS (Apple Silicon) if available, else CPU. Never hard-codes CUDA."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine.")
        return torch.device("mps")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
