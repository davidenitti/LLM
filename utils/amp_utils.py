import torch
from contextlib import nullcontext


def get_model_device(model: torch.nn.Module) -> torch.device:
    param = next(model.parameters(), None)
    return param.device if param is not None else torch.device("cpu")


def get_amp_context(precision: str):
    """
    Return a context manager for AMP based on a precision string.

    precision: one of {"bf16", "fp16", "float32", "32", "16"}

    - For bf16/fp16: returns torch.autocast(device_type="cuda", dtype=...)
    - For fp32/float32/32: returns nullcontext (no AMP)
    """

    p = precision.lower()
    if p in {"bf16", "bfloat16"}:
        amp_dtype = torch.bfloat16
    elif p in {"fp16", "float16", "16"}:
        amp_dtype = torch.float16
    elif p in {"fp32", "float32", "32"}:
        amp_dtype = None
    else:
        raise ValueError(f"Unknown precision string: {precision}")

    if torch.cuda.is_available() and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()
