import functools
import gc
import inspect
import torch


def clean_gpu():
    gc.collect()
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats()
    if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
        torch.cuda.reset_accumulated_memory_stats()


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Best-effort check for CUDA OOM across PyTorch error variants."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    # PyTorch sometimes raises RuntimeError with this substring.
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def retry_on_cuda_oom(
    *,
    batch_size_kw: str,
    min_batch_size: int = 1,
    reduce_factor: float = 0.7,
    cleanup_fn=clean_gpu,
    is_oom_fn=_is_cuda_oom_error,
):
    """Decorator: retry a function on CUDA OOM by shrinking a batch-size kwarg.

    - Adjusts argument `batch_size_kw` (supports positional or keyword passing)
      by multiplying by `reduce_factor` until `min_batch_size`.
    - Calls `cleanup_fn()` (e.g. `clean_gpu`) between retries.
    - If still OOM at `min_batch_size`, re-raises the last exception.

    Notes
    -----
    This decorator uses `inspect.signature` to bind args/kwargs so it can
    rewrite the batch-size argument even if it was passed positionally.
    """

    if int(min_batch_size) < 1:
        raise ValueError("min_batch_size must be >= 1")
    assert reduce_factor < 0.9, "reduce_factor must be < 0.9"

    def _decorator(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            last_exc: BaseException | None = None

            bound = sig.bind_partial(*args, **kwargs)
            # Ensure defaults (e.g. batch_size_val=10) are present.
            bound.apply_defaults()

            if batch_size_kw not in bound.arguments:
                raise TypeError(
                    f"retry_on_cuda_oom expected argument '{batch_size_kw}' to be provided when calling {fn.__name__}"
                )

            cur_bsz = int(bound.arguments[batch_size_kw])
            if cur_bsz < 1:
                raise ValueError(f"{batch_size_kw} must be at least 1")

            while cur_bsz >= int(min_batch_size):
                try:
                    bound.arguments[batch_size_kw] = cur_bsz
                    return fn(*bound.args, **bound.kwargs)
                except BaseException as exc:
                    if is_oom_fn is None or not is_oom_fn(exc):
                        raise
                    last_exc = exc

                    if cleanup_fn is not None:
                        cleanup_fn()

                    if cur_bsz <= int(min_batch_size):
                        break

                    next_bsz = int(max(int(min_batch_size), cur_bsz * reduce_factor))
                    if next_bsz == cur_bsz:
                        next_bsz = cur_bsz - 1
                    cur_bsz = next_bsz
                    print(f"Retrying {fn.__name__} with smaller {batch_size_kw}={cur_bsz} due to CUDA OOM.")

            assert last_exc is not None
            raise last_exc

        # Preserve the original callable signature for help()/IDE tooling.
        _wrapped.__signature__ = sig
        return _wrapped

    return _decorator
