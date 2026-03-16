from __future__ import annotations

from typing import Dict, Optional

import torch
from accelerate import Accelerator


def _as_scalar_float_tensor(accelerator: Accelerator, value: object) -> torch.Tensor:
    if value is None:
        raise ValueError("Metric value cannot be None")
    if torch.is_tensor(value):
        t = value.detach()
    else:
        t = torch.tensor(value)
    assert t.numel() == 1, "Metric value must be a scalar"
    t = t.float()
    return t.to(accelerator.device)


def accumulate_weighted_metrics(
    accelerator: Accelerator,
    weighted_sums: Dict[str, torch.Tensor],
    weight_sums: Dict[str, torch.Tensor],
    metrics: Dict[str, object] | None,
    *,
    weight_key: str = "num_valid",
    require_weight: bool = True,
) -> None:
    """Accumulate metrics as weighted sums using `weight_key`.

    Assumes each metric value is an average over `weight_key` units (e.g. tokens).
    Stores per-metric:
      - weighted_sums[k] += metric_value * weight
      - weight_sums[k] += weight

    Also stores a running sum for `weight_key` itself in weighted_sums[weight_key].
    """

    if not metrics:
        return

    # Default weight=1 if not provided.
    weight_value = metrics.get(weight_key, None)
    if require_weight and weight_value is None:
        raise KeyError(
            f"Missing required weight_key '{weight_key}' in metrics. "
            "Expected the model to return it in outputs.metrics."
        )
    weight_t = _as_scalar_float_tensor(accelerator, weight_value) if weight_value is not None else None
    if weight_t is None:
        weight_t = torch.tensor(1.0, device=accelerator.device)
    assert weight_t >= 0.0, "Weight tensor must be non-negative."

    # Track the raw sum of weights for debugging/logging.
    if weight_key not in weighted_sums:
        weighted_sums[weight_key] = weight_t
    else:
        weighted_sums[weight_key] = weighted_sums[weight_key] + weight_t

    for k, v in metrics.items():
        if v is None or k == weight_key:
            continue
        metric_t = _as_scalar_float_tensor(accelerator, v)
        wsum_add = metric_t * weight_t
        if k not in weighted_sums:
            weighted_sums[k] = wsum_add
            weight_sums[k] = weight_t
        else:
            weighted_sums[k] = weighted_sums[k] + wsum_add
            weight_sums[k] = weight_sums[k] + weight_t


def reduce_weighted_metrics(
    accelerator: Accelerator,
    weighted_sums: Dict[str, torch.Tensor],
    weight_sums: Dict[str, torch.Tensor],
    *,
    weight_key: str = "num_valid",
) -> Dict[str, float]:
    """Reduce weighted metric accumulators across processes and return weighted means."""

    out: Dict[str, float] = {}

    # Reduce weight_key sum if present.
    if weight_key in weighted_sums:
        w_total = accelerator.reduce(weighted_sums[weight_key], reduction="sum")
        out[weight_key] = float(w_total.item())

    for k, wsum_t in weighted_sums.items():
        if k == weight_key:
            continue
        wsum_total = accelerator.reduce(wsum_t, reduction="sum")
        w_total = accelerator.reduce(weight_sums[k], reduction="sum")
        denom = float(w_total.item())
        if denom > 0:
            out[k] = float((wsum_total / w_total).item())

    return out


class TrainMetricsAccumulator:
    def __init__(self, accelerator: Accelerator, weight_key: str = "num_valid") -> None:
        self.accelerator = accelerator
        self.weight_key = weight_key
        self.metrics_mini_wsum: Dict[str, torch.Tensor] = {}
        self.metrics_mini_w: Dict[str, torch.Tensor] = {}
        self.reset_all()

    def reset_all(self) -> None:
        self.total_loss = 0.0
        self.num_loss = 0
        self.total_loss_mini = 0.0
        self.num_loss_mini = 0
        self.total_loss_mini_acc = 0.0
        self.num_loss_mini_acc = 0
        self.grad_norm_total = 0.0
        self.grad_norm_count = 0
        self.grad_norm_max = 0.0
        self.metrics_mini_wsum.clear()
        self.metrics_mini_w.clear()

    def update(self, loss: torch.Tensor, metrics: Optional[Dict[str, object]] = None) -> float:
        loss_value = self._mean_loss(loss)
        self.total_loss += loss_value
        self.total_loss_mini += loss_value
        self.num_loss += 1
        self.num_loss_mini += 1

        accumulate_weighted_metrics(
            self.accelerator,
            self.metrics_mini_wsum,
            self.metrics_mini_w,
            metrics,
            weight_key=self.weight_key,
        )
        return loss_value

    def update_grad(self, grad_norm: object) -> None:
        if grad_norm is None:
            return
        value = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)
        self.grad_norm_total += value
        self.grad_norm_count += 1
        self.grad_norm_max = max(self.grad_norm_max, value)

    def build_log_payload(self) -> Dict[str, float]:
        self.total_loss_mini_acc += self.total_loss_mini
        self.num_loss_mini_acc += self.num_loss_mini
        payload = {
            "train_loss_logtrain_steps": self.total_loss_mini / self.num_loss_mini,
            "train_loss_logtrain_steps_accum": self.total_loss_mini_acc / self.num_loss_mini_acc,
        }
        if self.grad_norm_count > 0:
            payload["grad_norm_avg"] = self.grad_norm_total / self.grad_norm_count
            payload["grad_norm_max"] = self.grad_norm_max
        if self.metrics_mini_wsum:
            reduced_metrics = reduce_weighted_metrics(
                self.accelerator,
                self.metrics_mini_wsum,
                self.metrics_mini_w,
                weight_key=self.weight_key,
            )
            for k, v in reduced_metrics.items():
                payload[f"train_{k}_logtrain_steps"] = v
        return payload

    def reset_window(self) -> None:
        self.total_loss_mini = 0.0
        self.num_loss_mini = 0
        self.grad_norm_total = 0.0
        self.grad_norm_count = 0
        self.grad_norm_max = 0.0
        self.metrics_mini_wsum.clear()
        self.metrics_mini_w.clear()

    def total_loss_avg(self) -> float:
        return self.total_loss / self.num_loss

    def train_acc_loss_avg(self) -> float:
        return self.total_loss_mini_acc / self.num_loss_mini_acc

    def reset_total_losses(self) -> None:
        self.total_loss = 0.0
        self.num_loss = 0
        self.total_loss_mini_acc = 0.0
        self.num_loss_mini_acc = 0

    def _mean_loss(self, loss: torch.Tensor) -> float:
        loss_mean = self.accelerator.gather_for_metrics(loss.detach()).mean().float()
        return float(loss_mean.item())
