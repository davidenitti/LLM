import torch
from torch import nn
from torch.nn import functional as F


def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def StandardLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    aligned: bool = False,
    **kwargs,
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    if aligned:
        shift_labels = labels
    else:
        # the labels are basically the input, so we need to shift them for next token prediction
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.reshape(-1, vocab_size)
    shift_labels = shift_labels.reshape(-1).to(logits.device)

    valid = shift_labels != ignore_index
    num_valid = valid.sum()
    metrics = {
        "num_valid": num_valid,
        "ratio_correct": torch.zeros((), device=logits.device, dtype=torch.float32),
    }

    if not valid.any():
        # Avoid potential NaNs when every label is ignored.
        loss_zero = logits.sum() * 0.0
        return loss_zero, metrics

    preds = logits[valid].argmax(dim=-1)
    metrics["ratio_correct"] = (preds == shift_labels[valid]).float().mean()

    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss, metrics


def ArgmaxFocusLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    aligned: bool = False,
    weight_correct: float = 0.1,
    **kwargs,
):
    """Standard LM cross-entropy, but down-weight positions already predicted correctly.

    Per-token weight is weight_correct when `label == argmax(logits)` and 1.0 otherwise.
    Shifting and `ignore_index` semantics match `StandardLoss`.
    """

    logits = logits.float()
    labels = labels.to(logits.device)

    if aligned:
        shift_labels = labels
    else:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device)

    valid = shift_labels != ignore_index
    num_valid = valid.sum()
    if not valid.any():
        loss_zero = logits.sum() * 0.0
        metrics = {
            "ratio_correct": torch.zeros((), device=logits.device, dtype=torch.float32),
            "num_valid": num_valid,
        }
        return loss_zero, metrics

    logits_valid = logits[valid]
    labels_valid = shift_labels[valid]

    per_token_loss = F.cross_entropy(
        logits_valid,
        labels_valid,
        reduction="none",
        **kwargs,
    )

    preds = logits_valid.argmax(dim=-1)
    metrics = {}
    metrics["ratio_correct"] = (preds == labels_valid).float().mean()
    metrics["num_valid"] = num_valid
    weights = torch.where(
        preds == labels_valid,
        per_token_loss.new_full((), weight_correct),
        per_token_loss.new_full((), 1.0),
    )
    weighted_loss = per_token_loss * weights

    if num_items_in_batch is not None:
        return weighted_loss.sum() / num_items_in_batch, metrics
    return weighted_loss.mean(), metrics


def MultiMarginLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    aligned: bool = False,
    **kwargs,
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)

    if aligned:
        shift_labels = labels
    else:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device)

    valid = shift_labels != ignore_index
    num_valid = valid.sum()
    metrics = {
        "num_valid": num_valid,
        "ratio_correct": torch.zeros((), device=logits.device, dtype=torch.float32),
    }

    if not valid.any():
        return logits.sum() * 0.0, metrics

    logits = logits[valid]
    shift_labels = shift_labels[valid]

    preds = logits.argmax(dim=-1)
    metrics["ratio_correct"] = (preds == shift_labels).float().mean()

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.multi_margin_loss(
        logits,
        shift_labels,
        p=kwargs.get("p", 1),
        margin=kwargs.get("margin", 1.0),
        reduction=reduction,
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss, metrics


def FocalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    aligned: bool = False,
    gamma: float = 1.0,
    detach: bool = True,
    **kwargs,
):
    """Standard LM cross-entropy with focal weighting.

    Computes per-token CE (respecting `ignore_index` and shifting like `StandardLoss`),
    then applies the focal factor: `(1 - p_t) ** gamma`, where `p_t = exp(-CE)`.

    Args:
        gamma: Focusing parameter (>= 0). 0.0 reduces to plain CE.

    Returns:
        (loss, metrics) where metrics contains `num_valid` and `ratio_correct`.
    """

    logits = logits.float()
    labels = labels.to(logits.device)

    if aligned:
        shift_labels = labels
    else:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits = logits.reshape(-1, vocab_size)
    shift_labels = shift_labels.reshape(-1).to(logits.device)

    valid = shift_labels != ignore_index
    num_valid = valid.sum()
    metrics = {
        "num_valid": num_valid,
        "ratio_correct": torch.zeros((), device=logits.device, dtype=torch.float32),
    }

    if not valid.any():
        loss_zero = logits.sum() * 0.0
        return loss_zero, metrics

    logits_valid = logits[valid]
    labels_valid = shift_labels[valid]

    preds = logits_valid.argmax(dim=-1)
    metrics["ratio_correct"] = (preds == labels_valid).float().mean()

    per_token_ce = F.cross_entropy(
        logits_valid,
        labels_valid,
        reduction="none",
        **kwargs,
    )

    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")

    pt = torch.exp(-per_token_ce)
    focal_factor = (1.0 - pt).clamp(min=0.0) ** gamma
    if detach:
        focal_factor = focal_factor.detach()
    per_token_loss = per_token_ce * focal_factor

    if num_items_in_batch is not None:
        return per_token_loss.sum() / num_items_in_batch, metrics
    return per_token_loss.mean(), metrics
