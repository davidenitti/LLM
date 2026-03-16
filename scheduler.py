import math
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler
import torch
from blocks.basic_transformer import LowRankLinear


def get_fast_start_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float | None = None,
    min_lr_rate: float | None = None,
    speed: int = 8,
    change_speed: float = 0.5,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")

    lr_lambda = partial(
        _fast_start_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
        speed=speed,
        change_speed=change_speed,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_lr_scheduler(
    optimizer: Optimizer,
    lr_scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float | None = None,
    scheduler_specific_kwargs: dict | None = None,
):
    scheduler_specific_kwargs = scheduler_specific_kwargs or {}

    if lr_scheduler_type == "fast_start_cosine_with_min_lr":
        return get_fast_start_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr,
        )

    return get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )


def _fast_start_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_rate: float,
    speed: int,
    change_speed: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress < change_speed / speed:
        progress = progress * speed
    else:
        progress = change_speed + (progress - change_speed / speed) * (1 - change_speed) / (
            1 - change_speed / speed
        )
    cosine_decay = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return min_lr_rate + (1.0 - min_lr_rate) * cosine_decay


def create_optimizer(
    model,
    weight_decay,
    learning_rate,
    beta1,
    beta2=0.95,
    optimizer_type="adamw",
    scale_lowrank=1.0,
    scale_lowrank_lr=1.0,
):
    """Build an optimizer with separate groups for decay, no-decay, and LowRankLinear params."""

    lowrank_param_ids = set()
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            for param in module.parameters():
                lowrank_param_ids.add(id(param))

    decay_params = []
    lowrank_decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Skipping {name} from optimizer because requires_grad=False")
            continue
        if param.dim() <= 1:
            no_decay_params.append(param)
        elif id(param) in lowrank_param_ids:
            lowrank_decay_params.append(param)
            print(f"LowRankLinear weight decay applied to {name}")
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append(
            {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate, "group_name": "decay"}
        )
    if lowrank_decay_params:
        param_groups.append(
            {
                "params": lowrank_decay_params,
                "weight_decay": weight_decay * scale_lowrank,
                "lr": learning_rate * scale_lowrank_lr,
                "group_name": "lowrank_decay",
            }
        )
    if no_decay_params:
        param_groups.append(
            {"params": no_decay_params, "weight_decay": 0.0, "lr": learning_rate, "group_name": "no_decay"}
        )
    if optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups, lr=learning_rate, betas=(beta1, beta2))
    elif optimizer_type == "sgd":
        return torch.optim.SGD(param_groups, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def update_optimizer_weight_decay(optimizer, weight_decay, scale_lowrank):
    """Refresh optimizer weight decay values while preserving custom param-group rules."""
    for group in optimizer.param_groups:
        group_name = group.get("group_name")
        if group_name == "decay":
            group["weight_decay"] = weight_decay
        elif group_name == "lowrank_decay":
            group["weight_decay"] = weight_decay * scale_lowrank
        elif group_name == "no_decay":
            group["weight_decay"] = 0.0
        elif group.get("weight_decay", 0.0) > 0.0:
            group["weight_decay"] = weight_decay
