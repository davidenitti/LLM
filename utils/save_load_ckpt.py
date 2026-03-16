import glob
import os

import torch
from accelerate import Accelerator
import shutil
import traceback

from scheduler import create_lr_scheduler, create_optimizer, update_optimizer_weight_decay


def get_last_checkpoint(output_dir):
    """Return the most recent accelerator checkpoint directory in `output_dir`."""
    checkpoints = []
    if os.path.exists(output_dir):
        for check_path in glob.glob(os.path.join(output_dir, "epoch_*_accelerator")):
            if os.path.isdir(check_path):
                try:
                    epoch_num = int(os.path.basename(check_path).split("_")[1])
                    checkpoints.append((epoch_num, check_path))
                except:
                    pass
    if len(checkpoints) == 0:
        return None
    checkpoints = sorted(checkpoints, key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    tokenizer,
    args,
    epoch: int,
) -> None:
    """Save a synchronized epoch checkpoint for the accelerator, model, and tokenizer.

    This function contains `accelerator.wait_for_everyone()` barriers, so every
    rank must call it consistently. Save failures are treated as non-fatal and
    partial checkpoint directories are cleaned up on a best-effort basis.
    """
    # Important: to be multi-GPU safe, every rank must hit the same barriers even if saving fails.
    # We treat save failures as non-fatal (do not stop training), but we also avoid deleting the
    # previous checkpoint unless the new one is confirmed saved successfully across all ranks.

    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")

    # Align ranks before saving.
    accelerator.wait_for_everyone()

    local_ok = True
    try:
        accelerator.save_state(output_dir + "_accelerator")
    except Exception as e:
        local_ok = False
        if accelerator.is_main_process:
            print(f"Warning: failed to save accelerator state at epoch {epoch}: {e}")

            traceback.print_exc()

    accelerator.wait_for_everyone()

    try:
        unwrapped_model = accelerator.unwrap_model(model)
        # If torch.compile wrapped the model, unwrap to the original module.
        if hasattr(unwrapped_model, "_orig_mod") and hasattr(unwrapped_model._orig_mod, "save_pretrained"):
            unwrapped_model = unwrapped_model._orig_mod
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
    except Exception as e:
        local_ok = False
        if accelerator.is_main_process:
            print(f"Warning: failed to save model/tokenizer at epoch {epoch}: {e}")

            traceback.print_exc()

    accelerator.wait_for_everyone()

    # Make sure *all* ranks agree whether saving succeeded.
    global_ok = local_ok
    if (
        accelerator.num_processes > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        t = torch.tensor([1 if local_ok else 0], device=accelerator.device, dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
        global_ok = bool(int(t.item()))

    # If save failed anywhere, keep previous checkpoint pointer and optionally cleanup partial dirs on main.
    if not global_ok:
        if accelerator.is_main_process:
            print("Warning: checkpoint save failed on at least one rank; continuing training.")
            try:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                if os.path.exists(output_dir + "_accelerator"):
                    shutil.rmtree(output_dir + "_accelerator")
            except Exception as e:
                print(f"Warning: failed to cleanup partial checkpoint at epoch {epoch}: {e}")
        accelerator.wait_for_everyone()

    # New checkpoint is good; now cleanup old checkpoint dirs only on main (best-effort).
    if accelerator.is_main_process:
        dir2delete = glob.glob(os.path.join(args.output_dir, "epoch_*"))
        for d in dir2delete:
            try:
                if os.path.exists(d) and d != output_dir and d != output_dir + "_accelerator":
                    shutil.rmtree(d)
                    print("Removed old checkpoint:", d)
            except Exception as e:
                print(f"Warning: failed to remove old checkpoint {d}: {e}")

    accelerator.wait_for_everyone()


def resume_training(
    *,
    args,
    accelerator: Accelerator,
    model,
    optimizer,
    lr_scheduler,
    train_dataloader,
    num_update_steps_per_epoch,
    num_training_steps,
    warmup_steps,
    scheduler_specific_kwargs,
):
    """Resume from a checkpoint and return the restored training state."""
    checkpoint_path = None
    dir_checkpoint = None
    starting_epoch = 1
    completed_steps = 0
    resume_step = None

    if args.resume is not None:
        if args.resume != "":
            checkpoint_path = args.resume.rstrip("/")
            dir_checkpoint = os.path.basename(checkpoint_path)
        else:
            checkpoint_path = get_last_checkpoint(args.output_dir)
            if checkpoint_path is None:
                print(f"No checkpoint found in output directory {args.output_dir}")
            else:
                dir_checkpoint = os.path.basename(checkpoint_path)

    if checkpoint_path is None:
        return checkpoint_path, optimizer, lr_scheduler, starting_epoch, completed_steps, resume_step

    accelerator.print(f"Resume from checkpoint: {checkpoint_path}")
    accelerator.load_state(checkpoint_path)

    if args.reset_resume:
        accelerator.print("Resetting optimizer and scheduler states after resume.")
        base_model = accelerator.unwrap_model(model)
        optimizer = create_optimizer(
            base_model,
            args.weight_decay,
            args.learning_rate,
            args.beta1,
            scale_lowrank=args.scale_lowrank,
        )
        lr_scheduler = create_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler_type=args.lr_scheduler_type,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=args.min_learning_rate,
            scheduler_specific_kwargs=scheduler_specific_kwargs,
        )
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
        return checkpoint_path, optimizer, lr_scheduler, starting_epoch, completed_steps, resume_step

    # Keep optimizer state but apply the newly requested weight decay to decay groups.
    update_optimizer_weight_decay(
        optimizer,
        args.weight_decay,
        scale_lowrank=args.scale_lowrank,
    )

    training_difference = os.path.splitext(dir_checkpoint)[0]
    if "epoch" in training_difference:
        starting_epoch = int(training_difference.replace("epoch_", "").split("_")[0]) + 1
        completed_steps = (starting_epoch - 1) * num_update_steps_per_epoch
    else:
        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_dataloader) + 1
        completed_steps = resume_step // args.gradient_accumulation_steps
        resume_step -= resume_step // len(train_dataloader) * len(train_dataloader)

    return checkpoint_path, optimizer, lr_scheduler, starting_epoch, completed_steps, resume_step
