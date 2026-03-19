import glob
import os
import shutil
import traceback

import torch
from accelerate import Accelerator

from scheduler import create_lr_scheduler, create_optimizer, update_optimizer_weight_decay


CHECKPOINTS_TO_KEEP = 2


def paired_model_checkpoint_path(checkpoint_path):
    if checkpoint_path.endswith("_accelerator"):
        return checkpoint_path[: -len("_accelerator")]
    return checkpoint_path


def checkpoint_step_from_path(checkpoint_path):
    checkpoint_name = os.path.basename(paired_model_checkpoint_path(checkpoint_path))
    return int(checkpoint_name.replace("step_", ""))


def list_checkpoint_accelerators(output_dir):
    """Return accelerator checkpoint dirs sorted from newest to oldest."""
    checkpoints = []
    if output_dir is None or not os.path.exists(output_dir):
        return checkpoints

    for check_path in glob.glob(os.path.join(output_dir, "step_*_accelerator")):
        if not os.path.isdir(check_path):
            continue
        try:
            checkpoints.append((checkpoint_step_from_path(check_path), check_path))
        except ValueError:
            continue
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in checkpoints]


def list_checkpoint_dirs(output_dir):
    checkpoint_dirs = []
    if output_dir is None or not os.path.exists(output_dir):
        return checkpoint_dirs

    for check_path in glob.glob(os.path.join(output_dir, "step_*")):
        if os.path.isdir(check_path):
            checkpoint_dirs.append(check_path)
    return checkpoint_dirs


def checkpoint_recovery_hint(checkpoint_path):
    model_dir = paired_model_checkpoint_path(checkpoint_path)
    model_weights = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(model_weights):
        return (
            f" Paired model checkpoint exists at {model_dir}; "
            f"if {model_weights} is still readable, you can restart from weights only "
            f"with --model_name_or_path {model_weights}."
        )
    return ""


def get_last_checkpoint(output_dir):
    """Return the most recent accelerator checkpoint directory in `output_dir`."""
    checkpoints = list_checkpoint_accelerators(output_dir)
    if len(checkpoints) == 0:
        return None
    return checkpoints[0]


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    tokenizer,
    args,
    completed_steps: int,
) -> None:
    """Save a synchronized checkpoint for the accelerator, model, and tokenizer.

    This function contains `accelerator.wait_for_everyone()` barriers, so every
    rank must call it consistently. Save failures are treated as non-fatal and
    partial checkpoint directories are cleaned up on a best-effort basis.
    """
    if args.output_dir is None:
        return

    checkpoint_name = f"step_{completed_steps}"
    output_dir = os.path.join(args.output_dir, checkpoint_name)
    accelerator_output_dir = output_dir + "_accelerator"

    accelerator.wait_for_everyone()

    local_ok = True
    try:
        accelerator.save_state(accelerator_output_dir)
    except Exception as e:
        local_ok = False
        if accelerator.is_main_process:
            print(f"Warning: failed to save accelerator state at {checkpoint_name}: {e}")
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
            print(f"Warning: failed to save model/tokenizer at {checkpoint_name}: {e}")
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

    # If save failed anywhere, keep older checkpoints and cleanup partial dirs on main.
    if not global_ok:
        if accelerator.is_main_process:
            print("Warning: checkpoint save failed on at least one rank; continuing training.")
            try:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                if os.path.exists(accelerator_output_dir):
                    shutil.rmtree(accelerator_output_dir)
            except Exception as e:
                print(f"Warning: failed to cleanup partial checkpoint at {checkpoint_name}: {e}")
        accelerator.wait_for_everyone()
        return

    # New checkpoint is good; now cleanup old checkpoint dirs only on main (best-effort).
    if accelerator.is_main_process:
        keep_accelerator_dirs = set(list_checkpoint_accelerators(args.output_dir)[:CHECKPOINTS_TO_KEEP])
        keep_dirs = keep_accelerator_dirs | {
            paired_model_checkpoint_path(checkpoint_path) for checkpoint_path in keep_accelerator_dirs
        }
        for checkpoint_dir in list_checkpoint_dirs(args.output_dir):
            try:
                if os.path.exists(checkpoint_dir) and checkpoint_dir not in keep_dirs:
                    shutil.rmtree(checkpoint_dir)
                    print("Removed old checkpoint:", checkpoint_dir)
            except Exception as e:
                print(f"Warning: failed to remove old checkpoint {checkpoint_dir}: {e}")

    accelerator.wait_for_everyone()


def resume_candidates(*, resume, output_dir):
    if resume == "":
        # return all checkpoints in output_dir to resume
        return list_checkpoint_accelerators(output_dir)

    checkpoint_path = resume.rstrip("/")
    candidates = [checkpoint_path]
    if output_dir is None:
        return candidates

    checkpoint_path_abs = os.path.abspath(checkpoint_path)
    for candidate in list_checkpoint_accelerators(output_dir):
        if os.path.abspath(candidate) != checkpoint_path_abs:
            candidates.append(candidate)
    return candidates


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
    starting_epoch = 1
    completed_steps = 0
    resume_step = None

    if args.resume is not None:
        candidate_checkpoints = resume_candidates(resume=args.resume, output_dir=args.output_dir)
        if len(candidate_checkpoints) == 0:
            print(f"No checkpoint found in output directory {args.output_dir}")
        else:
            load_errors = []
            for index, candidate_path in enumerate(candidate_checkpoints):
                accelerator.print(f"Resume from checkpoint: {candidate_path}")
                try:
                    accelerator.load_state(candidate_path)
                    checkpoint_path = candidate_path
                    if index > 0:
                        accelerator.print(f"Loaded fallback checkpoint {candidate_path}.")
                    break
                except Exception as e:
                    load_errors.append((candidate_path, str(e)))
                    if index + 1 < len(candidate_checkpoints):
                        accelerator.print(f"Failed to load checkpoint {candidate_path}: {e}")

            if checkpoint_path is None and len(load_errors) > 0:
                if len(load_errors) == 1:
                    failed_path, error = load_errors[0]
                    raise RuntimeError(
                        f"Failed to load checkpoint {failed_path}: {error}.{checkpoint_recovery_hint(failed_path)}"
                    )
                error_summary = "\n".join(
                    f"  - {failed_path}: {error}{checkpoint_recovery_hint(failed_path)}"
                    for failed_path, error in load_errors
                )
                raise RuntimeError("Failed to load all available checkpoints:\n" + error_summary)

    if checkpoint_path is None:
        return checkpoint_path, optimizer, lr_scheduler, starting_epoch, completed_steps, resume_step

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

    # Need to multiply `gradient_accumulation_steps` to reflect real steps.
    completed_steps = checkpoint_step_from_path(checkpoint_path)
    resume_step = completed_steps * args.gradient_accumulation_steps
    starting_epoch = resume_step // len(train_dataloader) + 1
    resume_step -= resume_step // len(train_dataloader) * len(train_dataloader)

    return checkpoint_path, optimizer, lr_scheduler, starting_epoch, completed_steps, resume_step
