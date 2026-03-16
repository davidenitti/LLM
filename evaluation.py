import math
from random import random
from time import time
from inference import generate
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
from utils.amp_utils import get_amp_context, get_model_device
from utils.metrics_utils import accumulate_weighted_metrics, reduce_weighted_metrics
import numpy as np
from collections import Counter
from utils.cuda_utils import clean_gpu, retry_on_cuda_oom
from arc_agi_dataset import decode_predicted_output_grids, inverse_augmentation


def _tta_worker_init_fn(worker_id: int) -> None:
    import random
    import numpy as np
    import torch

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _identity_collate(batch):
    return batch


class _RepeatIndexSampler(Sampler[int]):
    def __init__(self, num_tasks: int, num_augmentations: int) -> None:
        self.num_tasks = int(num_tasks)
        self.num_augmentations = int(num_augmentations)

    def __len__(self) -> int:
        return self.num_tasks * self.num_augmentations

    def __iter__(self):
        for idx in range(self.num_tasks):
            for _ in range(self.num_augmentations):
                yield idx


def build_tta_dataloader(
    dataset,
    num_augmentations: int,
    *,
    tta_num_workers: int,
    tta_loader_batch_size: int,
):
    assert num_augmentations >= 1, "num_augmentations must be at least 1"
    assert tta_loader_batch_size >= 1, "tta_loader_batch_size must be at least 1"
    assert tta_num_workers >= 0, "tta_num_workers must be non-negative"

    sampler = _RepeatIndexSampler(len(dataset.data), num_augmentations)
    loader_kwargs = {}
    if tta_num_workers > 0:
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=tta_loader_batch_size,
        sampler=sampler,
        num_workers=tta_num_workers,
        collate_fn=_identity_collate,
        worker_init_fn=_tta_worker_init_fn,
        **loader_kwargs,
    )

def compute_valv2(
    model,
    eval_dataloader,
    accelerator,
    detach_every,
    generalized,
    precision,
    aligned_labels,
    train_progress,
):
    """
    Evaluate model perplexity with globally correct multi-GPU aggregation.

    - Token-weighted loss across processes and mini-steps
    - Works with uneven last batches
    - Keeps a simple mean-of-means for comparison and prints the delta when generalized=True

    Returns (eval_loss, perplexity, valid_token_count or None, metrics_avg).
    """
    model.eval()

    total_loss_sum = 0.0
    total_valid_tokens = 0

    simple_loss_sum = 0.0
    simple_loss_cnt = 0

    # Token-weighted metric aggregation across the full eval set.
    # Each metric is assumed to be an average over valid (non -100) tokens.
    metric_wsum: dict[str, torch.Tensor] = {}
    metric_w: dict[str, torch.Tensor] = {}

    ctx = get_amp_context(precision)
    iter_dl = tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval", leave=False)
    for step, orig_batch in enumerate(iter_dl):
        # Handle optional detach scheduling
        if detach_every is not None:
            detach_size = detach_every
            assert orig_batch["input_ids"].shape[1] % detach_size == 0
        else:
            detach_size = orig_batch["input_ids"].shape[1]
        num_mini_steps = orig_batch["input_ids"].shape[1] // detach_size

        hidden_state = None
        for mini_step in range(num_mini_steps):
            if num_mini_steps == 1:
                batch = orig_batch
            else:
                batch = {}
                for k in orig_batch:
                    if orig_batch[k] is None:
                        batch[k] = None
                    else:
                        batch[k] = orig_batch[k][:, mini_step * detach_size : (mini_step + 1) * detach_size]
                batch["hidden"] = hidden_state

            with torch.inference_mode(), ctx:
                outputs = model(**batch, eval_mode=True, train_progress=train_progress)
                hidden_state = getattr(outputs, "hidden_states", None)

            # Ensure scalar loss per process; upcast before reduction for stability
            loss = outputs.loss
            if isinstance(loss, torch.Tensor):
                loss = loss.detach()
                # Prefer float32 reduction to avoid fp16/bf16 accumulation error
                loss = loss.to(torch.float32)
                assert loss.dim() == 0, "Loss tensor must be scalar"
            else:
                loss = torch.tensor(
                    float(loss),
                    device=orig_batch["input_ids"].device,
                    dtype=torch.float32,
                )

            labels = batch.get("labels", None)
            assert labels is not None, "Batch must contain 'labels' for validation loss aggregation"
            # Correct valid-token counting under detach: only drop the very first
            # token of the full sequence, not the first token of each chunk.
            # FIXME ForCausalLMLoss would remove the last token for each chunk!
            if aligned_labels:
                labels_for_count = labels
            else:
                labels_for_count = labels[:, 1:]
                assert not detach_every
            local_valid = (labels_for_count != -100).sum()

            metrics = getattr(outputs, "metrics", None) or {}
            if not isinstance(metrics, dict):
                metrics = {}

            # Accumulate using num_valid coming from outputs.metrics.
            accumulate_weighted_metrics(
                accelerator,
                metric_wsum,
                metric_w,
                metrics,
                weight_key="num_valid",
                require_weight=True,
            )

            # Use the same num_valid for loss weighting + global token counts.
            num_valid_t = metrics.get("num_valid", None)
            if num_valid_t is None:
                # Should not happen due to require_weight=True above, but keep a clear error.
                raise KeyError("Missing 'num_valid' in outputs.metrics")
            if not isinstance(num_valid_t, torch.Tensor):
                raise TypeError(f"compute_valv2 expected tensor num_valid, got {type(num_valid_t)}")
            num_valid_t = num_valid_t.detach().to(device=loss.device, dtype=torch.float32)

            # Optional sanity check (doesn't affect results): ensure model-provided count matches label-derived count.
            # local_valid is an integer tensor; num_valid_t may be float.
            if torch.is_tensor(local_valid):
                if float(local_valid.detach().to(torch.float32).item()) != float(num_valid_t.item()):
                    # Keep it silent by default; enable when debugging.
                    pass

            # Gather token-weighted loss and counts across processes
            loss_sum_local = loss * local_valid.to(torch.float32)
            gathered_loss_sum = accelerator.gather_for_metrics(loss_sum_local)
            gathered_valid = accelerator.gather_for_metrics(local_valid)

            total_loss_sum += gathered_loss_sum.sum().item()
            total_valid_tokens += int(gathered_valid.sum().item())

            # Also keep a simple mean across processes (unweighted)
            gathered_loss = accelerator.gather_for_metrics(loss)
            simple_loss_sum += gathered_loss.sum().item()
            simple_loss_cnt += gathered_loss.numel()

    # Weighted/global metrics
    if total_valid_tokens > 0:
        eval_loss_weighted = total_loss_sum / total_valid_tokens
    else:
        eval_loss_weighted = float("nan")
    try:
        perplexity_weighted = math.exp(eval_loss_weighted)
    except OverflowError:
        perplexity_weighted = float("inf")

    # Simple mean for comparison
    if simple_loss_cnt > 0:
        eval_loss_simple = simple_loss_sum / simple_loss_cnt
        try:
            perplexity_simple = math.exp(eval_loss_simple)
        except OverflowError:
            perplexity_simple = float("inf")
    else:
        eval_loss_simple = float("nan")
        perplexity_simple = float("inf")

    model.train()

    # Reduce metrics across processes and normalize by total valid tokens.
    metrics_avg = {}
    if metric_wsum:
        reduced = reduce_weighted_metrics(
            accelerator,
            metric_wsum,
            metric_w,
            weight_key="num_valid",
        )
        metrics_avg = {k: v for k, v in reduced.items() if k != "num_valid"}

    if generalized:
        if not math.isnan(eval_loss_simple):
            print(f"perplexity_generalized - perplexity = {perplexity_weighted - perplexity_simple:.4f}")
        return eval_loss_weighted, perplexity_weighted, total_valid_tokens, metrics_avg
    else:
        return eval_loss_simple, perplexity_simple, None, metrics_avg


def compute_acc_math(
    model, dataloader, tokenizer, precision, aligned_labels, max_data=50000, debug=True, train_progress=1.0
):
    model.eval()
    device = get_model_device(model)
    sum_acc = 0
    acc_num = 0
    acc_dict = {"+": 0, "-": 0, "*": 0, "/": 0}
    count_dict = {"+": 0, "-": 0, "*": 0, "/": 0}
    operations_dict = {op: tokenizer.convert_tokens_to_ids(op) for op in ["+", "-", "*", "/"]}

    ctx = get_amp_context(precision)

    with torch.inference_mode(), ctx:
        iter_dl = tqdm(
            dataloader,
            total=len(dataloader) if hasattr(dataloader, "__len__") else None,
            desc="AccMath",
            leave=False,
        )
        for step, batch in enumerate(iter_dl):
            if acc_num >= max_data:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, train_progress=train_progress)
            logits = outputs.logits

            if aligned_labels:
                gt = batch["labels"]
                predicted_tokens = torch.argmax(logits, dim=-1)
            else:
                predicted_tokens = torch.argmax(logits, dim=-1)[:, :-1]
                gt = batch["labels"][:, 1:]
            del logits, outputs
            input_ids = batch["input_ids"]
            assert predicted_tokens.shape == gt.shape
            predicted_tokens[gt == -100] = -100
            acc = (predicted_tokens == gt).float().min(dim=-1).values
            for op in operations_dict.keys():
                res = torch.where(input_ids == operations_dict[op])
                if len(res[0]) > 0:
                    op_indices = res[0]
                    for i in range(len(op_indices)):
                        relevant = (
                            res[1][i]
                            < torch.where(input_ids[op_indices[i]] == tokenizer.convert_tokens_to_ids("="))[
                                0
                            ].min()
                        )
                        if relevant:
                            count_dict[op] += 1
                            acc_dict[op] += acc[op_indices[i]].item()
                    # relevant = res[1] < (torch.where(input_ids[res[0]] == tokenizer.convert_tokens_to_ids("="))[1])
                    # op_indices = op_indices[relevant]
                    # count_dict[op] += len(op_indices)
                    # acc_dict[op] += acc[op_indices].sum().item()
            sum_acc += acc.sum().item()
            acc_num += acc.shape[0]
            assert acc_num == sum(count_dict.values())
            if debug:
                predicted_tokens[gt == -100] = tokenizer.pad_token_id
                gt[gt == -100] = tokenizer.pad_token_id
                for b in range(len(batch["input_ids"])):
                    if step == 1 or tokenizer.decode(batch["input_ids"][b], skip_special_tokens=True) in [
                        "12-2=10",
                        "7*2=14",
                        "121+4=125",
                        "10+2=12",
                    ]:
                        print(
                            "input",
                            tokenizer.decode(batch["input_ids"][b], skip_special_tokens=True),
                        )
                        print(
                            "predicted",
                            tokenizer.decode(predicted_tokens[b], skip_special_tokens=True),
                        )
                        print("gt", tokenizer.decode(gt[b], skip_special_tokens=True))
                        print("acc", acc[b])
                        # print((predicted_tokens == gt), (predicted_tokens == gt).shape)
                        break

    avg_acc = sum_acc / acc_num
    assert acc_num == sum(count_dict.values())
    assert sum_acc == sum(acc_dict.values())
    for op in operations_dict.keys():
        if count_dict[op] > 0:
            acc_dict[op] = acc_dict[op] / count_dict[op]
        else:
            acc_dict[op] = -1
    model.train()
    return avg_acc, acc_dict


@retry_on_cuda_oom(batch_size_kw="batch_size_val")
def compute_acc_tta_pass2(
    model,
    dataset,
    *,
    num_augmentations: int = 100,
    batch_size_val: int = 10,
    precision: str = "bf16",
    tokenizer=None,
    train_progress: float = 1.0,
    tta_num_workers: int = 15,
    strict: bool = True,
    deterministic: bool = False,
):
    """Compute ARC-AGI test-time-augmentation accuracy (pass@2).

    For each underlying task `idx` in `dataset.data`, this function:
    1) Samples `num_augmentations` different augmentations by calling `dataset[idx]`.
    2) Uses `model.generate(...)` (as in `compute_acc` with `use_generate=True`) to
       predict each masked test output segment.
    3) Decodes predicted grids, applies inverse augmentation to map them back to
       the original task space.
    4) Votes per test-output index, keeps the top-2 voted grids.
    5) Scores 1 if, for every test output in the task, the GT grid is among the
       top-2 voted predictions; 0 otherwise.

        Requirements/assumptions
    ------------------------
    - `dataset` must be an `ArcAGI` with `aug=True` and `compute_mask_input=True`,
      and must return `aug_params` in its samples.
    - For meaningful inversion, pass `shuffle_examples=False`
      when building the eval dataset.
    - For batching (`batch_size_val>1`), augmentations may produce different
        `mask_input` spans (e.g. non-square grids under transpose/rot90). This
        implementation handles that by grouping augmentations by identical `mask_input`
        and batching within each group.
    - Only `use_generate=True` behavior is implemented here.
    - To parallelize augmentation sampling across processes, set
      `tta_num_workers>0` (uses a DataLoader with repeated indices).
    """

    if tokenizer is None:
        tokenizer = dataset.get_tokenizer() if hasattr(dataset, "get_tokenizer") else None
    if tokenizer is None:
        raise ValueError("tokenizer must be provided or dataset must support get_tokenizer()")

    dataset_aug = bool(getattr(dataset, "aug", False))
    if not getattr(dataset, "compute_mask_input", False):
        raise ValueError("dataset.compute_mask_input must be True for TTA")
    if getattr(dataset, "num_synthetic_data", 0) != 0:
        raise ValueError("TTA evaluation expects num_synthetic_data=0")
    if getattr(dataset, "reduce_train", False):
        raise ValueError("TTA evaluation expects dataset.reduce_train=False")
    if getattr(dataset, "shuffle_examples", True):
        raise ValueError("TTA evaluation expects dataset.shuffle_examples=False")

    # Hard constraints for batched generation correctness & stable spans.
    # These only matter when augmentations are enabled.
    if dataset_aug:
        if getattr(dataset, "max_aug_idx", 7) > 7:
            raise ValueError("TTA batching requires no upscale; set dataset.max_aug_idx<=7")
        if float(getattr(dataset, "shuffle_prob", 0.0)) != 0.0:
            raise ValueError("TTA batching requires no shuffle augmentation; set dataset.shuffle_prob=0.0")
    if getattr(dataset, "rand_think", False):
        raise ValueError("TTA batching requires deterministic think tokens; set dataset.rand_think=False")
    if int(getattr(dataset, "repeat_test", 1)) != 1:
        raise ValueError("TTA batching currently requires dataset.repeat_test=1")

    assert batch_size_val >= 1, "batch_size_val must be at least 1"

    model.eval()
    ctx = get_amp_context(precision)
    device = get_model_device(model)

    total = 0
    correct = 0

    tta_loader = build_tta_dataloader(
        dataset,
        num_augmentations,
        tta_num_workers=tta_num_workers,
        tta_loader_batch_size=batch_size_val,
    )
    tta_iter = iter(tta_loader)
    pending_batch = []

    with torch.inference_mode(), ctx:
        for idx in tqdm(
            range(len(dataset.data)),
            total=len(dataset.data),
            desc="TTA pass@2",
            leave=False,
        ):
            if idx % 5 == 0 or num_augmentations > 10:
                clean_gpu()
            base_task = dataset.data[idx]
            gt_outputs = [np.asarray(ex["output"], dtype=np.int32).tolist() for ex in base_task["test"]]
            num_test = len(gt_outputs)

            votes = [Counter() for _ in range(num_test)]

            num_aug = int(num_augmentations)
            samples = []
            starttime = time()
            while len(samples) < num_aug:
                if not pending_batch:
                    try:
                        pending_batch = next(tta_iter)
                    except StopIteration as exc:
                        raise RuntimeError("TTA dataloader exhausted early") from exc
                take = min(num_aug - len(samples), len(pending_batch))
                samples.extend(pending_batch[:take])
                pending_batch = pending_batch[take:]
            if random() < 0.1:
                print("time", time() - starttime)
            # Basic per-sample consistency checks.
            max_size = 0
            for s in samples:
                max_size = max(max_size, len(s["input_ids"]))
                if "idx" in s and int(s["idx"]) != int(idx):
                    raise AssertionError(f"TTA sample idx mismatch: got {s['idx']}, expected {idx}")
                if "mask" in s and s["mask"] is not None:
                    if len(s["mask"]) != len(s["input_ids"]):
                        raise AssertionError("mask length must match input_ids length")

            # Group by identical mask_input spans. Some dihedral transforms can
            # change rectangular grid shapes (e.g. transpose/rot90), which changes
            # the serialized prompt length and therefore mask_input boundaries.
            groups: dict[tuple[tuple[int, int], ...], list[int]] = {}
            for bi, s in enumerate(samples):
                key = tuple((int(a), int(b)) for a, b in s["mask_input"] if a >= 0)
                groups.setdefault(key, []).append(bi)

            for mask_key, idxs in groups.items():
                mask_input = [list(x) for x in mask_key]

                # Validate spans: monotonic and in-bounds for the padded sequence.
                prev_end = 0
                for start, length in mask_input:
                    start = int(start)
                    length = int(length)
                    if start < 0 or length < 0:
                        raise ValueError(f"Invalid mask_input span: start={start}, length={length}")
                    if start < prev_end:
                        raise ValueError(
                            f"mask_input spans must be non-overlapping and sorted, got start={start} < prev_end={prev_end}"
                        )
                    prev_end = start + length

                remaining = len(idxs)
                offset = 0
                factor_batch = max_size / max(1, prev_end)
                # print(f"prev_end {prev_end}, max_size {max_size}, factor_batch {factor_batch}")
                # print(f"bsz ", end=" ")
                while remaining > 0:
                    bsz = min(int(factor_batch * batch_size_val), remaining)
                    bsz = max(1, bsz)
                    # print(f"{bsz}", end=" ")
                    chunk_idxs = idxs[offset : offset + bsz]

                    # Basic per-chunk consistency checks.
                    L = len(samples[chunk_idxs[0]]["input_ids"])
                    for i in chunk_idxs:
                        if len(samples[i]["input_ids"]) != L:
                            raise AssertionError(
                                "input_ids length differs across batch; padding must be fixed"
                            )
                        if "mask" in samples[i] and samples[i]["mask"] is not None:
                            if len(samples[i]["mask"]) != L:
                                raise AssertionError("mask length must match input_ids length")

                    # Stack full prompts for this subgroup chunk.
                    input_ids = torch.as_tensor(
                        [samples[i]["input_ids"] for i in chunk_idxs],
                        dtype=torch.long,
                        device=device,
                    )
                    position_idx_full = None
                    if "position_idx" in samples[chunk_idxs[0]]:
                        position_idx_full = torch.as_tensor(
                            [samples[i]["position_idx"] for i in chunk_idxs],
                            dtype=torch.long,
                            device=device,
                        )
                    elif getattr(model.config, "use_rot_emb_2d", False) or getattr(
                        model.config, "use_pos_emb_2d", False
                    ):
                        raise ValueError(
                            "position_idx is required for ARC-AGI TTA generation with 2D embeddings"
                        )

                    if prev_end > input_ids.shape[1]:
                        raise ValueError(
                            f"mask_input span end {prev_end} exceeds sequence length {input_ids.shape[1]}"
                        )

                    # Batched generation for this subgroup chunk.
                    predicted_tokens_gen = torch.zeros(bsz, 0, dtype=input_ids.dtype, device=device)
                    for start, length in mask_input:
                        start = int(start)
                        length = int(length)
                        if start < predicted_tokens_gen.shape[1]:
                            raise AssertionError(
                                f"mask_input start {start} is behind generated length {predicted_tokens_gen.shape[1]}"
                            )
                        predicted_tokens_gen = torch.cat(
                            [predicted_tokens_gen, input_ids[:, predicted_tokens_gen.shape[1] : start]],
                            dim=1,
                        )
                        assert predicted_tokens_gen.shape[1] == start
                        position_idx_slice = None
                        if position_idx_full is not None:
                            position_idx_slice = position_idx_full[:, : start + length, :]
                        predicted_tokens_gen = model.generate(
                            predicted_tokens_gen,
                            None,
                            max_length=length,
                            temperature=1,
                            top_k=1 if not deterministic else 0,
                            eos_token_id=None,
                            return_cache=True,
                            train_progress=train_progress,
                            position_idx=position_idx_slice,
                        )

                    # Append any remaining tail from the original input_ids (padding etc.).
                    if predicted_tokens_gen.shape[1] < input_ids.shape[1]:
                        predicted_full = torch.cat(
                            [predicted_tokens_gen, input_ids[:, predicted_tokens_gen.shape[1] :]],
                            dim=1,
                        )
                    else:
                        predicted_full = predicted_tokens_gen[:, : input_ids.shape[1]]

                    predicted_full_list = predicted_full.tolist()

                    for local_bi, sample_idx in enumerate(chunk_idxs):
                        sample = samples[sample_idx]

                        # Decode & parse predicted grids (in augmented space).
                        decoded = decode_predicted_output_grids(
                            predicted_full_list[local_bi],
                            sample["mask"],
                            tokenizer,
                            num_test=num_test,
                            strict=strict,
                            skip_special_tokens=False,
                            mask_input=mask_input,
                        )

                        if dataset_aug:
                            aug_params = sample["aug_params"]
                            # print(aug_params)
                            if aug_params["did_shuffle"]:
                                raise AssertionError(
                                    "TTA sample unexpectedly shuffled examples; set shuffle_examples=False"
                                )

                            # Disallow any variable-size / tricky transforms.
                            if int(aug_params["transf_id_inp"]) in (8, 9):
                                raise AssertionError("TTA batching requires no upscale/shuffle transforms")
                            if int(aug_params["transf_id_out"]) in (8, 9):
                                raise AssertionError("TTA batching requires no upscale/shuffle transforms")

                            trans_id_out = int(aug_params["transf_id_out"])
                            mapping_color_out = aug_params["mapping_color_out"]
                            perm_out = aug_params["perm_out"]
                            assert mapping_color_out is not None, "Missing mapping_color_out in aug_params"

                            inv_map = inverse_augmentation(
                                trans_id_out,
                                np.asarray(mapping_color_out, dtype=np.uint8),
                                perm=np.asarray(perm_out, dtype=np.uint16) if perm_out is not None else None,
                            )
                        else:
                            # No augmentation: identity map back to original space.
                            inv_map = lambda x: x

                        for t in range(num_test):
                            g = decoded["grids"][t]
                            if g is None:
                                continue
                            try:
                                g_np = np.asarray(g, dtype=np.int32)
                                assert g_np.min() >= 0, "Negative values in predicted grid"
                                g_inv = inv_map(g_np)
                                g_key = tuple(tuple(int(v) for v in row) for row in g_inv.tolist())
                                votes[t][g_key] += 1
                            except Exception:
                                continue

                    remaining -= bsz
                    offset += bsz

                assert remaining == 0
                assert offset == len(idxs)
            # Select top-2 per test and compute pass@2.
            ok = True
            for t in range(num_test):
                top2 = [k for (k, _) in votes[t].most_common(2)]
                gt_key = tuple(tuple(int(v) for v in row) for row in gt_outputs[t])
                if gt_key not in top2:
                    ok = False
                    break

            correct += 1 if ok else 0
            total += 1
            if (idx + 1) % 20 == 0:
                print(f"TTA pass@2 interim: {100*correct/total:.2f}%")

    model.train()
    return (correct / total) if total > 0 else 0.0


def compute_acc(
    model,
    dataloader,
    use_generate,
    precision,
    aligned_labels,
    tokenizer=None,
    num_runs=1,
    train_progress=1.0,
    deterministic=False,
):
    model.eval()
    device = get_model_device(model)
    sum_acc = 0
    acc_num = 0
    acc_by_size = {}
    do_check = False
    ctx = get_amp_context(precision)

    with torch.inference_mode(), ctx:
        iter_dl = tqdm(
            dataloader,
            total=len(dataloader) if hasattr(dataloader, "__len__") else None,
            desc="Acc",
            leave=False,
        )
        for step, batch in enumerate(iter_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_generate:
                assert batch["input_ids"].shape[0] == 1, f"{batch['input_ids'].shape}"
                inp_gen = batch["input_ids"]
                position_idx_full = batch.get("position_idx")
                if position_idx_full is None and (
                    getattr(model.config, "use_rot_emb_2d", False)
                    or getattr(model.config, "use_pos_emb_2d", False)
                ):
                    raise ValueError("position_idx is required for ARC-AGI generation with 2D embeddings")
                mask_input = batch["mask_input"][0]
                predicted_tokens_gen = torch.zeros(1, 0, dtype=inp_gen.dtype, device=inp_gen.device)
                for start_t, length_t in mask_input:
                    start = int(start_t.item()) if isinstance(start_t, torch.Tensor) else int(start_t)
                    length = int(length_t.item()) if isinstance(length_t, torch.Tensor) else int(length_t)
                    if start < 0:
                        break
                    assert length > 0, f"length={length} must be positive"
                    predicted_tokens_gen = torch.cat(
                        [
                            predicted_tokens_gen,
                            inp_gen[:, predicted_tokens_gen.shape[1] : start],
                        ],
                        dim=1,
                    )
                    position_idx_slice = None
                    if position_idx_full is not None:
                        position_idx_slice = position_idx_full[:, : start + length, :]
                    predicted_tokens_gen = model.generate(
                        predicted_tokens_gen,
                        None,
                        max_length=length,
                        temperature=1,
                        top_k=1 if not deterministic else 0,
                        eos_token_id=None,
                        return_cache=True,
                        train_progress=train_progress,
                        position_idx=position_idx_slice,
                    )
                predicted_tokens_gen = predicted_tokens_gen[:, 1:]
            mask = batch["mask"]
            if aligned_labels:
                gt = batch["labels"]
            else:
                gt = batch["labels"][:, 1:]
                mask = mask[:, 1:]
            gt *= mask
            if not use_generate or do_check:
                if num_runs == 1:
                    outputs = model(**batch, eval_mode=True, train_progress=train_progress)
                    logits = outputs.logits
                    if aligned_labels:
                        predicted_tokens = torch.argmax(logits, dim=-1)
                    else:
                        predicted_tokens = torch.argmax(logits, dim=-1)[:, :-1]
                else:
                    predicted_tokens_list = []
                    for run_id in range(num_runs):
                        eval_mode = True if run_id == 0 else False
                        outputs = model(**batch, eval_mode=eval_mode, train_progress=train_progress)
                        logits = outputs.logits
                        if aligned_labels:
                            predicted_tokens_run = torch.argmax(logits, dim=-1)
                        else:
                            predicted_tokens_run = torch.argmax(logits, dim=-1)[:, :-1]
                        predicted_tokens_list.append(predicted_tokens_run)
                    predicted_tokens = torch.stack(predicted_tokens_list, dim=0).mode(dim=0).values
                del logits, outputs  # free ASAP
                predicted_tokens *= mask
                assert predicted_tokens.shape == gt.shape
            if use_generate:
                predicted_tokens_gen *= mask[:, : predicted_tokens_gen.shape[1]]

            assert (gt != -100).all()
            # predicted_tokens[gt == -100] = -100
            if use_generate and not do_check:
                assert gt[:, predicted_tokens_gen.shape[1] :].sum().item() == 0
                assert mask[:, predicted_tokens_gen.shape[1] :].sum().item() == 0
                gt = gt[:, : predicted_tokens_gen.shape[1]]
                acc = (predicted_tokens_gen == gt).float().min(dim=-1).values
            else:
                acc = (predicted_tokens == gt).float().min(dim=-1).values
            if use_generate and do_check:
                print(step, end=" ")
                acc_gen = (predicted_tokens_gen == gt).float().min(dim=-1).values
                if not (acc == acc_gen).all():
                    min_index = 0  # TO REMOVE
                    print(
                        "predicted_tokens    ",
                        tokenizer.decode(predicted_tokens[0, min_index:], skip_special_tokens=False),
                    )
                    print(
                        "predicted_tokens_gen",
                        tokenizer.decode(
                            predicted_tokens_gen[0, min_index:],
                            skip_special_tokens=False,
                        ),
                    )
                    print(
                        "gt                  ",
                        tokenizer.decode(gt[0, min_index:], skip_special_tokens=False),
                    )
                    breakpoint()
                assert (acc == acc_gen).all()
            sum_acc += acc.sum().item()
            acc_num += acc.shape[0]
            assert len(acc) == batch["size_input"].shape[0]
            for idx in range(len(acc)):
                if batch["size_input"][idx] > batch["size_output"][idx]:
                    size = int(batch["size_input"][idx]) // 10 * 10
                    size = f"{size}-{size+9}"
                    if size not in acc_by_size:
                        acc_by_size[size] = []
                    acc_by_size[size].append(acc[idx].item())
                else:
                    size = int(batch["size_output"][idx]) // 10 * 10
                    size = f"{size}-{size+9}"
                    if size not in acc_by_size:
                        acc_by_size[size] = []
                    acc_by_size[size].append(acc[idx].item())
    avg_acc = sum_acc / acc_num
    for k, v in acc_by_size.items():
        acc_by_size[k] = sum(v) / len(v) if v else 0
        print("acc", k, acc_by_size[k])
    model.train()
    return avg_acc


def sample_generations(dataset_name, model, tokenizer, precision, train_progress, steps2think=1):
    """Emit a few representative generations depending on the dataset."""

    prompts = []
    extra_kwargs = []

    if dataset_name == "custom:math_dataset":
        repetitions = steps2think if steps2think is not None else 1
        equal = "=" * repetitions
        prompts = [
            f"<bos>121+4{equal}",
            f"<bos>12-2{equal}",
            f"<bos>7*2{equal}",
            f"<bos>5021+5{equal}",
            f"<bos>5010-7{equal}",
        ]
        extra_kwargs = [{"temperature": 0.1, "top_k": 1}] * len(prompts)
    elif dataset_name == "OpenAssistant/oasst1":
        eos = tokenizer.eos_token
        prompts = [
            f"User: What is the capital of France?{eos}Assistant:",
            f"User: What is the meaning of life?{eos}Assistant:",
            f"User: How to train a model?{eos}Assistant:",
            f"User: What is the capital of Italy?{eos}Assistant:",
            f"User: What is the largest mammal?{eos}Assistant:",
            f"User: What is the largest mammal?{eos}Assistant:",
        ]
        extra_kwargs = [{}] * len(prompts)
    elif dataset_name != "custom:arc_agi":
        prompts = [
            "Once upon a time",
            "There are",
            "was born",
        ]
        extra_kwargs = [{}] * len(prompts)
    else:
        return

    for text, kwargs in zip(prompts, extra_kwargs):
        generate(
            model,
            tokenizer,
            text,
            precision,
            train_progress=train_progress,
            **kwargs,
        )


def compute_dataset_accuracy(
    args,
    model,
    train_dataloader_single,
    eval_dataloader_single,
    tokenizer,
    completed_steps,
    *,
    train_test_dataset=None,
    validation_test_dataset=None,
    tta_batch_size_val=10,
):
    """Compute and print accuracy metrics for supported custom datasets."""
    clean_gpu()
    acc_results = {
        "acc_train": None,
        "acc_val": None,
    }

    train_progress = completed_steps / args.max_train_steps

    if args.dataset_name == "custom:math_dataset":
        acc_results["acc_val"], acc_results["acc_dict_val"] = compute_acc_math(
            model,
            eval_dataloader_single,
            tokenizer,
            precision=args.precision,
            aligned_labels=args.aligned_labels,
            train_progress=train_progress,
        )
        acc_results["acc_train"], acc_results["acc_dict_train"] = compute_acc_math(
            model,
            train_dataloader_single,
            tokenizer,
            precision=args.precision,
            aligned_labels=args.aligned_labels,
            train_progress=train_progress,
        )

        print(f"Accuracy train: {acc_results['acc_train']*100:.1f}%")
        for op in acc_results["acc_dict_train"]:
            print(f"Accuracy train {op}: {acc_results['acc_dict_train'][op]*100:.1f}%")

        print(f"Accuracy val: {acc_results['acc_val']*100:.1f}%")
        for op in acc_results["acc_dict_val"]:
            print(f"Accuracy val {op}: {acc_results['acc_dict_val'][op]*100:.1f}%")

        # Flatten per-operation metrics into the result dict for downstream logging
        dict_str_ops = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
        }
        for op in acc_results["acc_dict_val"]:
            op_key = dict_str_ops[op]
            acc_results[f"acc_val_{op_key}"] = acc_results["acc_dict_val"][op]
            acc_results[f"acc_train_{op_key}"] = acc_results["acc_dict_train"][op]
        del acc_results["acc_dict_val"]
        del acc_results["acc_dict_train"]
    elif args.dataset_name == "custom:arc_agi":
        print("train")
        acc_results["acc_train"] = compute_acc(
            model,
            train_dataloader_single,
            use_generate=False,
            tokenizer=tokenizer,
            precision=args.precision,
            aligned_labels=args.aligned_labels,
            train_progress=train_progress,
            deterministic=args.debug,
        )
        if args.debug:
            # Generation-based (autoregressive) train accuracy, comparable to TTA(num_augmentations=1).
            # This can differ from acc_train because acc_train is teacher-forced next-token accuracy.
            acc_results["acc_train_generate"] = compute_acc(
                model,
                train_dataloader_single,
                use_generate=True,
                tokenizer=tokenizer,
                precision=args.precision,
                aligned_labels=args.aligned_labels,
                train_progress=train_progress,
                deterministic=args.debug,
            )
            print(f"Accuracy train (generate): {acc_results['acc_train_generate']*100:.2f}%")

        print("val")
        acc_results["acc_val"] = compute_acc(
            model,
            eval_dataloader_single,
            use_generate=args.use_generate,
            tokenizer=tokenizer,
            precision=args.precision,
            aligned_labels=args.aligned_labels,
            train_progress=train_progress,
            deterministic=args.debug,
        )
        print(f"Accuracy train: {acc_results['acc_train']*100:.2f}%")
        print(f"Accuracy val: {acc_results['acc_val']*100:.2f}%")

        assert train_test_dataset is not None and validation_test_dataset is not None

        if args.debug:
            print("train TTA pass@2 (num_augmentations=5)")
            acc_results["acc_train_tta_pass2"] = compute_acc_tta_pass2(
                model,
                train_test_dataset,
                num_augmentations=5,
                batch_size_val=5,
                precision=args.precision,
                tokenizer=tokenizer,
                train_progress=train_progress,
                deterministic=args.debug,
            )
            print(f"Accuracy train TTA pass@2: {acc_results['acc_train_tta_pass2']*100:.2f}%")

        print("val TTA pass@2")
        key_tta = "acc_val_tta_pass2"
        if args.num_runs_accuracy != 100:
            key_tta = f"acc_val_tta_pass2_{args.num_runs_accuracy}aug"
        acc_results[key_tta] = compute_acc_tta_pass2(
            model,
            validation_test_dataset,
            num_augmentations=args.num_runs_accuracy,
            batch_size_val=tta_batch_size_val,
            precision=args.precision,
            tokenizer=tokenizer,
            train_progress=train_progress,
            deterministic=args.debug,
        )
        print(f"Accuracy val TTA pass@2: {acc_results[key_tta]*100:.2f}%")
        if args.debug:
            acc_val_tta_pass2_10aug_strict = compute_acc_tta_pass2(
                model,
                validation_test_dataset,
                num_augmentations=10,
                batch_size_val=10,
                precision=args.precision,
                tokenizer=tokenizer,
                train_progress=train_progress,
                strict=True,
                deterministic=args.debug,
            )
            print(f"Accuracy val TTA pass@2 10aug strict: {acc_val_tta_pass2_10aug_strict*100:.2f}%")

            acc_val_tta_pass2_10aug_notstrict = compute_acc_tta_pass2(
                model,
                validation_test_dataset,
                num_augmentations=10,
                batch_size_val=10,
                precision=args.precision,
                tokenizer=tokenizer,
                train_progress=train_progress,
                strict=False,
                deterministic=args.debug,
            )
            print(f"Accuracy val TTA pass@2 10aug not strict: {acc_val_tta_pass2_10aug_notstrict*100:.2f}%")

            print("val multiple runs", 10)
            acc_results[f"acc_val_10runs"] = compute_acc(
                model,
                eval_dataloader_single,
                use_generate=args.use_generate,
                tokenizer=tokenizer,
                precision=args.precision,
                aligned_labels=args.aligned_labels,
                num_runs=10,
                train_progress=train_progress,
            )

    return acc_results
