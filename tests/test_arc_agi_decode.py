import numpy as np
import arc_agi_dataset

from arc_agi_dataset import ArcAGI, decode_predicted_output_grids


def test_decode_predicted_output_grids_recovers_test_output():
    # Deterministic small task; aug=False so no random IO swaps/shuffles.
    task = {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
            {"input": [[4, 5], [6, 7]], "output": [[7, 6], [5, 4]]},
        ],
        "test": [
            {"input": [[8, 9], [0, 1]], "output": [[1, 0], [9, 8]]},
        ],
        "name": "unit",
    }

    ds = ArcAGI(
        data=[task],
        padding=256,
        aug=False,
        text_out=True,
        truncate=False,
        gt_labels_only=False,
        causal=True,
        think_budget=0,
        rand_think=False,
        repeat_test=1,
        encode=True,
        compute_mask_input=False,
    )

    sample = ds[0]
    tokenizer = ds.get_tokenizer()

    # Use ground-truth tokens as "predictions".
    decoded = decode_predicted_output_grids(
        sample["input_ids"],
        sample["mask"],
        tokenizer,
        num_test=1,
        strict=True,
        skip_special_tokens=False,
    )

    assert len(decoded["grids"]) == 1
    assert decoded["grids"][0] == task["test"][0]["output"]


def test_decode_predicted_output_grids_handles_missing_grids():
    # If mask has no 1s, we should return None when num_test is requested.
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_file="arc_agi_tokenizer.json",
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    pred = [tok.bos_token_id] + [0] * 10
    mask = [0] * len(pred)

    decoded = decode_predicted_output_grids(pred, mask, tok, num_test=2)
    assert decoded["grids"] == [None, None]


def test_decode_strictness_for_trailing_separator_mismatch():
    task = {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
        ],
        "test": [
            {"input": [[4, 5], [6, 7]], "output": [[7, 6], [5, 4]]},
        ],
        "name": "unit_token_mismatch",
    }

    ds = ArcAGI(
        data=[task],
        padding=256,
        aug=False,
        text_out=True,
        truncate=False,
        gt_labels_only=False,
        causal=True,
        think_budget=0,
        rand_think=False,
        repeat_test=1,
        encode=True,
        compute_mask_input=True,
    )

    sample = ds[0]
    tokenizer = ds.get_tokenizer()
    mask_input = [span for span in sample["mask_input"] if span[0] >= 0]

    decoded = decode_predicted_output_grids(
        sample["input_ids"],
        sample["mask"],
        tokenizer,
        num_test=1,
        strict=True,
        skip_special_tokens=False,
        mask_input=mask_input,
    )
    assert decoded["grids"][0] == task["test"][0]["output"]

    predicted = list(sample["input_ids"])
    pipe_id = tokenizer("|", add_special_tokens=False)["input_ids"][0]
    for start, length in mask_input:
        end_idx = int(start) + int(length) - 1
        assert tokenizer.decode([predicted[end_idx]], skip_special_tokens=False) == "\n"
        predicted[end_idx] = pipe_id

    # Token-level mismatch exists on masked positions (would fail compute_acc).
    mismatch = any(
        (predicted[i] != sample["labels"][i]) for i in range(1, len(predicted)) if sample["mask"][i] == 1
    )
    assert mismatch
    decoded_modified_relaxed = decode_predicted_output_grids(
        predicted,
        sample["mask"],
        tokenizer,
        num_test=1,
        strict=False,
        skip_special_tokens=False,
        mask_input=mask_input,
    )
    assert decoded_modified_relaxed["grids"][0] == task["test"][0]["output"]

    decoded_modified_strict = decode_predicted_output_grids(
        predicted,
        sample["mask"],
        tokenizer,
        num_test=1,
        strict=True,
        skip_special_tokens=False,
        mask_input=mask_input,
    )
    assert decoded_modified_strict["grids"][0] is None


def test_arcagi_can_skip_optional_training_metadata_fields():
    task = {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
            {"input": [[4, 5], [6, 7]], "output": [[7, 6], [5, 4]]},
        ],
        "test": [
            {"input": [[8, 9], [0, 1]], "output": [[1, 0], [9, 8]]},
        ],
        "name": "unit_trimmed_training_fields",
    }

    ds = ArcAGI(
        data=[task],
        padding=256,
        aug=False,
        text_out=True,
        truncate=False,
        gt_labels_only=True,
        causal=True,
        think_budget=0,
        rand_think=False,
        repeat_test=1,
        encode=True,
        compute_mask_input=False,
        include_position_idx=False,
    )

    sample = ds[0]

    assert "mask_input" not in sample
    assert "position_idx" not in sample
    assert {"input_ids", "labels", "mask", "idx", "size_input", "size_output", "task_size"} <= set(sample)


def test_arcagi_disables_upscale_for_wide_tasks(monkeypatch):
    task = {
        "train": [
            {"input": [[0] * 12 for _ in range(4)], "output": [[1] * 12 for _ in range(4)]},
            {"input": [[2] * 12 for _ in range(4)], "output": [[3] * 12 for _ in range(4)]},
        ],
        "test": [
            {"input": [[4] * 12 for _ in range(4)], "output": [[5] * 12 for _ in range(4)]},
        ],
        "name": "wide_rectangular_task",
    }

    ds = ArcAGI(
        data=[task],
        padding=512,
        aug=True,
        text_out=True,
        truncate=False,
        gt_labels_only=True,
        causal=True,
        think_budget=0,
        rand_think=False,
        max_aug_idx=8,
        repeat_test=1,
        encode=True,
        compute_mask_input=False,
        shuffle_prob=0.0,
        shuffle_examples=False,
        reduce_train=False,
        return_aug_params=True,
    )

    monkeypatch.setattr(arc_agi_dataset.random, "random", lambda: 1.0)
    monkeypatch.setattr(arc_agi_dataset.random, "randint", lambda a, b: b)

    sample = ds[0]

    assert sample["aug_params"]["transf_id_inp"] == 7
    assert sample["aug_params"]["transf_id_out"] == 7
    assert sample["size_input"] == 12
    assert sample["size_output"] == 12
