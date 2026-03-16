import random
from types import SimpleNamespace

import numpy as np
import torch

import arc_agi_dataset
from arc_agi_dataset import ArcAGI
from evaluation import compute_acc_tta_pass2


class FixedPatternGenerateModel(torch.nn.Module):
    """A tiny model that ignores the prompt and appends a fixed token pattern."""

    def __init__(self, pattern_ids):
        super().__init__()
        self.pattern_ids = torch.as_tensor(pattern_ids, dtype=torch.long)
        self.config = SimpleNamespace(use_rot_emb_2d=False, use_pos_emb_2d=False)
        # ensure model has parameters (device inference in compute_acc_tta_pass2)
        self._p = torch.nn.Parameter(torch.zeros(()))

    def generate(self, input_ids, _cache, *, max_length, **kwargs):
        # Repeat pattern to exactly max_length tokens.
        pattern = self.pattern_ids.to(device=input_ids.device)
        reps = int((max_length + pattern.numel() - 1) // pattern.numel())
        append_1 = pattern.repeat(reps)[:max_length].view(1, -1)
        append = append_1.expand(input_ids.size(0), -1)
        return torch.cat([input_ids, append], dim=1)


class _DigitTokenizer:
    """Minimal tokenizer compatible with decode_predicted_output_grids."""

    def decode(self, ids, skip_special_tokens=False):
        # Map 0-9 to their digit chars; everything else becomes '?'.
        out = []
        for tid in ids:
            try:
                tid_i = int(tid)
            except Exception:
                tid_i = -1
            if 0 <= tid_i <= 9:
                out.append(str(tid_i))
            else:
                out.append("?")
        return "".join(out)


class _ConstantTokenGenerateModel(torch.nn.Module):
    """A tiny model that appends a single token id repeatedly."""

    def __init__(self, token_id: int = 9):
        super().__init__()
        self.token_id = int(token_id)
        self.config = SimpleNamespace(use_rot_emb_2d=False, use_pos_emb_2d=False)
        self._p = torch.nn.Parameter(torch.zeros(()))

    def generate(self, input_ids, _cache, *, max_length, **kwargs):
        append = torch.full(
            (input_ids.size(0), int(max_length)),
            fill_value=self.token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, append], dim=1)


class _RecordBatchSizesGenerateModel(torch.nn.Module):
    """Model stub that records batch sizes passed to generate."""

    def __init__(self, token_id: int = 0):
        super().__init__()
        self.token_id = int(token_id)
        self.batch_sizes = []
        self.config = SimpleNamespace(use_rot_emb_2d=False, use_pos_emb_2d=False)
        self._p = torch.nn.Parameter(torch.zeros(()))

    def generate(self, input_ids, _cache, *, max_length, **kwargs):
        self.batch_sizes.append(int(input_ids.size(0)))
        append = torch.full(
            (input_ids.size(0), int(max_length)),
            fill_value=self.token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, append], dim=1)


class _AlternatingMaskInputDataset:
    """Dataset stub that alternates between two mask_input shapes.

    This simulates the real ARC-AGI behavior where some dihedral transforms
    (rot90/transpose) change prompt serialization length for non-square grids,
    yielding different mask_input spans across augmentations.
    """

    def __init__(self, padding: int = 64):
        self.data = [
            {
                "train": [],
                "test": [
                    {"input": [[0, 0]], "output": [[1, 2]]},
                ],
                "name": "alt_mask_input",
            }
        ]
        self.padding = int(padding)

        # Attributes required by compute_acc_tta_pass2.
        self.aug = True
        self.compute_mask_input = True
        self.num_synthetic_data = 0
        self.reduce_train = False
        self.shuffle_examples = False
        self.max_aug_idx = 7
        self.shuffle_prob = 0.0
        self.rand_think = False
        self.repeat_test = 1

        self._counter = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx == 0
        self._counter += 1

        # Same start, different length -> different mask_input keys.
        if (self._counter % 2) == 0:
            mask_input = [[5, 3]]
            trans_id_out = 0
        else:
            mask_input = [[5, 4]]
            trans_id_out = 6  # transpose; stands in for non-square changing serialization

        # Match ArcAGI behavior: pad mask_input to a fixed length with sentinel spans.
        # This ensures HF default_data_collator can stack mask_input for batch_size>1.
        for _ in range(32 - len(mask_input)):
            mask_input.append([-1, -1])

        input_ids = [0] * self.padding
        mask = [0] * self.padding

        aug_params = {
            "did_shuffle": False,
            "transf_id_inp": 0,
            "transf_id_out": trans_id_out,
            "mapping_color_out": np.arange(10, dtype=np.uint8),
            "perm_out": None,
        }

        return {
            "input_ids": input_ids,
            "mask": mask,
            "mask_input": mask_input,
            "aug_params": aug_params,
        }


def _make_tta_dataset():
    # Use 1x1 outputs so the output text is always "0\n" (no '|' row separators).
    task = {
        "train": [
            {"input": [[0]], "output": [[0]]},
            {"input": [[1]], "output": [[1]]},
        ],
        "test": [
            {"input": [[2]], "output": [[0]]},
            {"input": [[3]], "output": [[0]]},
        ],
        "name": "tta_unit",
    }

    ds = ArcAGI(
        data=[task],
        padding=128,
        aug=True,
        text_out=True,
        truncate=False,
        gt_labels_only=False,
        causal=True,
        think_budget=0,
        rand_think=False,
        aug_inp_out=False,
        max_aug_idx=7,  # avoid upscale (id 8)
        num_synthetic_data=0,
        noise_prob=0.0,
        use_unk_token_for_gt=False,
        repeat_test=1,
        encode=True,
        compute_mask_input=True,
        exclude_black=True,
        shuffle_prob=0.0,  # avoid shuffle (id 9)
        aligned_labels=False,
        shuffle_examples=False,
        reduce_train=False,
        return_aug_params=True,
    )
    return ds


def test_compute_acc_tta_pass2_perfect():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ds = _make_tta_dataset()
    tok = ds.get_tokenizer()

    # Always generate "0\n" for each test output span.
    pattern_ids = tok("0\n", add_special_tokens=False)["input_ids"]
    model = FixedPatternGenerateModel(pattern_ids)

    acc = compute_acc_tta_pass2(
        model,
        ds,
        num_augmentations=6,
        batch_size_val=3,
        precision="float32",
        tokenizer=tok,
        tta_num_workers=0,
    )
    assert acc == 1.0


def test_compute_acc_tta_pass2_always_wrong():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ds = _make_tta_dataset()
    tok = ds.get_tokenizer()

    # Always generate "1\n" -> after inverse color mapping it's never 0 when exclude_black=True.
    pattern_ids = tok("1\n", add_special_tokens=False)["input_ids"]
    model = FixedPatternGenerateModel(pattern_ids)

    acc = compute_acc_tta_pass2(
        model,
        ds,
        num_augmentations=6,
        batch_size_val=3,
        precision="float32",
        tokenizer=tok,
        tta_num_workers=0,
    )
    assert acc == 0.0


def test_compute_acc_tta_pass2_groups_by_mask_input_spans():
    """Regression: batching must not assume identical mask_input across augmentations."""

    torch.manual_seed(0)
    ds = _AlternatingMaskInputDataset(padding=64)
    tok = _DigitTokenizer()
    model = _ConstantTokenGenerateModel(token_id=9)

    acc = compute_acc_tta_pass2(
        model,
        ds,
        num_augmentations=8,
        batch_size_val=8,
        precision="float32",
        tokenizer=tok,
        tta_num_workers=0,
    )

    # We only care that it runs; the stub model is not expected to solve the task.
    assert isinstance(acc, float)


def test_compute_acc_tta_pass2_batches_within_group():
    """Ensure batch sizing can scale within each mask_input group."""

    torch.manual_seed(0)
    ds = _AlternatingMaskInputDataset(padding=64)
    tok = _DigitTokenizer()
    model = _RecordBatchSizesGenerateModel(token_id=0)

    acc = compute_acc_tta_pass2(
        model,
        ds,
        num_augmentations=10,
        batch_size_val=4,
        precision="float32",
        tokenizer=tok,
        tta_num_workers=0,
    )

    assert isinstance(acc, float)
    assert sorted(model.batch_sizes) == [5, 5]


def test_compute_acc_tta_pass2_real_arcagi_non_square_span_variation(monkeypatch):
    """Integration-ish regression: real ArcAGI can yield different mask_input spans.

    We force alternating transforms (identity vs transpose) on a non-square output,
    which changes prompt serialization length (number of '|' separators), hence
    changes mask_input boundaries across augmentations.

    The main assertion is that compute_acc_tta_pass2 runs without error while
    batching/grouping by mask_input.
    """

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Non-square output: transpose changes number of rows -> changes '|' count -> changes text length.
    task = {
        "train": [
            {"input": [[0, 1, 2], [3, 4, 5]], "output": [[9, 9, 9], [8, 8, 8]]},
        ],
        "test": [
            {"input": [[6, 7, 8], [9, 0, 1]], "output": [[0, 0, 0], [0, 0, 0]]},
        ],
        "name": "non_square_span_variation",
    }

    ds = ArcAGI(
        data=[task],
        padding=256,
        aug=True,
        text_out=True,
        truncate=False,
        gt_labels_only=False,
        causal=True,
        gt_all_out=True,
        think_budget=0,
        rand_think=False,
        aug_inp_out=False,
        single_inp_out_coloraug_prob=0.0,
        max_aug_idx=7,
        num_synthetic_data=0,
        noise_prob=0.0,
        use_unk_token_for_gt=False,
        repeat_test=1,
        encode=True,
        compute_mask_input=True,
        exclude_black=True,
        shuffle_prob=0.0,
        aligned_labels=False,
        shuffle_examples=False,
        reduce_train=False,
        return_aug_params=True,
    )

    # Force alternating augmentation ids: 0 (identity) then 6 (transpose), repeating.
    seq = [0, 6]
    state = {"i": 0}

    def _alt_randint(a, b):
        # ArcAGI calls randint for transf_id in [0, max_aug_idx_used].
        v = seq[state["i"] % len(seq)]
        state["i"] += 1
        assert a <= v <= b
        return v

    monkeypatch.setattr(arc_agi_dataset.random, "randint", _alt_randint)

    tok = ds.get_tokenizer()

    # Pre-check: two consecutive samples must produce different mask_input spans.
    s1 = ds[0]
    s2 = ds[0]
    k1 = tuple((int(a), int(b)) for a, b in s1["mask_input"])
    k2 = tuple((int(a), int(b)) for a, b in s2["mask_input"])
    assert k1 != k2, "Expected mask_input spans to differ under transpose for non-square grids"

    # Generate something that parses as a grid (even if it's not the correct answer).
    pattern_ids = tok("000|000\n", add_special_tokens=False)["input_ids"]
    model = FixedPatternGenerateModel(pattern_ids)

    acc = compute_acc_tta_pass2(
        model,
        ds,
        num_augmentations=6,
        batch_size_val=6,
        precision="float32",
        tokenizer=tok,
        tta_num_workers=0,
    )
    assert isinstance(acc, float)
