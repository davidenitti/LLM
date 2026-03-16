import math
import torch
import pytest

from evaluation import compute_valv2

# NOTE: ForCausalLMLoss would remove the last token for each chunk!


class FakeAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def gather_for_metrics(self, x):
        # Identity in single-process tests; ensure tensor output
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    def reduce(self, tensor, reduction="mean"):
        # Single-process identity.
        return tensor


class DummyOutput:
    def __init__(self, loss, hidden_states=None, metrics=None):
        self.loss = loss
        self.hidden_states = hidden_states
        self.metrics = metrics


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids=None, labels=None, hidden=None, **kwargs):
        """
        Per-sequence constant loss c, invariant to detach boundaries:
        - On first mini-step (hidden is None), compute c from input_ids[0,0].
        - On subsequent mini-steps, reuse c via `hidden` so loss stays consistent.
        """
        assert input_ids is not None and labels is not None
        device = input_ids.device
        if hidden is None:
            c = (input_ids[0, 0].float() / 10.0).to(torch.float32)
        else:
            c = torch.as_tensor(hidden, dtype=torch.float32, device=device)
        loss = c.clone()
        # compute_valv2 requires outputs.metrics['num_valid'] for weighting.
        # For these tests, local_valid equals (labels != -100).sum() for both aligned/un-aligned cases.
        num_valid = (labels != -100).sum()
        return DummyOutput(loss=loss, hidden_states=c, metrics={"num_valid": num_valid})


def make_batch(batch_size, seq_len, c_value, valid_tokens_per_seq, aligned_labels=False):
    device = torch.device("cpu")
    # input_ids: encode c_value in [0,0] position as c*10
    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    input_ids[:, 0] = int(round(c_value * 10))

    # labels: set first position arbitrary (ignored by eval functions which use labels[:,1:])
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    # Mark positions 1..valid_tokens_per_seq as valid (any token value)
    additional_offset = 1 if not aligned_labels else 0
    if valid_tokens_per_seq > 0:
        labels[:, additional_offset : additional_offset + valid_tokens_per_seq] = 1

    batch = {
        "input_ids": input_ids,
        "labels": labels,
    }
    return batch


@pytest.mark.parametrize("aligned_labels", [False, True])
def test_compute_valv2_generalized_no_detach(aligned_labels):
    model = DummyModel()
    model.eval()
    accel = FakeAccelerator()

    # Build two batches with different c and different valid token counts
    # Batch1: c=0.5, valid per seq=2, B=2 -> valid tokens = 4
    # Batch2: c=1.0, valid per seq=8 if aligned_labels else 6, B=2
    valid_b2 = 8 if aligned_labels else 6
    b1 = make_batch(
        batch_size=2,
        seq_len=8,
        c_value=0.5,
        valid_tokens_per_seq=2,
        aligned_labels=aligned_labels,
    )
    b2 = make_batch(
        batch_size=2,
        seq_len=8,
        c_value=1.0,
        valid_tokens_per_seq=valid_b2,
        aligned_labels=aligned_labels,
    )
    dl = [b1, b2]

    # Expected weighted loss: (0.5*4 + 1.0*valid_b2*2) / (4 + valid_b2*2)
    expected = (0.5 * 4 + 1.0 * valid_b2 * 2) / (4 + valid_b2 * 2)

    loss2, ppl2, valid2, metrics2 = compute_valv2(
        model,
        dl,
        accel,
        detach_every=None,
        generalized=True,
        precision="fp16",
        aligned_labels=aligned_labels,
        train_progress=1.0,
    )

    assert pytest.approx(loss2, rel=1e-6) == expected
    assert pytest.approx(ppl2, rel=1e-6) == math.exp(expected)
    # total valid tokens depend on aligned_labels: 16 (unaligned) or 20 (aligned)
    expected_valid = 16 if not aligned_labels else 20
    assert valid2 == expected_valid
    assert metrics2 == {}


@pytest.mark.parametrize("aligned_labels", [False, True])
def test_compute_valv2_non_generalized_mean_no_detach(aligned_labels):
    model = DummyModel()
    model.eval()
    accel = FakeAccelerator()

    valid_b2 = 8 if aligned_labels else 6
    b1 = make_batch(
        batch_size=2,
        seq_len=8,
        c_value=0.5,
        valid_tokens_per_seq=2,
        aligned_labels=aligned_labels,
    )
    b2 = make_batch(
        batch_size=2,
        seq_len=8,
        c_value=1.0,
        valid_tokens_per_seq=valid_b2,
        aligned_labels=aligned_labels,
    )
    dl = [b1, b2]

    # Expected simple mean of losses across steps (unweighted by valid tokens)
    expected_mean = (0.5 + 1.0) / 2.0

    loss2, ppl2, valid2, metrics2 = compute_valv2(
        model,
        dl,
        accel,
        detach_every=None,
        generalized=False,
        precision="fp16",
        aligned_labels=aligned_labels,
        train_progress=1.0,
    )

    assert valid2 is None

    assert pytest.approx(loss2, rel=1e-6) == expected_mean
    assert pytest.approx(ppl2, rel=1e-6) == math.exp(expected_mean)
    assert metrics2 == {}


@pytest.mark.parametrize("aligned_labels", [True])
def test_compute_valv2_generalized_with_detach(aligned_labels):
    model = DummyModel()
    model.eval()
    accel = FakeAccelerator()

    # Same batches but use detach_every that splits seq_len in two
    seq_len = 8
    valid_b2 = 8 if aligned_labels else 5
    b1 = make_batch(
        batch_size=2,
        seq_len=seq_len,
        c_value=0.5,
        valid_tokens_per_seq=3,
        aligned_labels=aligned_labels,
    )
    b2 = make_batch(
        batch_size=2,
        seq_len=seq_len,
        c_value=1.0,
        valid_tokens_per_seq=valid_b2,
        aligned_labels=aligned_labels,
    )
    dl = [b1, b2]

    # Total valid tokens: (2*3) + (2*valid_b2) = 6 + 2*valid_b2
    total_valid = 6 + 2 * valid_b2
    expected = (0.5 * 6 + 1.0 * (2 * valid_b2)) / float(total_valid)

    loss2, ppl2, valid2, metrics2 = compute_valv2(
        model,
        dl,
        accel,
        detach_every=seq_len // 2,
        generalized=True,
        precision="fp16",
        aligned_labels=aligned_labels,
        train_progress=1.0,
    )

    assert pytest.approx(loss2, rel=1e-6) == expected
    assert pytest.approx(ppl2, rel=1e-6) == math.exp(expected)
    # valid token count matches total_valid: 16 (unaligned) or 22 (aligned)
    assert valid2 == total_valid
    assert metrics2 == {}
