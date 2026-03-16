import numpy as np
import pytest

from arc_agi_dataset import augmentation, inverse_augmentation, map_color


@pytest.mark.parametrize("exclude_black", [True, False])
@pytest.mark.parametrize("trans_id", list(range(0, 10)))
def test_augmentation_inverse_roundtrip(trans_id: int, exclude_black: bool):
    rng = np.random.default_rng(1234 + trans_id + (100 if exclude_black else 0))
    mapping = map_color(exclude_black=exclude_black)

    # keep shapes small; upscale (id=8) will double them.
    h = int(rng.integers(2, 8))
    w = int(rng.integers(2, 8))
    grid = rng.integers(0, 10, size=(h, w), dtype=np.uint8)

    perm = None
    if trans_id == 9:
        # For shuffle, provide a deterministic perm so inverse is well-defined.
        perm = rng.permutation(np.arange(0, 1000, dtype=np.uint16))

    fwd, mapping_used = augmentation(
        trans_id, mapping_color=mapping, exclude_black=exclude_black, perm=perm
    )
    inv = inverse_augmentation(trans_id, mapping_color=mapping_used, perm=perm)

    out = fwd(grid)
    restored = inv(out)

    assert restored.shape == grid.shape
    assert np.array_equal(restored, grid)
