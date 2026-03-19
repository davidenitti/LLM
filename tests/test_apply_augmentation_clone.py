import copy
import random

import numpy as np
import pytest

from arc_agi_dataset import apply_augmentation, clone_task_structure


def build_task():
    return {
        "task_id": "synthetic",
        "meta": {"source": "pytest"},
        "train": [
            {
                "input": [[0, 1], [2, 3]],
                "output": [[4, 5], [6, 7]],
            },
            {
                "input": [[1, 0], [0, 1]],
                "output": [[2, 2], [3, 3]],
            },
        ],
        "test": [
            {
                "input": [[8, 8], [9, 9]],
                "output": [[7, 6], [5, 4]],
            }
        ],
    }


def map_grid_inp(grid: np.ndarray) -> np.ndarray:
    return (grid + 1) % 10


def map_grid_out(grid: np.ndarray) -> np.ndarray:
    return np.rot90((grid + 2) % 10)


@pytest.mark.parametrize(
    ("shuffle_examples", "shuffle_train_examples", "random_values", "expected_did_shuffle"),
    [
        pytest.param(True, True, [0.1, 0.1], True, id="shuffle-train-test-together-and-swap-io"),
        pytest.param(True, True, [0.9, 0.1], True, id="shuffle-within-splits-and-swap-io"),
        pytest.param(False, True, [0.1], False, id="shuffle-train-only"),
    ],
)
def test_apply_augmentation_with_clone_task_structure_keeps_original_task_unchanged(
    monkeypatch,
    shuffle_examples,
    shuffle_train_examples,
    random_values,
    expected_did_shuffle,
):
    task = build_task()
    pristine = copy.deepcopy(task)

    random_iter = iter(random_values)
    monkeypatch.setattr(random, "random", lambda: next(random_iter))
    monkeypatch.setattr(random, "shuffle", lambda seq: seq.reverse())

    augmented_task, meta = apply_augmentation(
        clone_task_structure(task),
        map_grid_inp,
        map_grid_out,
        shuffle_examples=shuffle_examples,
        shuffle_train_examples=shuffle_train_examples,
    )

    assert task == pristine
    assert task["train"] is not augmented_task["train"]
    assert task["test"] is not augmented_task["test"]
    assert meta["did_shuffle"] is expected_did_shuffle
    assert augmented_task != pristine
