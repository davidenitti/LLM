import pytest

from arc_agi_dataset import clean_prompt, get_position_idx


def test_clean_prompt_positions_basic():
    s = "I[[1, 2], [3, 4]]"

    cleaned = clean_prompt(s)
    assert cleaned == "I12|34\n"

    pos = get_position_idx(cleaned)
    assert len(pos) == len(cleaned)

    # I 1 2 | 3 4 \n (all positions shifted by +10)
    assert pos[0] == (9, 9)  # 'I'
    assert pos[1] == (10, 10)
    assert pos[2] == (10, 11)
    assert pos[3] == (7, 7)  # '|'
    assert pos[4] == (11, 10)
    assert pos[5] == (11, 11)
    assert pos[6] == (6, 6)  # '\n'


def test_clean_prompt_positions_all_none_for_non_digits():
    cleaned = clean_prompt("O[[0]]")
    pos = get_position_idx(cleaned)
    assert pos[0] == (8, 8)
    assert pos[1] == (10, 10)
    assert pos[2] == (6, 6)
