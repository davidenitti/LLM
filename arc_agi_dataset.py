import glob
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import torch
from transformers import PreTrainedTokenizerFast
import random
from matplotlib.colors import ListedColormap
import copy


def pad_grid(grid, *, pad_value: int = -1, max_size: int = 30) -> torch.Tensor:
    """Pad a 2D ARC grid to exactly ``max_size x max_size``."""

    arr = np.asarray(grid, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {arr.shape}")
    assert (
        arr.shape[0] <= max_size and arr.shape[1] <= max_size
    ), f"Grid shape {arr.shape} exceeds max_size {max_size}"
    h, w = arr.shape
    out = np.full((max_size, max_size), int(pad_value), dtype=np.int32)
    out[:h, :w] = arr[:h, :w]
    return torch.from_numpy(out)


def to_1d_list(x):
    """Convert common tensor/array/list inputs to a Python 1D list."""

    if torch.is_tensor(x):
        x = x.detach().cpu().tolist()
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif not isinstance(x, list):
        x = list(x)
    return x


def parse_arc_grid_text(grid_text: str, *, strict: bool = True):
    """Parse a single ARC grid from the compact text format

    The expected format is a sequence of rows separated by `|`, with each row being
    a sequence of digits 0-9 (e.g. `"012|340"`).

    Parameters
    ----------
    grid_text : str
        The grid content without the leading `O` marker and without the trailing newline.
    strict : bool, optional
        If True, reject any non-digit characters inside rows and require a rectangular grid.

    Returns
    -------
    list[list[int]] | None
        Parsed grid, or None if the input is invalid.
    """

    if grid_text is None:
        return None
    grid_text = str(grid_text)
    if strict:
        if grid_text == "":
            return None
        if grid_text.strip() != grid_text:
            return None
        rows_raw = grid_text.split("|")
        if any(r == "" for r in rows_raw):
            return None
    else:
        grid_text = grid_text.strip()
        if grid_text == "":
            return None
        rows_raw = [r for r in grid_text.split("|") if r != ""]
        if not rows_raw:
            return None

    rows = []
    for r in rows_raw:
        if strict:
            if not r.isdigit():
                return None
            digits = list(r)
        else:
            r = r.strip()
            digits = [ch for ch in r if ch.isdigit()]
            if not digits:
                continue
        rows.append([int(ch) for ch in digits])

    if not rows:
        return None

    if strict:
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            return None
    return rows


def decode_predicted_output_grids(
    predicted_token_ids,
    mask,
    tokenizer,
    *,
    num_test: int | None = None,
    strict: bool = True,
    skip_special_tokens: bool = False,
    mask_input: list[list[int]] | None = None,
):
    """Decode and extract predicted ARC output grids from a masked token sequence.

    This is intended for ARC-AGI prompts produced by `create_prompt`, where:
    - Each test output grid is terminated by a newline (`\n`).
    - Rows are separated by `|`.
    - `mask` (typically `tokens["mask"]`) is 1 exactly on positions belonging to the
      test output(s) that should be evaluated (excluding the leading `O` marker).

    Parameters
    ----------
    predicted_token_ids : 1D sequence (list/np/torch)
        The model's predicted token ids for the full sequence (same length as `mask`).
    mask : 1D sequence (list/np/torch)
        Binary mask (same length as `predicted_token_ids`) indicating which token
        positions belong to the test output(s).
    tokenizer : PreTrainedTokenizerFast-like
        Must provide `.decode(list_of_ids, skip_special_tokens=...)`.
    num_test : int | None, optional
        If provided, return exactly this many grids (missing ones become `None`,
        extras are truncated).
    strict : bool, optional
        If True, require rectangular grids and reject non-digit characters.
    skip_special_tokens : bool, optional
        Passed to tokenizer.decode.
    mask_input : list[[start, length]] | None, optional
        Optional spans (as produced by `create_prompt(..., compute_mask_input=True)`).
        If provided, grids are decoded per-span rather than split by newlines.

    Returns
    -------
    dict
        Keys:
        - `decoded_text`: str (concatenated decoded masked output)
        - `grid_texts`: list[str]
        - `grids`: list[list[list[int]] | None]
    """

    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    pred = to_1d_list(predicted_token_ids)
    m = to_1d_list(mask)
    if len(pred) != len(m):
        raise ValueError(f"predicted_token_ids and mask must have same length, got {len(pred)} and {len(m)}")

    # If we have explicit spans for each test output, decode per-span.
    if mask_input is not None:
        grid_texts = []
        decoded_chunks = []
        for start, length in mask_input:
            start = int(start)
            length = int(length)
            if length <= 0:
                grid_texts.append("")
                decoded_chunks.append("")
                continue
            # mask_input spans correspond to the output content (excluding the leading 'O')
            # so we decode exactly that slice.
            ids_slice = pred[start : start + length]
            decoded = tokenizer.decode(ids_slice, skip_special_tokens=skip_special_tokens)
            decoded_chunks.append(decoded)
            if strict:
                if decoded.count("\n") != 1 or not decoded.endswith("\n"):
                    grid_texts.append("")
                else:
                    grid_texts.append(decoded[:-1])
            else:
                grid_texts.append(decoded.rstrip("\n"))

        decoded_text = "".join(decoded_chunks)
    else:
        masked_ids = [pid for pid, mi in zip(pred, m) if int(mi) == 1]
        decoded_text = tokenizer.decode(masked_ids, skip_special_tokens=skip_special_tokens)

        # Split into one grid per newline terminator.
        parts = decoded_text.split("\n")
        if strict:
            if decoded_text == "":
                grid_texts = []
            else:
                if decoded_text.endswith("\n"):
                    parts = parts[:-1]
                elif parts:
                    parts[-1] = ""
                grid_texts = parts
        else:
            grid_texts = [p for p in (s.strip() for s in parts) if p != ""]

    grids = []
    for grid_text in grid_texts:
        grids.append(parse_arc_grid_text(grid_text, strict=strict))

    if num_test is not None:
        num_test = int(num_test)
        if num_test < 0:
            raise ValueError("num_test must be >= 0")
        if len(grids) < num_test:
            grids = grids + [None] * (num_test - len(grids))
            grid_texts = grid_texts + [""] * (num_test - len(grid_texts))
        else:
            grids = grids[:num_test]
            grid_texts = grid_texts[:num_test]

    return {
        "decoded_text": decoded_text,
        "grid_texts": grid_texts,
        "grids": grids,
    }


def map_color(exclude_black: bool) -> np.ndarray:
    if exclude_black:
        mapping_color = np.concatenate(
            [
                np.arange(0, 1, dtype=np.uint8),
                np.random.permutation(np.arange(1, 10, dtype=np.uint8)),
            ]
        )  # Permute colors, Excluding "0" (black)
    else:
        mapping_color = np.random.permutation(
            np.arange(0, 10, dtype=np.uint8)
        )  # Permute colors including "0" (black)
    return mapping_color


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """
    from https://github.com/arcprize/hierarchical-reasoning-model-analysis/
    8 dihedral symmetries by rotate, flip and mirror
    """

    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)  # horizontal flip
    elif tid == 5:
        return np.flipud(arr)  # vertical flip
    elif tid == 6:
        return arr.T  # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        raise ValueError(f"Invalid transformation id: {tid}")


def extended_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    if tid < 8:
        return dihedral_transform(arr, tid)
    elif tid == 8:  # upscale
        scale = 2
        upscaled = np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)
        return upscaled
    else:
        raise ValueError(f"Invalid transformation id: {tid}")


def shuffle_transform(
    numel: int,
    mapping_color: np.ndarray = None,
    exclude_black=True,
    perm: np.ndarray | None = None,
):
    if perm is None:
        perm = np.random.permutation(np.arange(0, numel, dtype=np.uint16))
    else:
        perm = np.asarray(perm)
        if perm.shape != (numel,):
            raise ValueError(f"perm must have shape ({numel},), got {perm.shape}")
        if perm.dtype != np.uint16:
            perm = perm.astype(np.uint16, copy=False)
    if mapping_color is None:
        mapping_color = map_color(exclude_black=exclude_black)
    else:
        mapping_color = np.asarray(mapping_color)
        if mapping_color.shape != (10,):
            raise ValueError(f"mapping_color must have shape (10,), got {mapping_color.shape}")

    def map_grid(grid: np.ndarray):
        mapped_grid = mapping_color[grid]
        assert mapped_grid.shape == grid.shape
        # assert grid.size == perm.shape[0]
        if grid.size < perm.shape[0]:
            perm_used = perm[perm < grid.size]
        else:
            assert grid.size == perm.shape[0]
            perm_used = perm
        return mapped_grid.reshape(
            -1,
        )[
            perm_used
        ].reshape(*grid.shape)

    return map_grid, mapping_color


def augmentation(
    trans_id: int,
    mapping_color: np.ndarray = None,
    exclude_black=True,
    perm: np.ndarray | None = None,
):
    """Return (map_grid, mapping_color) for the chosen transform."""

    if trans_id == 9:
        map_grid, mapping_color = shuffle_transform(
            1000,
            mapping_color,
            exclude_black=exclude_black,
            perm=perm,
        )
        return map_grid, mapping_color
    if mapping_color is None:
        mapping_color = map_color(exclude_black=exclude_black)
    else:
        mapping_color = np.asarray(mapping_color)
        if mapping_color.shape != (10,):
            raise ValueError(f"mapping_color must have shape (10,), got {mapping_color.shape}")

    def map_grid(grid: np.ndarray):
        return extended_transform(mapping_color[grid], trans_id)

    return map_grid, mapping_color


def _inverse_color_mapping(mapping_color: np.ndarray) -> np.ndarray:
    mapping_color = np.asarray(mapping_color)
    if mapping_color.shape != (10,):
        raise ValueError(f"mapping_color must have shape (10,), got {mapping_color.shape}")
    inv = np.empty_like(mapping_color)
    inv[mapping_color] = np.arange(mapping_color.shape[0], dtype=mapping_color.dtype)
    return inv


def _inverse_dihedral_id(tid: int) -> int:
    # 0: I, 1: R90, 2: R180, 3: R270, 4: flip LR, 5: flip UD, 6: transpose, 7: anti-diagonal reflection
    if tid == 0:
        return 0
    if tid == 1:
        return 3
    if tid == 2:
        return 2
    if tid == 3:
        return 1
    if tid in (4, 5, 6, 7):
        return tid
    raise ValueError(f"Invalid transformation id: {tid}")


def _inverse_extended_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    if tid < 8:
        return dihedral_transform(arr, _inverse_dihedral_id(tid))
    if tid == 8:
        h, w = arr.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Cannot invert upscale for odd shape {arr.shape}")
        blocks = arr.reshape(h // 2, 2, w // 2, 2)
        base = blocks[:, 0, :, 0]
        if not np.all(blocks == base[:, None, :, None]):
            raise ValueError("Cannot invert upscale: blocks are not constant")
        return base
    raise ValueError(f"Invalid transformation id: {tid}")


def inverse_augmentation(
    trans_id: int,
    mapping_color: np.ndarray,
    perm: np.ndarray | None = None,
):
    """Return the inverse of :func:`augmentation`.

    Notes
    -----
        - For ids 0..8 this inverts the dihedral/upscale transform AND the color mapping.
        - For id 9 (shuffle), inversion requires providing the same ``perm`` used for
            the forward shuffle, since the default forward permutation is random.
    """

    if mapping_color is None:
        raise ValueError(
            "inverse_augmentation requires the explicit mapping_color used in the forward augmentation"
        )

    inv_color = _inverse_color_mapping(mapping_color)

    if trans_id == 9:
        if perm is None:
            raise ValueError("inverse_augmentation for trans_id=9 requires perm")
        perm = np.asarray(perm)
        if perm.shape != (1000,):
            raise ValueError(f"perm must have shape (1000,), got {perm.shape}")
        if perm.dtype != np.uint16:
            perm = perm.astype(np.uint16, copy=False)

        def map_grid(grid: np.ndarray):
            grid = np.asarray(grid)
            flat = grid.reshape(-1)
            if flat.size < perm.shape[0]:
                perm_used = perm[perm < flat.size]
            else:
                assert flat.size == perm.shape[0]
                perm_used = perm
            unshuffled = np.empty_like(flat)
            # forward did: out_flat = mapped_flat[perm_used]
            # so inverse is: mapped_flat[perm_used] = out_flat
            unshuffled[perm_used] = flat
            return inv_color[unshuffled].reshape(grid.shape)

        return map_grid

    def map_grid(grid: np.ndarray):
        grid = np.asarray(grid)
        unmapped = _inverse_extended_transform(grid, trans_id)
        return inv_color[unmapped]

    return map_grid


def synthetic_data(
    max_size=20,
    max_num_examples=5,
    single_inp_out_coloraug_prob=0,
    shuffle_prob=0.0,
    exclude_black=True,
):
    size = random.randint(2, max_size)
    example = []
    num_examples = random.randint(3, max_num_examples)
    for _ in range(num_examples):
        input = np.random.randint(0, 10, size=(size, size))
        output = input.copy()
        example.append({"input": input, "output": output})
    task = {"train": example[: num_examples - 1], "test": example[num_examples - 1 :]}
    if random.random() < single_inp_out_coloraug_prob:
        mapping_color = map_color(exclude_black=exclude_black)
    else:
        mapping_color = None
    if size < 15:
        transf_id = random.randint(0, 8)
        transf_id_out = random.randint(0, 8)
    else:
        # avoid long prompts by skipping upsampling
        transf_id = random.randint(0, 7)
        transf_id_out = random.randint(0, 7)
    if random.random() < shuffle_prob:
        transf_id = 9
    if random.random() < shuffle_prob:
        transf_id_out = 9
    map_grid_inp, _ = augmentation(transf_id, mapping_color=mapping_color, exclude_black=exclude_black)
    map_grid_out, _ = augmentation(transf_id_out, mapping_color=mapping_color, exclude_black=exclude_black)
    task, _ = apply_augmentation(
        task,
        map_grid_inp,
        map_grid_out,
        shuffle_examples=True,
        shuffle_train_examples=False,
    )
    return task


def read_tasks_from_single_file(challenge_file: str, solution_file=None):
    """
    Read tasks from a single file
    """
    with open(challenge_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if solution_file is not None:
        with open(solution_file, "r", encoding="utf-8") as handle:
            solutions = json.load(handle)
            for key, value in solutions.items():
                for idx, solution in enumerate(value):
                    data[key]["test"][idx]["output"] = solution

    all_tasks = []
    for task_name, subtask in data.items():
        subtask["name"] = task_name
        all_tasks.append(subtask)

    return all_tasks


def read_re_arc_tasks(dir_files: str):
    """
    Read tasks from a single file for re-arc
    """
    files = glob.glob(dir_files + "/*.json")
    assert len(files) > 0, f"No files found in {dir_files}"
    all_tasks = []
    max_elem = 0
    for file in files:
        with open(file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        random.shuffle(data)
        for i in range(0, len(data), 4):
            if len(data) - i < 4:
                break
            task = {}
            task["train"] = data[i : i + 3]
            task["test"] = data[i + 3 : i + 4]
            name = os.path.basename(file).replace(".json", "")
            task["name"] = name + "_re_arc" + str(i // 4)
            all_tasks.append(task)
            sum_elem = 0
            for example in task["train"] + task["test"]:
                sum_elem += np.array(example["input"]).size + np.array(example["output"]).size
            # print("sum elements:", sum_elem)
            max_elem = max(max_elem, sum_elem)
    print("max elements in re-arc tasks:", max_elem)
    return all_tasks


def clean_prompt(prompt):
    """Clean prompt"""
    prompt = prompt.replace(", ", "").replace("][", "|").replace("[[", "")
    prompt = prompt.replace("]]", "\n")
    return prompt


def get_position_idx(cleaned_prompt: str):
    """Return per-character position indices for an already-clean ARC prompt chunk.

    This function does *not* perform any JSON cleanup. It assumes
    ``cleaned_prompt`` is already in the compact format produced by
    :func:`clean_prompt`, e.g.:

        "I12|34\n" or "O012|340\n"

    It returns a list aligned to the input string where each digit character
    maps to its grid coordinate ``(y, x)``, then shifts *all* coordinates by
    ``+10`` to keep indices non-negative.

    Non-digit characters are mapped to fixed "special" indices *before* the shift:
    - 'I' -> (-1, -1)
    - 'O' -> (-2, -2)
    - '|' -> (-3, -3)
    - '\n' -> (-4, -4)
    - 'T' -> (-5, -5)
    After the +10 shift, these become:
    - 'I' -> (9, 9)
    - 'O' -> (8, 8)
    - '|' -> (7, 7)
    - '\n' -> (6, 6)
    - 'T' -> (5, 5)
    Note: the maximum returned index is ``(max_grid_dim - 1 + 10)``; ensure
    downstream embeddings (pos_emb_x/y or rotary freqs) are sized accordingly.
    Examples
    --------
    "I12|34\n" -> [(9,9),(10,10),(10,11),(7,7),(11,10),(11,11),(6,6)]
    """
    special_idxs = {
        "I": (-1, -1),
        "O": (-2, -2),
        "|": (-3, -3),
        "\n": (-4, -4),
        "T": (-5, -5),
    }
    cleaned = str(cleaned_prompt)

    positions: list[tuple[int, int]] = []
    y = 0
    x = 0
    for ch in cleaned:
        if ch.isdigit():
            positions.append((y, x))
            x += 1
        elif ch == "|":
            positions.append(special_idxs[ch])
            y += 1
            x = 0
        else:
            positions.append(special_idxs[ch])
    assert len(positions) == len(cleaned), f"Length mismatch: {len(positions)} != {len(cleaned)}"
    for idx, pos in enumerate(positions):
        positions[idx] = (pos[0] + 10, pos[1] + 10)  # shift to make all positive
    return positions


def create_prompt(task, think_budget=0, repeat_test=1, max_length=None, compute_position_idx=True):
    """Create prompt"""
    prompt = ""
    mask_all_out = []
    position_idx = [] if compute_position_idx else None
    for i, example in enumerate(task["train"]):
        inp = clean_prompt(f"I{json.dumps(example['input'])}")
        inp += "T" * think_budget
        if compute_position_idx:
            position_idx += get_position_idx(inp)
        prompt += inp
        mask_all_out += [0 for _ in range(len(inp))]
        out = clean_prompt(f"O{json.dumps(example['output'])}")
        if compute_position_idx:
            position_idx += get_position_idx(out)
        prompt += out
        mask_value = 1 if i > 0 else 0
        mask_all_out += [0] + [mask_value for _ in range(len(out) - 1)]
    mask = [0 for _ in range(len(prompt))]
    mask_input = []
    mask_noise = [0 for _ in range(len(prompt))]
    len_test = 1
    for rep in range(repeat_test):
        if max_length is not None and rep > 0 and len(prompt) > max_length - len_test * 2:
            last_rep = True
        else:
            last_rep = rep == repeat_test - 1
        for test_idx in range(len(task["test"])):
            inp = clean_prompt(f"I{json.dumps(task['test'][test_idx]['input'])}")
            inp += "T" * think_budget
            if compute_position_idx:
                position_idx += get_position_idx(inp)
            prompt += inp
            mask_all_out += [0 for _ in range(len(inp))]
            mask += [0 for _ in range(len(inp))]
            mask_noise += [0 for _ in range(len(inp))]

            out = clean_prompt(f"O{json.dumps(task['test'][test_idx]['output'])}")
            mask_input.append([len(prompt) + 1, len(out) - 1])
            if compute_position_idx:
                position_idx += get_position_idx(out)
            prompt += out
            val_mask = 1 if last_rep else 0
            mask_all_out += [0] + [1 for _ in range(len(out) - 1)]
            mask += [0] + [val_mask for _ in range(len(out) - 1)]
            mask_noise += [0] + [1 for _ in range(len(out) - 1)]
            if rep == 0:
                len_test += len(inp) + len(out)
        if last_rep:
            assert val_mask == 1
            if repeat_test > 1:
                assert rep > 0
            break
    assert len(mask) == len(prompt), f"Length mismatch: {len(mask)} != {len(prompt)}"
    if compute_position_idx:
        assert len(position_idx) == len(prompt), f"Length mismatch: {len(position_idx)} != {len(prompt)}"
    prompt = "[BOS]" + prompt
    if compute_position_idx:
        position_idx = [(0, 0)] + position_idx
    mask = [0] + mask
    mask_all_out = [0] + mask_all_out
    mask_noise = [0] + mask_noise
    mask_input = [[s + 1, l] for s, l in mask_input]
    assert len(mask) == len(mask_all_out), f"Mask length mismatch: {len(mask)} != {len(mask_all_out)}"
    assert len(mask) == len(mask_noise), f"Mask length mismatch: {len(mask)} != {len(mask_noise)}"
    return prompt, mask, mask_all_out, mask_input, mask_noise, position_idx


def clone_task_structure(task):
    """Create a lightweight copy of an ARC task before in-place augmentation."""

    cloned = {k: v for k, v in task.items() if k not in {"train", "test"}}
    cloned["train"] = [dict(example) for example in task["train"]]
    cloned["test"] = [dict(example) for example in task["test"]]
    return cloned


def grid_max_dim(grid) -> int:
    """Return the larger of a grid's height and width."""

    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {arr.shape}")
    return max(int(arr.shape[0]), int(arr.shape[1]))


def task_max_grid_dim(task) -> int:
    """Return the largest height/width appearing anywhere in an ARC task."""

    max_dim = 0
    for example in task["train"] + task["test"]:
        for key in ("input", "output"):
            max_dim = max(max_dim, grid_max_dim(example[key]))
    return max_dim


def get_char_tokenizer(dataset=["01234"], output_file="arc_agi_tokenizer.json"):
    for i in range(len(dataset)):
        dataset[i] = dataset[i].replace("[BOS]", "")
        dataset[i] = dataset[i].replace("[EOS]", "")
        dataset[i] = dataset[i].replace("[PAD]", "")
        if "[" in dataset[i]:
            print(dataset[i])
            breakpoint()
        assert "[UNK]" not in dataset[i]
    # 1. Create tokenizer with Unigram model
    tokenizer = Tokenizer(models.Unigram())

    # 2. Use pre-tokenizer that splits into single characters
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern="",  # Match every character
        behavior="isolated",
        invert=False,
    )

    # 3. Prepare trainer with your vocab of Unicode characters
    special_tokens = ["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
    initial_vocab = ["0", "1", "T"]
    trainer = trainers.UnigramTrainer(
        vocab_size=100,
        special_tokens=special_tokens,
        initial_alphabet=initial_vocab,  # Force Unigram to include your char set
        unk_token="[UNK]",
        max_piece_length=1,  # Each piece is a single character
    )

    tokenizer.train_from_iterator(dataset, trainer=trainer)
    tokenizer.post_processor = None
    tokenizer.decoder = decoders.Sequence([])
    # save the tokenizer to a file if needed
    tokenizer.save(output_file)
    return tokenizer


def get_arc_agi_dataset(
    include_val_train=False,
    include_concept=False,
    no_train=False,
    re_arc=False,
    max_task_size=None,
):
    all_tasks_concept = read_tasks_from_single_file(
        "arc_agi/arc-agi_concept-challenges.json",
        solution_file="arc_agi/arc-agi_concept-solutions.json",
    )
    if no_train:
        assert include_val_train
        all_tasks_train = []
    else:
        all_tasks_train = read_tasks_from_single_file(
            "arc_agi/arc-agi_training-challenges.json",
            solution_file="arc_agi/arc-agi_training-solutions.json",
        )
    all_tasks_val = read_tasks_from_single_file(
        "arc_agi/arc-agi_evaluation-challenges.json",
        solution_file="arc_agi/arc-agi_evaluation-solutions.json",
    )
    if include_val_train:
        all_tasks_val_train = read_tasks_from_single_file(
            "arc_agi/arc-agi_evaluation-challenges.json",
        )
        for task in all_tasks_val_train:
            task_train_val = {}
            task_train_val["train"] = copy.deepcopy(task["train"][:-1])
            task_train_val["test"] = copy.deepcopy(task["train"][-1:])
            all_tasks_train.append(task_train_val)
    if include_concept:
        all_tasks_train += all_tasks_concept

    if re_arc:
        all_tasks_train += read_re_arc_tasks("arc_agi/re_arc/")
    dataset = {
        "concept": all_tasks_concept,
        "train": all_tasks_train,
        "val": all_tasks_val,
    }
    if max_task_size is not None:
        for key in dataset:
            filtered_tasks = []
            for task in dataset[key]:
                task_size = size_elem_task(task)
                if task_size <= max_task_size:
                    filtered_tasks.append(task)
            print(
                f"Filtered {(len(dataset[key]) - len(filtered_tasks))/len(dataset[key]):.2%} tasks from {key} with avg size > {max_task_size}"
            )
            dataset[key] = filtered_tasks
    return dataset


def get_arc_agi_dataset_prompt(dataset, think_budget):
    dataset_prompt = {
        "train": {"text": [], "mask": []},
        "val": {"text": [], "mask": []},
    }
    max_len = 0
    for task in dataset["train"]:
        prompt, mask, _, _, _, _ = create_prompt(task, think_budget)
        dataset_prompt["train"]["text"].append(prompt)
        dataset_prompt["train"]["mask"].append(mask)
        max_len = max(max_len, len(prompt))
    for task in dataset["val"]:
        prompt, mask, _, _, _, _ = create_prompt(task, think_budget)
        dataset_prompt["val"]["text"].append(prompt)
        dataset_prompt["val"]["mask"].append(mask)
        max_len = max(max_len, len(prompt))
    print("Max length of prompts:", max_len)
    return dataset_prompt


def apply_augmentation(
    task,
    map_grid_inp,
    map_grid_out,
    *,
    shuffle_examples: bool,
    shuffle_train_examples: bool,
):
    """
    Apply augmentation to the task.

    Returns (task, meta) where meta reports shuffling/IO swap flags.
    """
    assert isinstance(task["train"], list) and isinstance(task["test"], list)

    did_shuffle = False
    if shuffle_examples:
        if random.random() < 0.5:
            task_all = task["train"] + task["test"]
            random.shuffle(task_all)
            task["train"] = task_all[: len(task["train"])]
            task["test"] = task_all[len(task["train"]) :]
        else:
            random.shuffle(task["train"])
            random.shuffle(task["test"])
        did_shuffle = True

    if shuffle_train_examples and not shuffle_examples and random.random() < 0.5:
        # we can shuffle train examples with no issues for evaluation (no need to set did_shuffle)
        random.shuffle(task["train"])
    if shuffle_examples and random.random() < 0.4:
        # swap input and output
        for example in task["train"]:
            example["input"], example["output"] = example["output"], example["input"]
        for example in task["test"]:
            example["input"], example["output"] = example["output"], example["input"]
        did_shuffle = True

    for example in task["train"]:
        example["input"] = map_grid_inp(np.array(example["input"])).tolist()
        example["output"] = map_grid_out(np.array(example["output"])).tolist()
    for example in task["test"]:
        example["input"] = map_grid_inp(np.array(example["input"])).tolist()
        example["output"] = map_grid_out(np.array(example["output"])).tolist()
    return task, {"did_shuffle": did_shuffle}


def size_elem_task(task, use_max=False):
    sum_elem_inp = 0
    sum_elem_out = 0
    num_elem = 0
    max_elem_inp = 0
    max_elem_out = 0
    for example in task["train"] + task["test"]:
        if use_max:
            max_elem_inp = max(max_elem_inp, np.array(example["input"]).size)
            max_elem_out = max(max_elem_out, np.array(example["output"]).size)
        else:
            sum_elem_inp += np.array(example["input"]).size
            sum_elem_out += np.array(example["output"]).size
            num_elem += 1
    if use_max:
        return max(max_elem_inp, max_elem_out)
    return (sum_elem_inp / num_elem + sum_elem_out / num_elem) / 2


class ArcAGI(Dataset):
    def __init__(
        self,
        data,
        padding,
        aug=True,
        text_out=True,
        truncate=False,
        gt_labels_only=False,
        causal=True,
        gt_all_out=True,
        think_budget=0,
        rand_think=True,
        aug_inp_out=False,
        single_inp_out_coloraug_prob=0,
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
        shuffle_examples: bool = True,
        shuffle_train_examples: bool = True,
        reduce_train: bool = True,
        return_aug_params: bool = False,
        return_grids: bool = False,
        include_position_idx: bool = True,
    ):
        """Initialize the ArcAGI dataset wrapper.

        Parameters
        ----------
        data : list
            List of ARC-AGI tasks. Each task is a dict with keys ``"train"`` and
            ``"test"``, whose values are lists of examples. Each example is a
            dict with keys ``"input"`` and ``"output"`` containing 2D integer
            grids (lists of lists with values in ``[0, 9]``).
        padding : int
            Maximum sequence length for tokenized prompts. All sequences are
            padded or truncated to this length.
        aug : bool, optional
            If ``True``, apply grid augmentations (dihedral / upscaling /
            shuffling, color permutations, train/test shuffling, IO swapping).
        text_out : bool, optional
            If ``True``, ``__getitem__`` returns encoded tokens (or raw prompt
            text if ``encode=False``). If ``False``, it returns the structured
            task dict without tokenization.
        truncate : bool, optional
            If ``True``, sequences longer than ``padding`` are truncated
            consistently across all token fields. If ``False``, an assertion
            error is raised when a sequence is too long.
        gt_labels_only : bool, optional
            If ``True``, loss is computed only on ground-truth output tokens,
            i.e. labels for non-output tokens are set to ``-100``.
            False (and causal) loss is computed on all tokens, so also it learns next
            token prediction for thinking tokens and input pattern tokens.
        causal : bool, optional
            If ``True``, use standard causal LM training where inputs are
            visible and labels are shifted. If ``False``, masked positions in
            ``input_ids`` are replaced with ``[UNK]`` and labels outside the
            mask are set to ``-100``.
        gt_all_out : bool, optional
            When ``gt_labels_only=True``, controls whether all outputs
            (including from intermediate train examples) are supervised.
        think_budget : int, optional
            Maximum number of ``"T"`` thinking tokens appended after each input
            grid.
        rand_think : bool, optional
            If ``True``, the actual number of thinking tokens is sampled
            uniformly from ``[0, think_budget]`` for each example with
            probability 0.5.
        aug_inp_out : bool, optional
            If ``True``, input and output grids can receive different
            augmentations; otherwise the same transform is applied to both.
        single_inp_out_coloraug_prob : float, optional
            Probability of using a single shared color permutation for both
            inputs and outputs (color augmentation).
        max_aug_idx : int, optional
            Maximum augmentation id used when sampling transformations in
            :func:`augmentation`. Values correspond to dihedral transforms,
            upscaling, or shuffling.
        num_synthetic_data : int, optional
            Number of additional synthetic tasks to append on top of ``data``.
            These are generated on-the-fly by :func:`synthetic_data`.
        noise_prob : float, optional
            Probability of replacing a supervised token by random noise (sampled
            from the set of tokens present in the sequence). Applied only on
            ground-truth positions when ``aug=True``.
        use_unk_token_for_gt : bool, optional
            If ``True``, ``input_ids`` at ground-truth positions are set to the
            tokenizer ``[UNK]`` id (labels remain the correct targets).
        repeat_test : int, optional
            Number of times test examples are repeated in the prompt, with only
            the last repetition contributing to the validation mask.
        encode : bool, optional
            If ``True``, prompts are tokenized using ``arc_agi_tokenizer.json``.
            If ``False``, ``__getitem__`` returns the raw string prompt and
            associated masks.
        compute_mask_input : bool, optional
            If ``True``, return an additional ``"mask_input"`` field encoding
            input spans for test outputs.
        exclude_black : bool, optional
            If ``True``, color permutations avoid changing the ``0`` (black)
            value when constructing mappings.
        shuffle_prob : float, optional
            Probability of applying shuffle-based augmentation (id 9) instead of
            dihedral/scale transforms.
        aligned_labels : bool, optional
            If ``True``, labels are shifted by one position to align with the
            next-token prediction setup, and the last position is masked out.
        """
        self.encode = encode
        self.compute_mask_input = compute_mask_input
        self.single_inp_out_coloraug_prob = single_inp_out_coloraug_prob
        self.num_synthetic_data = num_synthetic_data
        self.aug_inp_out = aug_inp_out
        self.max_aug_idx = max_aug_idx
        self.tokenizer_file = "arc_agi_tokenizer.json"
        self.tokenizer = None
        self.data = data
        self.padding = padding
        self.aug = aug
        self.text_out = text_out
        self.truncate = truncate
        self.gt_labels_only = gt_labels_only
        self.causal = causal
        # use all outputs as gt after the first example
        self.gt_all_out = gt_all_out
        self.think_budget = think_budget
        self.rand_think = rand_think
        if self.num_synthetic_data > 0:
            assert self.aug
        self.noise_prob = noise_prob
        self.use_unk_token_for_gt = use_unk_token_for_gt
        self.repeat_test = repeat_test
        self.exclude_black = exclude_black
        self.shuffle_prob = shuffle_prob
        self.aligned_labels = aligned_labels
        self.shuffle_examples = shuffle_examples
        self.shuffle_train_examples = shuffle_train_examples
        self.reduce_train = reduce_train
        self.return_aug_params = return_aug_params
        self.return_grids = return_grids
        self.include_position_idx = include_position_idx

    def get_tokenizer(self):
        if self.tokenizer is None:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=self.tokenizer_file,
                bos_token="[BOS]",
                eos_token="[EOS]",
                pad_token="[PAD]",
                unk_token="[UNK]",
            )
            assert tokenizer.pad_token_id is not None
            assert tokenizer.unk_token_id is not None
            assert tokenizer.bos_token_id is not None
            assert tokenizer.eos_token_id is not None
        else:
            tokenizer = self.tokenizer
        return tokenizer

    def __len__(self):
        return len(self.data) + self.num_synthetic_data

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        self.tokenizer = self.get_tokenizer()  # lazy init of the tokenizer
        if index >= len(self.data):
            task = synthetic_data(
                single_inp_out_coloraug_prob=self.single_inp_out_coloraug_prob,
                shuffle_prob=self.shuffle_prob,
                exclude_black=self.exclude_black,
            )
            assert self.aug
        else:
            task = self.data[index]
        if self.aug:
            if random.random() < self.single_inp_out_coloraug_prob:
                mapping_color_shared = map_color(exclude_black=self.exclude_black)
            else:
                mapping_color_shared = None
            max_size = task_max_grid_dim(task)
            if max_size >= 12:
                max_aug_idx_used = min(self.max_aug_idx, 7)
            else:
                max_aug_idx_used = self.max_aug_idx
            transf_id = random.randint(0, max_aug_idx_used)
            perm_inp = None
            if random.random() < self.shuffle_prob:
                transf_id = 9
                perm_inp = np.random.permutation(np.arange(0, 1000, dtype=np.uint16))
            map_grid_inp, mapping_color_inp = augmentation(
                transf_id,
                mapping_color=mapping_color_shared,
                exclude_black=self.exclude_black,
                perm=perm_inp,
            )
            transf_id_out = transf_id
            if self.aug_inp_out and random.random() < 0.5:
                transf_id_out = random.randint(0, max_aug_idx_used)
                perm_out = None
                if random.random() < self.shuffle_prob:
                    transf_id_out = 9
                    perm_out = np.random.permutation(np.arange(0, 1000, dtype=np.uint16))
                map_grid_out, mapping_color_out = augmentation(
                    transf_id_out,
                    mapping_color=mapping_color_shared,
                    exclude_black=self.exclude_black,
                    perm=perm_out,
                )
            else:
                map_grid_out = map_grid_inp
                perm_out = perm_inp
                mapping_color_out = mapping_color_inp
            num_train = len(task["train"])

            task, aug_meta = apply_augmentation(
                clone_task_structure(task),
                map_grid_inp,
                map_grid_out,
                shuffle_examples=self.shuffle_examples,
                shuffle_train_examples=self.shuffle_train_examples,
            )
            if self.reduce_train and random.random() < 0.3:  # and not self.gt_labels_only
                task["train"] = task["train"][: random.randint(1, num_train)]
            if self.rand_think and random.random() < 0.5:
                think_budget = random.randint(0, self.think_budget)
            else:
                think_budget = self.think_budget
        else:
            think_budget = self.think_budget
        if not self.text_out:
            return task

        if self.return_grids:
            # Build padded grid tensors for downstream models/analysis.
            # input_grids: train inputs + train outputs + test inputs
            # output_grids: test outputs only
            # We pad the *number of grids* to a fixed maximum to make batching work.
            max_grids = 32
            pad_value = -1
            max_size_grid = 40
            input_grids_list: list[torch.Tensor] = []
            for ex in task["train"]:
                input_grids_list.append(pad_grid(ex["input"], pad_value=pad_value, max_size=max_size_grid))
                input_grids_list.append(pad_grid(ex["output"], pad_value=pad_value, max_size=max_size_grid))
            for ex in task["test"]:
                input_grids_list.append(pad_grid(ex["input"], pad_value=pad_value, max_size=max_size_grid))
                input_grids_list.append(
                    torch.full((max_size_grid, max_size_grid), pad_value, dtype=torch.int32)
                )

            output_grids_list: list[torch.Tensor] = [
                pad_grid(ex["output"], pad_value=pad_value, max_size=max_size_grid) for ex in task["test"]
            ]
            assert len(input_grids_list) <= max_grids
            assert len(output_grids_list) <= max_grids // 2

            while len(input_grids_list) < max_grids:
                input_grids_list.append(
                    torch.full((max_size_grid, max_size_grid), pad_value, dtype=torch.int32)
                )
            while len(output_grids_list) < max_grids // 2:
                output_grids_list.append(
                    torch.full((max_size_grid, max_size_grid), pad_value, dtype=torch.int32)
                )

            input_grids_tensor = torch.stack(input_grids_list, dim=0).to(torch.int32)
            output_grids_tensor = torch.stack(output_grids_list, dim=0).to(torch.int32)

        prompt, mask_gt, mask_all_out, mask_input, mask_noise, position_idx = create_prompt(
            task,
            think_budget,
            self.repeat_test,
            max_length=self.padding - 10,
            compute_position_idx=self.include_position_idx,
        )
        if not self.encode:
            return prompt, mask_gt, mask_all_out, mask_input, mask_noise
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.padding,
            truncation=False,
            return_token_type_ids=False,
        )
        tokens["idx"] = index
        if self.return_grids:
            tokens["input_grids"] = input_grids_tensor
            tokens["output_grids"] = output_grids_tensor

        if self.aug and self.return_aug_params:
            # Store augmentation parameters under a dedicated dict so downstream
            # length checks don't treat them as token-aligned sequences.
            tokens["aug_params"] = {
                "transf_id_inp": int(transf_id),
                "transf_id_out": int(transf_id_out) if self.aug_inp_out else int(transf_id),
                "mapping_color_inp": (
                    np.asarray(mapping_color_inp, dtype=np.uint8).tolist()
                    if mapping_color_inp is not None
                    else None
                ),
                "mapping_color_out": (
                    np.asarray(mapping_color_out, dtype=np.uint8).tolist()
                    if mapping_color_out is not None
                    else None
                ),
                "perm_inp": (
                    np.asarray(perm_inp, dtype=np.uint16).tolist() if perm_inp is not None else None
                ),
                "perm_out": (
                    np.asarray(perm_out, dtype=np.uint16).tolist() if perm_out is not None else None
                ),
                "did_shuffle": bool(aug_meta.get("did_shuffle", False)),
            }

        tokens["size_input"] = np.mean([grid_max_dim(example["input"]) for example in task["train"]])
        tokens["size_output"] = np.mean([grid_max_dim(example["output"]) for example in task["train"]])
        tokens["task_size"] = size_elem_task(task)
        if self.tokenizer.unk_token_id in tokens["input_ids"]:
            raise ValueError("Input contains unknown tokens (UNK).")
        assert (
            self.padding - len(mask_gt) >= 0
        ), f"Padding {self.padding} too small for sequence of length {len(mask_gt)}"
        # mask is 1 for positions to predict (for accuracy computation)
        tokens["mask"] = mask_gt + [0] * (self.padding - len(mask_gt))
        mask_noise = mask_noise + [0] * (self.padding - len(mask_noise))
        if self.include_position_idx:
            position_idx = position_idx + [(0, 0)] * (self.padding - len(position_idx))
            tokens["position_idx"] = position_idx
        if self.compute_mask_input:
            tokens["mask_input"] = mask_input
            assert len(tokens["mask_input"]) <= 32
            # padding to have fixed size
            for _ in range(32 - len(mask_input)):
                tokens["mask_input"].append([-1, -1])
        tokens["labels"] = tokens["input_ids"].copy()
        if not self.causal:
            for i in range(len(tokens["input_ids"])):
                if tokens["mask"][i] == 1:
                    tokens["input_ids"][i] = self.tokenizer.unk_token_id
        mask2use = None
        if self.gt_labels_only:
            if self.gt_all_out:
                mask2use = mask_all_out + [0] * (self.padding - len(mask_all_out))
            else:
                mask2use = tokens["mask"]

        if self.use_unk_token_for_gt:
            assert self.tokenizer.unk_token_id is not None
            assert self.noise_prob == 0.0
            ids = np.array(tokens["input_ids"], dtype=np.int32)
            mask = np.array(mask2use, dtype=np.int32)
            ids[mask == 1] = self.tokenizer.unk_token_id
            tokens["input_ids"] = ids.tolist()
        # Add token noise on masked positions only
        if self.noise_prob > 0 and self.aug:
            ids = np.array(tokens["input_ids"], dtype=np.int32)
            unique_vals = np.unique(ids)
            assert unique_vals.size > 0
            noise = np.random.choice(unique_vals, size=ids.shape[0])
            mask_noise_sampled = np.array(mask2use, dtype=np.int32) * (
                np.random.random(ids.shape[0]) < self.noise_prob
            )
            ids = ids * (1 - mask_noise_sampled) + noise * mask_noise_sampled
            tokens["input_ids"] = ids.tolist()

        for i in range(len(tokens["labels"])):
            if tokens["labels"][i] == self.tokenizer.pad_token_id:
                tokens["labels"][i] = -100
            if not self.causal and tokens["mask"][i] == 0:
                tokens["labels"][i] = -100
            if self.gt_labels_only and mask2use[i] == 0:
                tokens["labels"][i] = -100
        if "attention_mask" in tokens and self.causal:
            tokens["attention_mask"] = None
        if self.aligned_labels:
            tokens["labels"] = tokens["labels"][1:] + [-100]
            tokens["mask"] = tokens["mask"][1:] + [0]
        # Truncate consistently if needed
        if len(tokens["labels"]) > self.padding:
            assert self.truncate
            for k in list(tokens.keys()):
                if tokens[k] is None or k in [
                    "idx",
                    "size_input",
                    "size_output",
                    "mask_input",
                    "task_size",
                    "aug_params",
                    "input_grids",
                    "output_grids",
                ]:
                    continue
                tokens[k] = tokens[k][: self.padding]
        else:
            assert len(tokens["labels"]) == self.padding
            assert len(tokens["input_ids"]) == self.padding
            assert len(tokens["mask"]) == self.padding
        for k in tokens:
            if tokens[k] is None or k in [
                "size_input",
                "idx",
                "size_output",
                "mask_input",
                "task_size",
                "aug_params",
                "input_grids",
                "output_grids",
            ]:
                continue
            assert len(tokens[k]) == len(
                tokens["input_ids"]
            ), f"Token {k} has length {len(tokens[k])}, expected {len(tokens['input_ids'])}"
        return tokens


def check_contamination():
    from tqdm import tqdm

    dataset = get_arc_agi_dataset(include_val_train=True, no_train=True)

    db_train = ArcAGI(
        dataset["train"],
        9400,
        aug=True,
        causal=True,
        gt_labels_only=True,
        text_out=False,
        max_aug_idx=7,
    )
    db_val = ArcAGI(
        dataset["val"],
        9400,
        aug=False,
        causal=True,
        gt_labels_only=True,
        text_out=False,
        max_aug_idx=7,
    )
    val_examples = set()
    for task_val in db_val:
        for example_val in task_val["test"]:
            val_examples.add(f"{json.dumps(example_val['input'])}{json.dumps(example_val['output'])}")
    contaminated = set()
    num_contaminated = 0
    for _ in tqdm(range(10000)):
        for task in db_train:
            for example in task["train"]:
                example_str = f"{json.dumps(example['input'])}{json.dumps(example['output'])}"
                if example_str in val_examples:
                    contaminated.add(example_str)
                    if len(contaminated) > num_contaminated:
                        print("contaminated!", len(contaminated))
                    num_contaminated = len(contaminated)
                # for task_val in db_val:
                #     for example_val in task_val["test"]:
                #         if (example["input"] == example_val["input"]) and example["output"] == example_val["output"]:
                #             contaminated.add(json.dumps(example['input']))
                #             if len(contaminated)>num_contaminated:
                #                 print("contaminated!", len(contaminated))
                #             num_contaminated=len(contaminated)


if __name__ == "__main__":
    from matplotlib.colors import ListedColormap

    # Define a colormap with 10 distinct colors for values 0-9
    custom_cmap = ListedColormap(
        [
            "#000000",  # 0 - black
            "#e6194b",  # 1 - red
            "#3cb44b",  # 2 - green
            "#ffdd00",  # 3 - yellow
            "#4363d8",  # 4 - blue
            "#f58231",  # 5 - orange
            "#911eb4",  # 6 - purple
            "#46f0f0",  # 7 - cyan
            "#f032e6",  # 8 - magenta
            "#9a6422",  # 9 - brown
        ]
    )
    dataset = get_arc_agi_dataset(max_task_size=300)
    for subset in dataset:
        print(subset, len(dataset[subset]))
    dataset_prompt = get_arc_agi_dataset_prompt(dataset, 10)
    # generate the tokenizer
    if False:
        tokenizer = get_char_tokenizer(
            dataset_prompt["train"]["text"] + dataset_prompt["val"]["text"],
            "arc_agi_tokenizer.json",
        )
    # re_arc = read_re_arc_tasks("arc_agi/re_arc/")
    db = ArcAGI(
        dataset["train"],
        9400,
        aug=True,
        causal=True,
        gt_labels_only=True,
        think_budget=0,
        max_aug_idx=7,
        noise_prob=0.0,
    )

    # max_len = 0
    # for i, e in enumerate(db):
    #     #max_len = max(max_len, len(e["input_ids"]))
    #     print(f"Length of example {i}: {len(e['input_ids'])}")
    # print("Max length in dataset:", max_len)
    tokenizer = db.get_tokenizer()
    example = db[0]
    example = {k: np.array(example[k]) for k in example if example[k] is not None}
    for k in example:
        if k in [
            "mask",
            "attention_mask",
            "idx",
            "size_input",
            "size_output",
            "mask_input",
            "task_size",
            "input_grids",
            "output_grids",
        ]:
            print(f"{k}: {example[k]}")
        else:
            example[k][example[k] == -100] = tokenizer.unk_token_id
            print(f"{k}: {tokenizer.decode(example[k][example['mask'] == 1],skip_special_tokens=False)}")
    print(
        "orig",
        tokenizer.decode(example["input_ids"], skip_special_tokens=False).replace("[PAD]", ""),
    )
    print(
        "labels",
        tokenizer.decode(example["labels"], skip_special_tokens=False).replace("[UNK]", "K"),
    )
    # print(tokenizer.decode(db[0]["input_ids"], skip_special_tokens=False))  # keep [BOS]/[PAD] if present
    # print(repr(tokenizer.decode(db[0]["input_ids"])))  # shows escaped \n sequences
    # print("labels")
    # print(db[0]["labels"])  # keep [BOS]/[PAD] if present
    print("---")

    string = "123[EOS]IO\n[EOS]"
    encoded = tokenizer(string)
    print("Encoded IDs:", encoded["input_ids"])
    print("decoded")
    print(tokenizer.decode(encoded["input_ids"], skip_special_tokens=False))
    assert tokenizer.decode(encoded["input_ids"], skip_special_tokens=False) == string
    # print(encoded["tokens"])
    print("Vocab size:", tokenizer.vocab_size)
    print("---")
    db = ArcAGI(dataset["train"], 9400, aug=False, text_out=False)
    task = db[0]  # syntetic_data()
    print("size", size_elem_task(task))
    fig, axs = plt.subplots(len(task["train"]) + len(task["test"]), 4, figsize=(8, 4))
    for i, example in enumerate(task["train"]):
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])
        map_grid, _ = augmentation(1, map_color(True))
        axs[i, 0].imshow(input_grid, cmap=custom_cmap, interpolation="nearest", vmin=0, vmax=9)
        axs[i, 0].set_title("Input")
        axs[i, 0].axis("off")
        axs[i, 1].imshow(output_grid, cmap=custom_cmap, interpolation="nearest", vmin=0, vmax=9)
        axs[i, 1].set_title("Output")
        axs[i, 1].axis("off")
        axs[i, 2].imshow(
            map_grid(input_grid),
            cmap=custom_cmap,
            interpolation="nearest",
            vmin=0,
            vmax=9,
        )
        axs[i, 2].set_title("Input Augmentation")
        axs[i, 2].axis("off")
        axs[i, 3].imshow(
            map_grid(output_grid),
            cmap=custom_cmap,
            interpolation="nearest",
            vmin=0,
            vmax=9,
        )
        axs[i, 3].set_title("Output Augmentation")
        axs[i, 3].axis("off")

    for i, example in enumerate(task["test"]):
        input_grid = np.array(example["input"])
        output_grid = np.array(example["output"])
        axs[len(task["train"]) + i, 0].imshow(
            input_grid, cmap=custom_cmap, interpolation="nearest", vmin=0, vmax=9
        )
        axs[len(task["train"]) + i, 0].set_title("Input")
        axs[len(task["train"]) + i, 0].axis("off")
        axs[len(task["train"]) + i, 1].imshow(
            output_grid, cmap=custom_cmap, interpolation="nearest", vmin=0, vmax=9
        )
        axs[len(task["train"]) + i, 1].set_title("Output")
        axs[len(task["train"]) + i, 1].axis("off")
    plt.tight_layout()
    plt.show()
