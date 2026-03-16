from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import torch
import random
import re
from itertools import chain


def crop_data_collator(features, return_tensors="pt"):
    """
    Collate a batch and trim tensors to the last position with a valid label (labels != -100).

    Reduces compute on right padding while preserving shape alignment across
    input_ids, labels and other fields.

    Args:
        features: List of feature dicts from the dataset.
        return_tensors: Tensor type to return (default "pt").

    Returns:
        Dict[str, torch.Tensor]: Batched and cropped tensors.
    """
    out_features = default_data_collator(features, return_tensors=return_tensors)
    # max_valid = (out_features["labels"]!=-100).sum(1)
    # max_valid = max_valid.max().item()
    # Find the maximum index where any label is not -100
    valid_indices = (out_features["labels"] != -100).nonzero()
    if valid_indices.numel() > 0:
        max_index = valid_indices[:, 1].max().item()
    else:
        max_index = 0
    max_valid = max_index + 1
    labels_shape = out_features["labels"].shape
    assert (out_features["labels"][:, max_valid:] == -100).all()
    for k in out_features:
        if k in ["size_input", "size_output", "mask_input", "task_size", "idx"]:
            continue
        if k == "position_idx":
            # position_idx is shaped (B, T, 2) and must be cropped alongside tokens
            assert out_features[k].shape[:2] == labels_shape
            out_features[k] = out_features[k][:, :max_valid]
        elif k == "input_grids" or k == "output_grids":
            assert out_features[k].shape[0] == labels_shape[0]
        elif out_features[k] is not None:
            assert out_features[k].shape == labels_shape, f"{k} {out_features[k].shape} != {labels_shape}"
            out_features[k] = out_features[k][:, :max_valid]
    return out_features


def shift_batch(orig_batch, shift, eos):
    """
    Shift each tensor in the batch to the right by `shift` tokens, left-padding with EOS.

    Leaves attention_mask unchanged (must be all ones); used for random shift
    data augmentation.

    Args:
        orig_batch: Dict of batched tensors.
        shift: Number of tokens to shift to the right.
        eos: EOS token id used to pad on the left.

    Returns:
        Dict[str, torch.Tensor]: Shifted batch.
    """
    for key in orig_batch:
        if key == "attention_mask":
            assert orig_batch["attention_mask"].min() == 1
            continue
        if key == "position_idx":
            # Keep position indices aligned with shifted tokens.
            # Use the first position (typically BOS/sentinel) for left padding.
            pad = orig_batch[key][:, :1, :].expand(-1, shift, -1).contiguous()
            orig_batch[key] = torch.cat([pad, orig_batch[key][:, :-shift, :]], dim=1)
            print("" f"NOT TESTED! Shifted position_idx with pad shape {pad.shape}")
            continue
        eos_pad = torch.full(
            (orig_batch[key].shape[0], shift),
            eos,
            dtype=torch.long,
            device=orig_batch[key].device,
        )
        orig_batch[key] = torch.cat([eos_pad, orig_batch[key][:, :-shift]], dim=1)
    return orig_batch


def build_conversations_v1(rows, eos):
    """Build conversation samples by walking a running stack; emit after each assistant turn.

    Args:
        rows: Iterable of message dicts with message_id, parent_id, role, text.
        eos: End-of-segment delimiter appended to each utterance.

    Returns:
        List[Dict[str, str]]: Each with keys 'text', 'text_gt' (roles as 'U'/'A'), and 'debug_text'.
    """
    stack = []
    prev_message_id = None
    new_texts = []
    num_messages = 0
    simple_id_message = {}
    for row in rows:
        if row["message_id"] not in simple_id_message:
            simple_id_message[row["message_id"]] = num_messages
            num_messages += 1
        # if row['lang'] != "en":
        #     continue
        message_id = row["message_id"]
        assert message_id is not None, f"message_id is None in row {row}"

        parent_id = row["parent_id"]
        role = row["role"]
        if parent_id is None:
            stack = [row]
        # If parent_id is previous message, append
        elif prev_message_id is not None and parent_id == prev_message_id:
            stack.append(row)
        # Otherwise, pop until parent is found (or reset)
        else:
            while stack and (stack[-1]["message_id"] != parent_id):
                stack.pop()
            stack.append(row)
        prev_message_id = message_id
        # For each assistant message, output the conversation so far
        if role == "assistant":
            formatted = []
            formatted_gt = []
            debug_text = []
            for t in stack:
                if t["role"] == "prompter":
                    formatted.append(f"User: {t['text']}{eos}")
                    formatted_gt.append("U")
                    debug_text.append(f"User: {simple_id_message[t['message_id']]}")
                elif t["role"] == "assistant":
                    formatted.append(f"Assistant: {t['text']}{eos}")
                    formatted_gt.append("A")
                    debug_text.append(f"Assistant: {simple_id_message[t['message_id']]}")
            if len(stack) > 0:
                new_texts.append(
                    {
                        "text": "".join(formatted),
                        "text_gt": "".join(formatted_gt),
                        "debug_text": " ".join(debug_text),
                    }
                )
    for i in range(15):
        print(new_texts[i]["debug_text"])
    return new_texts


def build_conversations_v2(rows, eos):
    """Build conversation samples from leaf messages by tracing parents to the root.

    Only leaf messages are kept; returns formatted text, segment roles, and debug ids.

    Args:
        rows: Iterable of message dicts with message_id, parent_id, role, text.
        eos: End-of-segment delimiter appended to each utterance.

    Returns:
        List[Dict[str, str]]: Each with keys 'text', 'text_gt', 'debug_text'.
    """
    stack = []
    new_texts = []
    row_dict = {row["message_id"]: row for row in rows}
    parent_dict = {row["parent_id"] for row in rows}
    num_messages = 0
    simple_id_message = {}
    for row in rows:
        # if row["role"] == "prompter" and "italy" in row["text"].lower():
        #     print(row["text"])
        if row["message_id"] not in simple_id_message:
            simple_id_message[row["message_id"]] = num_messages
            num_messages += 1
        # if row['lang'] != "en":
        #     continue
        message_id = row["message_id"]
        assert message_id is not None
        parent_id = row["parent_id"]

        # we keep only the leaf messages
        if parent_id is None or message_id in parent_dict:
            continue
        stack = [row]
        while parent_id is not None:
            parent_row = row_dict[parent_id]
            stack = [parent_row] + stack
            parent_id = parent_row["parent_id"]

        formatted = []
        formatted_gt = []
        debug_text = []
        for t in stack:
            if t["role"] == "prompter":
                formatted.append(f"User: {t['text']}{eos}")
                formatted_gt.append("U")
                debug_text.append(f"User: {simple_id_message[t['message_id']]}")
            elif t["role"] == "assistant":
                formatted.append(f"Assistant: {t['text']}{eos}")
                formatted_gt.append("A")
                debug_text.append(f"Assistant: {simple_id_message[t['message_id']]}")
            else:
                raise ValueError(f"Unknown role {t['role']} in row {t}")
        new_texts.append(
            {
                "text": "".join(formatted),
                "text_gt": "".join(formatted_gt),
                "debug_text": " ".join(debug_text),
            }
        )

    # for i in range(15):
    #     print(new_texts[i]["debug_text"])
    return new_texts


build_conversations = build_conversations_v2


def group_texts_train(examples, block_size, stride, shift_label=False):
    """
    Main data processing function that will concatenate all texts
    from our dataset and generate chunks of block_size.
    you can use a stride lower then block_size if the dataset is small
    """
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = total_length - block_size + 1
    add_token = 1 if shift_label else 0
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size + add_token] for i in range(0, total_length, stride)]
        for k, t in concatenated_examples.items()
    }
    if "labels" not in result:
        result["labels"] = result["input_ids"].copy()
    if shift_label:
        for key in result:
            if key != "labels":
                result[key] = [t[:block_size] for t in result[key]]
        for i in range(len(result["labels"])):
            if len(result["labels"][i]) == block_size + 1:
                result["labels"][i] = result["labels"][i][1:]
            else:
                assert len(result["labels"][i]) == block_size
                result["labels"][i] = result["labels"][i][1:] + [-100]
    for key in result:
        for i in range(len(result[key])):
            assert len(result[key][i]) == block_size
    for i in range(len(result["attention_mask"])):
        if len(result["attention_mask"][i]) == 0:
            print("Warning: attention_mask is empty for example", i)
            result["attention_mask"][i] = None
        if result["attention_mask"][i] is not None and min(result["attention_mask"][i]) == 1:
            assert result["attention_mask"][i][0] == 1, "Expected the first element of attention_mask to be 1"
            result["attention_mask"][i] = None  # speed up computation
    return result


def group_texts_val(examples, block_size, stride, shift_label=False):
    """
    group texts in blocks with a given stride with labels -100 (ignored) when stride<block_size is used
    this is used only for validation
    """
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    result = {k: [] for k in examples.keys()}
    assert "labels" not in result
    result["labels"] = []
    add_token = 1 if shift_label else 0
    for i in range(0, total_length - block_size + 1, stride):
        for k in examples.keys():
            result[k].append(concatenated_examples[k][i : i + block_size])
        # append labels slice; when shift_label is True this may be length block_size+1
        result["labels"].append(concatenated_examples["input_ids"][i : i + block_size + add_token].copy())

        # If shift_label is requested, shift the labels now so labels correspond to next-token targets
        if shift_label:
            if len(result["labels"][-1]) == block_size + 1:
                # normal case: we have an extra token to form targets for all inputs
                result["labels"][-1] = result["labels"][-1][1:]
            else:
                # end-of-sequence case: pad the last target with -100
                assert len(result["labels"][-1]) == block_size
                result["labels"][-1] = result["labels"][-1][1:] + [-100]

        # For overlapping validation windows (stride < block_size) we mask the first
        # block_size - stride labels so they are not counted multiple times.
        if i > 0:
            for j in range(block_size - stride):
                result["labels"][-1][j] = -100
        if result["attention_mask"][-1] is not None and min(result["attention_mask"][-1]) == 1:
            assert (
                result["attention_mask"][-1][0] == 1
            ), "Expected the first element of attention_mask to be 1"
            result["attention_mask"][-1] = None  # speed up computation
    return result


def split_keep_strings(text, strings):
    """Split text on any delimiter in `strings` and keep the delimiters in the result.

    Args:
        text: Input string to split.
        strings: Delimiter strings to split on.

    Returns:
        List[str]: Parts including the delimiters.
    """
    # Create a regex pattern to split on each string and keep them
    pattern = "(" + "|".join(map(re.escape, strings)) + ")"
    parts = re.split(pattern, text)
    return [p for p in parts if p]  # Remove empty strings


def add_noise(
    text,
    noise_prob=0.01,
    capitalize_prob=0.01,
    add_char_prob=0.1,
    del_char_prob=0.01,
    special_tokens=[],
    condition_text="",
):
    """
    Add noise to the input text with a given probability.
    This is used to train the model to be robust to noise.
    """
    char = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?_-()[]{}<>\"'`~@#$%^&*+=|\\/ "
    new_text = split_keep_strings(text, special_tokens)

    assert text == "".join(new_text), f"Text mismatch: {text} != {''.join(new_text)}"
    for idx, text_part in enumerate(new_text):
        if text_part in special_tokens:
            continue
        if condition_text == "" or text_part.startswith(condition_text):
            text_part = list(text_part[len(condition_text) :])
            for i in range(len(text_part)):
                if random.random() < del_char_prob and len(text_part) > 1:
                    text_part[i] = ""
                else:
                    if random.random() < noise_prob:
                        if random.random() < add_char_prob:
                            text_part.insert(i, random.choice(char))
                        else:
                            text_part[i] = random.choice(char)
                    if random.random() < capitalize_prob:
                        text_part[i] = (
                            text_part[i].upper() if text_part[i].islower() else text_part[i].lower()
                        )
            new_text[idx] = condition_text + "".join(text_part)
        else:
            # If the text part does not start with the condition_text, we do not apply noise
            new_text[idx] = text_part
    return "".join(new_text)


def add_noise_token(
    tokens,
    noise_prob,
    capitalize_prob,
    add_char_prob,
    del_char_prob,
    tokenizer,
    condition_text,
):
    """
    Add noise to the input tokens with a given probability.
    This is used to train the model to be robust to noise.
    """
    noisy_token_list = []
    special_tokens = [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]
    special_tokens = [t for t in special_tokens if t is not None]
    special_tokens = list(set(special_tokens))  # remove duplicates
    for token_list in tokens:
        text = tokenizer.decode(token_list, skip_special_tokens=False)
        if condition_text == "":
            offset = random.randint(0, len(text) // 2)
            size = random.randint(len(text) // 3, len(text) // 2)
            text = text[offset : offset + size]
        noisy_text = add_noise(
            text,
            noise_prob,
            capitalize_prob,
            add_char_prob,
            del_char_prob,
            special_tokens=special_tokens,
            condition_text=condition_text,
        )

        if condition_text == "":
            noisy_text = (
                "User: fix the mispelling: "
                + noisy_text
                + tokenizer.eos_token
                + "Assistant: "
                + text
                + tokenizer.eos_token
            )
        # print(f"Original text: {text}")
        # print("---")
        # print(f"Noisy text: {noisy_text}")
        # print("---")
        # breakpoint()
        noisy_token_list.append(tokenizer.encode(noisy_text))
        if len(noisy_token_list[-1]) < len(token_list):
            noisy_token_list[-1] += [tokenizer.eos_token_id] * (len(token_list) - len(noisy_token_list[-1]))
        elif len(noisy_token_list[-1]) > len(token_list):
            noisy_token_list[-1] = noisy_token_list[-1][: len(token_list)]
        assert len(noisy_token_list[-1]) == len(token_list)
    tokens = torch.tensor(noisy_token_list, dtype=torch.long, device=tokens.device)
    return tokens


def tokenize_function(examples, tokenizer, text_column_name, prepend_bos, eos, padding=0, noise_prob=0.0):
    """Tokenize examples into input_ids/attention_mask (and labels if 'text_gt' present).

    For conversation-style inputs with 'text_gt', masks labels for user segments and can apply
    character-level noise. Optionally prepends BOS and applies right padding to a fixed length.

    Args:
        examples: Batch from datasets map, containing 'text' and optionally 'text_gt'.
        tokenizer: Hugging Face tokenizer instance.
        text_column_name: Column name to read text from when 'text_gt' is absent.
        prepend_bos: Whether to prepend BOS token.
        eos: EOS string used to split conversation segments.
        padding: If > 0, pad/truncate to this length.
        noise_prob: Probability of noise injection on 'text_gt' path.

    Returns:
        Dict[str, List[List[int]]]: Tokenized fields; includes 'labels' when built.
    """
    # if eos != tokenizer.eos_token:
    #     examples[text_column_name] = examples[text_column_name].replace(eos, tokenizer.eos_token)
    #     eos = tokenizer.eos_token
    if "text_gt" in examples:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        if noise_prob > 0:
            repeat = 20
        else:
            repeat = 1
        text_gt = examples["text_gt"] * repeat
        for i, text_i in enumerate(examples["text"] * repeat):
            splitted = text_i.split(eos)
            splitted = [s + eos for s in splitted if s != ""]  # add eos to each segment
            if noise_prob > 0:
                noisy_splitted = []
                for idx, s in enumerate(text_i.split(eos)):
                    if idx < len(text_gt[i]) and text_gt[i][idx] == "U":
                        noisy_splitted.append(
                            add_noise(s, noise_prob, noise_prob, noise_prob * 10, noise_prob) + eos
                        )
                    elif s != "":
                        noisy_splitted.append(s + eos)
                    else:
                        assert s == ""

                assert len(splitted) == len(
                    noisy_splitted
                ), f"Mismatch between splitted and noisy_splitted lengths: {splitted} != {noisy_splitted}"

            assert len(splitted) == len(
                text_gt[i]
            ), f"Mismatch between text and text_gt lengths: {splitted} != {(text_gt[i])}"

            if noise_prob > 0:
                tokens_i = tokenizer(noisy_splitted, add_special_tokens=False)
            else:
                tokens_i = tokenizer(splitted, add_special_tokens=False)

            # Build labels for each segment
            labels_i = []
            for j in range(len(tokens_i["input_ids"])):
                if text_gt[i][j] == "U":
                    labels_i.append([-100] * len(tokens_i["input_ids"][j]))
                else:
                    labels_i.append(tokens_i["input_ids"][j].copy())

            flat_input_ids = list(chain.from_iterable(tokens_i["input_ids"]))
            flat_attention_mask = list(chain.from_iterable(tokens_i["attention_mask"]))
            flat_labels = list(chain.from_iterable(labels_i))
            if prepend_bos:
                flat_input_ids = [tokenizer.bos_token_id] + flat_input_ids
                flat_attention_mask = [1] + flat_attention_mask
                flat_labels = [-100] + flat_labels
            if padding > 0:
                flat_input_ids = flat_input_ids + [tokenizer.eos_token_id] * (padding - len(flat_input_ids))
                flat_attention_mask = flat_attention_mask + [1] * (padding - len(flat_attention_mask))
                flat_labels = flat_labels + [-100] * (padding - len(flat_labels))
                if len(flat_input_ids) > padding:
                    # print(f"Warning: input_ids length {len(flat_input_ids)} is greater than padding {padding}")
                    flat_input_ids = flat_input_ids[:padding]
                    flat_attention_mask = flat_attention_mask[:padding]
                    flat_labels = flat_labels[:padding]
            input_ids_list.append(flat_input_ids)
            attention_mask_list.append(flat_attention_mask)
            labels_list.append(flat_labels)
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }
    else:
        tokens = tokenizer(examples[text_column_name])

        assert "labels" not in tokens
        if prepend_bos:
            for i in range(len(tokens["input_ids"])):
                tokens["input_ids"][i] = [tokenizer.bos_token_id] + tokens["input_ids"][i]
                if "attention_mask" in tokens:
                    tokens["attention_mask"][i] = [1] + tokens["attention_mask"][i]
        return tokens
