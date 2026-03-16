import torch


def test_crop_data_collator_crops_position_idx():
    from preprocess import crop_data_collator

    # Build a fake batch with right padding where labels are -100.
    # crop_data_collator should crop ALL token-aligned fields to max_valid.
    B = 2
    T = 20
    max_valid = 7

    features = []
    for b in range(B):
        input_ids = torch.arange(T, dtype=torch.long)
        labels = input_ids.clone()
        labels[max_valid:] = -100

        # position_idx is (T, 2)
        position_idx = torch.stack(
            [torch.arange(T, dtype=torch.long), torch.arange(T, dtype=torch.long)], dim=-1
        )

        features.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "position_idx": position_idx,
            }
        )

    out = crop_data_collator(features)
    assert out["labels"].shape == (B, max_valid)
    assert out["input_ids"].shape == (B, max_valid)
    assert out["position_idx"].shape == (B, max_valid, 2)
