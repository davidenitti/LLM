import torch

from model import CustomGPTConfig, CustomGPTmodel


def _make_position_idx(B: int, T: int, max_coord: int = 20) -> torch.Tensor:
    # (B, T, 2): (y, x)
    pos_y = torch.randint(0, max_coord, (B, T), dtype=torch.long)
    pos_x = torch.randint(0, max_coord, (B, T), dtype=torch.long)
    return torch.stack([pos_y, pos_x], dim=-1)


def test_use_rot_emb_2d_runs_with_selfattention():
    torch.manual_seed(0)
    B, T, V = 2, 16, 101
    config = CustomGPTConfig(
        vocab_size=V,
        context_len=64,
        embed_size=64,
        num_layers=2,
        heads=4,
        rotary_emb=8,
        use_rot_emb_2d=True,
        selfatt_class="SelfAttention",
        dropout=0.0,
    )
    model = CustomGPTmodel(config)

    input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
    position_idx = _make_position_idx(B, T)

    out = model(input_ids=input_ids, position_idx=position_idx)
    assert out.logits.shape == (B, T, V)


def test_use_rot_emb_2d_runs_with_selfattentionv2():
    torch.manual_seed(0)
    B, T, V = 2, 16, 101
    config = CustomGPTConfig(
        vocab_size=V,
        context_len=64,
        embed_size=64,
        num_layers=2,
        heads=4,
        rotary_emb=8,
        use_rot_emb_2d=True,
        selfatt_class="SelfAttentionV2",
        dropout=0.0,
    )
    model = CustomGPTmodel(config)

    input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
    position_idx = _make_position_idx(B, T)

    out = model(input_ids=input_ids, position_idx=position_idx)
    assert out.logits.shape == (B, T, V)


def test_use_rot_emb_2d_runs_with_cache_incremental_selfattention():
    torch.manual_seed(0)
    B, V = 2, 101
    T0, T1 = 8, 1
    config = CustomGPTConfig(
        vocab_size=V,
        context_len=64,
        embed_size=64,
        num_layers=2,
        heads=4,
        rotary_emb=8,
        use_rot_emb_2d=True,
        selfatt_class="SelfAttention",
        dropout=0.0,
    )
    model = CustomGPTmodel(config)

    input_ids0 = torch.randint(0, V, (B, T0), dtype=torch.long)
    position_idx0 = _make_position_idx(B, T0)
    out0 = model(input_ids=input_ids0, position_idx=position_idx0, return_cache=True)
    assert out0.logits.shape == (B, T0, V)

    k_cache = list(out0.past_key_values[0])
    v_cache = list(out0.past_key_values[1])

    input_ids1 = torch.randint(0, V, (B, T1), dtype=torch.long)
    position_idx1 = _make_position_idx(B, T1)
    out1 = model(
        input_ids=input_ids1,
        position_idx=position_idx1,
        k_cache=k_cache,
        v_cache=v_cache,
        return_cache=True,
    )
    assert out1.logits.shape == (B, T1, V)


def test_use_rot_emb_2d_runs_with_cache_incremental_selfattentionv2():
    torch.manual_seed(0)
    B, V = 2, 101
    T0, T1 = 8, 1
    config = CustomGPTConfig(
        vocab_size=V,
        context_len=64,
        embed_size=64,
        num_layers=2,
        heads=4,
        rotary_emb=8,
        use_rot_emb_2d=True,
        selfatt_class="SelfAttentionV2",
        dropout=0.0,
    )
    model = CustomGPTmodel(config)

    input_ids0 = torch.randint(0, V, (B, T0), dtype=torch.long)
    position_idx0 = _make_position_idx(B, T0)
    out0 = model(input_ids=input_ids0, position_idx=position_idx0, return_cache=True)
    assert out0.logits.shape == (B, T0, V)

    k_cache = list(out0.past_key_values[0])
    v_cache = list(out0.past_key_values[1])

    input_ids1 = torch.randint(0, V, (B, T1), dtype=torch.long)
    position_idx1 = _make_position_idx(B, T1)
    out1 = model(
        input_ids=input_ids1,
        position_idx=position_idx1,
        k_cache=k_cache,
        v_cache=v_cache,
        return_cache=True,
    )
    assert out1.logits.shape == (B, T1, V)


def test_generate_cache_maxlen1_allows_position_idx_prefix_len():
    torch.manual_seed(0)
    B, T, V = 2, 16, 101
    config = CustomGPTConfig(
        vocab_size=V,
        context_len=64,
        embed_size=64,
        num_layers=2,
        heads=4,
        rotary_emb=8,
        use_rot_emb_2d=True,
        selfatt_class="SelfAttention",
        dropout=0.0,
    )
    model = CustomGPTmodel(config)

    input_ids = torch.randint(0, V, (B, T), dtype=torch.long)
    position_idx = _make_position_idx(B, T)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=None,
        max_length=1,
        temperature=1.0,
        top_k=0,
        eos_token_id=None,
        return_cache=True,
        position_idx=position_idx,
    )
    assert out.shape == (B, T + 1)
