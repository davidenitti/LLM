import torch

from model import CustomGPTConfig, CustomGPTmodel


def test_cache_pos_emb_matches_full_forward():
    torch.manual_seed(0)
    B, T, V = 2, 8, 64
    config = CustomGPTConfig(
        vocab_size=V,
        context_len=32,
        embed_size=32,
        num_layers=2,
        heads=4,
        rotary_emb=0,
        dropout=0.0,
    )
    model = CustomGPTmodel(config)
    model.eval()

    input_ids = torch.randint(0, V, (B, T), dtype=torch.long)

    full = model(input_ids=input_ids)
    prefix = model(input_ids=input_ids[:, :-1], return_cache=True)
    cached = model(
        input_ids=input_ids[:, -1:],
        k_cache=list(prefix.past_key_values[0]),
        v_cache=list(prefix.past_key_values[1]),
        return_cache=True,
    )

    assert torch.allclose(full.logits[:, -1, :], cached.logits[:, -1, :], atol=1e-5, rtol=1e-5)
