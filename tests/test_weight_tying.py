import torch

from model import CustomGPTConfig, CustomGPTmodel


def test_tied_input_output_embeddings_share_weights():
    torch.manual_seed(0)
    config = CustomGPTConfig(
        vocab_size=64,
        context_len=16,
        embed_size=32,
        num_layers=2,
        heads=4,
        rotary_emb=4,
        dropout=0.0,
    )
    model = CustomGPTmodel(config)

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight

    assert input_weight is output_weight
    assert input_weight.data_ptr() == output_weight.data_ptr()


def test_tie_word_embeddings_false_does_not_share_weights():
    torch.manual_seed(0)
    config = CustomGPTConfig(
        vocab_size=64,
        context_len=16,
        embed_size=32,
        num_layers=2,
        heads=4,
        rotary_emb=0,
        dropout=0.0,
        tie_word_embeddings=False,
    )
    model = CustomGPTmodel(config)

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight

    assert input_weight is not output_weight
    assert input_weight.data_ptr() != output_weight.data_ptr()


def test_update_from_dict_accepts_tie_word_embeddings_override():
    config = CustomGPTConfig()

    updated = config.update_from_dict({"tie_word_embeddings": False})

    assert config.tie_word_embeddings is False
    assert updated["tie_word_embeddings"] is False
