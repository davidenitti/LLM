import torch
import pytest

from model import CustomGPTConfig as HFCustomGPTConfig
from model import CustomGPTmodel as HFCustomGPTmodel
from model_standalone import CustomGPTConfig as StandaloneCustomGPTConfig
from model_standalone import CustomGPTmodel as StandaloneCustomGPTmodel


def _config_kwargs():
    return {
        "vocab_size": 64,
        "context_len": 24,
        "embed_size": 32,
        "num_layers": 2,
        "heads": 4,
        "rotary_emb": 4,
        "dropout": 0.0,
        "tie_word_embeddings": True,
    }


def _assert_same_state_dict(model_a, model_b):
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    assert set(state_dict_a.keys()) == set(state_dict_b.keys())
    for key in state_dict_a:
        assert torch.equal(state_dict_a[key], state_dict_b[key]), key


def test_standalone_tied_input_output_embeddings_share_weights():
    torch.manual_seed(0)
    model = StandaloneCustomGPTmodel(StandaloneCustomGPTConfig(**_config_kwargs()))

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight

    assert input_weight is output_weight
    assert input_weight.data_ptr() == output_weight.data_ptr()


def test_standalone_tie_word_embeddings_false_does_not_share_weights():
    torch.manual_seed(0)
    config = StandaloneCustomGPTConfig(**{**_config_kwargs(), "tie_word_embeddings": False})
    model = StandaloneCustomGPTmodel(config)

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight

    assert input_weight is not output_weight
    assert input_weight.data_ptr() != output_weight.data_ptr()


def test_standalone_config_matches_original_config_behavior():
    base_kwargs = _config_kwargs()
    original = HFCustomGPTConfig(**base_kwargs)
    standalone = StandaloneCustomGPTConfig(**base_kwargs)

    override = {"repeat_model": 3, "tie_word_embeddings": False, "aligned_labels": True}

    original.update_from_dict(override)
    standalone.update_from_dict(override)

    expected_params = original._custom_params_dict()
    expected_params.pop("rnn_version")
    expected_params.pop("step_size_rnn_like")

    assert standalone._custom_params_dict() == expected_params
    assert not hasattr(standalone, "rnn_version")
    assert not hasattr(standalone, "step_size_rnn_like")
    assert standalone.embed_size == original.embed_size
    assert standalone.context_len == original.context_len
    assert standalone.heads == original.heads
    assert standalone.num_layers == original.num_layers


def test_legacy_config_names_are_rejected():
    with pytest.raises(ValueError, match="Legacy config keys"):
        HFCustomGPTConfig(n_positions=24)

    with pytest.raises(ValueError, match="Legacy config keys"):
        StandaloneCustomGPTConfig(n_positions=24)


def test_standalone_model_matches_original_forward_and_cache():
    torch.manual_seed(0)
    original_config = HFCustomGPTConfig(**_config_kwargs())
    standalone_config = StandaloneCustomGPTConfig(**original_config._custom_params_dict())

    original_model = HFCustomGPTmodel(original_config).eval()
    standalone_model = StandaloneCustomGPTmodel(standalone_config).eval()

    missing, unexpected = standalone_model.load_state_dict(original_model.state_dict(), strict=True)
    assert missing == []
    assert unexpected == []

    input_ids = torch.randint(0, original_config.vocab_size, (2, 10), dtype=torch.long)

    original_out = original_model(input_ids=input_ids, return_cache=True)
    standalone_out = standalone_model(input_ids=input_ids, return_cache=True)

    assert torch.allclose(original_out.logits, standalone_out.logits, atol=1e-6, rtol=1e-6)
    assert (
        original_model.get_input_embeddings().weight.data_ptr()
        == original_model.get_output_embeddings().weight.data_ptr()
    )
    assert (
        standalone_model.get_input_embeddings().weight.data_ptr()
        == standalone_model.get_output_embeddings().weight.data_ptr()
    )

    for original_cache, standalone_cache in zip(
        original_out.past_key_values[0], standalone_out.past_key_values[0]
    ):
        assert torch.allclose(original_cache, standalone_cache, atol=1e-6, rtol=1e-6)
    for original_cache, standalone_cache in zip(
        original_out.past_key_values[1], standalone_out.past_key_values[1]
    ):
        assert torch.allclose(original_cache, standalone_cache, atol=1e-6, rtol=1e-6)


def test_standalone_resize_and_save_round_trip_preserves_tying(tmp_path):
    torch.manual_seed(0)
    config = StandaloneCustomGPTConfig(**_config_kwargs())
    model = StandaloneCustomGPTmodel(config).eval()

    model.resize_token_embeddings(config.vocab_size + 5)

    assert model.config.vocab_size == config.vocab_size
    assert model.get_input_embeddings().weight.shape[0] == config.vocab_size
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()

    save_dir = tmp_path / "standalone_ckpt"
    model.save_pretrained(str(save_dir))

    reloaded = StandaloneCustomGPTmodel.from_pretrained(str(save_dir)).eval()
    assert reloaded.config.vocab_size == model.config.vocab_size
    assert (
        reloaded.get_input_embeddings().weight.data_ptr()
        == reloaded.get_output_embeddings().weight.data_ptr()
    )

    input_ids = torch.randint(0, reloaded.config.vocab_size, (2, 6), dtype=torch.long)
    original_out = model(input_ids=input_ids)
    reloaded_out = reloaded(input_ids=input_ids)

    assert torch.allclose(original_out.logits, reloaded_out.logits, atol=1e-6, rtol=1e-6)


def test_standalone_save_load_round_trip_preserves_state_config_and_logits(tmp_path):
    torch.manual_seed(0)
    config = StandaloneCustomGPTConfig(**_config_kwargs())
    model = StandaloneCustomGPTmodel(config).eval()

    save_dir = tmp_path / "standalone_exact_ckpt"
    model.save_pretrained(str(save_dir))

    reloaded, load_info = StandaloneCustomGPTmodel.from_pretrained(
        str(save_dir), output_loading_info=True, strict=True
    )
    reloaded = reloaded.eval()

    assert load_info == {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}
    assert reloaded.config._custom_params_dict() == model.config._custom_params_dict()
    _assert_same_state_dict(model, reloaded)
    assert (
        reloaded.get_input_embeddings().weight.data_ptr()
        == reloaded.get_output_embeddings().weight.data_ptr()
    )

    input_ids = torch.randint(0, config.vocab_size, (2, 7), dtype=torch.long)
    original_out = model(input_ids=input_ids, return_cache=True)
    reloaded_out = reloaded(input_ids=input_ids, return_cache=True)

    assert torch.allclose(original_out.logits, reloaded_out.logits, atol=1e-6, rtol=1e-6)
    for original_cache, reloaded_cache in zip(
        original_out.past_key_values[0], reloaded_out.past_key_values[0]
    ):
        assert torch.allclose(original_cache, reloaded_cache, atol=1e-6, rtol=1e-6)
    for original_cache, reloaded_cache in zip(
        original_out.past_key_values[1], reloaded_out.past_key_values[1]
    ):
        assert torch.allclose(original_cache, reloaded_cache, atol=1e-6, rtol=1e-6)


def test_standalone_untied_save_load_round_trip_preserves_untied_weights(tmp_path):
    torch.manual_seed(0)
    config = StandaloneCustomGPTConfig(**{**_config_kwargs(), "tie_word_embeddings": False})
    model = StandaloneCustomGPTmodel(config).eval()

    save_dir = tmp_path / "standalone_untied_ckpt"
    model.save_pretrained(str(save_dir))

    reloaded = StandaloneCustomGPTmodel.from_pretrained(str(save_dir)).eval()

    assert reloaded.config.tie_word_embeddings is False
    assert (
        reloaded.get_input_embeddings().weight.data_ptr()
        != reloaded.get_output_embeddings().weight.data_ptr()
    )
    _assert_same_state_dict(model, reloaded)
