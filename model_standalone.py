import glob
import inspect
import json
import logging
import math
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from blocks.basic_transformer import LowRankLinear
from blocks.blocks import SoftGradHardSigmoid, SoftGradHardTanh
from loss import ArgmaxFocusLoss, FocalLMLoss, MultiMarginLMLoss, StandardLoss
from rotary_pos import precompute_freqs
from utils.checks import check_model
from utils.config_utils import convert_string_format_to_json_like

from blocks.basic_transformer import SelfAttention, SelfAttentionV2


logger = logging.getLogger(__name__)


@dataclass
class CausalLMOutputWithCrossAttentions:
    """Structured output for causal language modeling.

    Attributes:
        loss: Optional scalar training loss.
        logits: Token logits of shape (batch, seq_len, vocab_size).
        past_key_values: Optional tuple of cached K/V tensors for fast generation.
        hidden_states: Optional list of intermediate hidden states per layer.
        attentions: Optional list of attention maps per layer.
        cross_attentions: Optional list of cross-attention maps per layer.
    """

    loss: Optional[torch.FloatTensor] = None
    metrics: Optional[dict] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class CustomGPTConfig:
    """Configuration for the standalone custom GPT model.

    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the model.
        context_len (`int`, *optional*, defaults to 1024):
            Maximum sequence length the model can attend to.
        embed_size (`int`, *optional*, defaults to 768):
            Dimensionality of the token embeddings and hidden states.
        num_layers (`int`, *optional*, defaults to 12):
            Number of transformer blocks.
        heads (`int`, *optional*, defaults to 12):
            Number of attention heads in each transformer block.
        bias (`bool`, *optional*, defaults to `True`):
            Whether linear and normalization layers use bias terms.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability used throughout the model.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            Epsilon used by layer normalization.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation used to initialize weights.
        selfatt_class (`str`, *optional*, defaults to `"SelfAttention"`):
            Name of the attention block class to instantiate.
        selfatt_class_kwargs (`dict`, *optional*, defaults to `{}`):
            Extra keyword arguments passed to the attention block class.
        megatron_init (`bool`, *optional*, defaults to `True`):
            Whether to apply Megatron-style scaled initialization to residual projections.
        rotary_emb (`int`, *optional*, defaults to 0):
            Rotary embedding dimension. Set to 0 to use learned absolute position embeddings.
        repeat_block (`int`, *optional*, defaults to 1):
            Number of times to repeat each middle block when block repetition is enabled.
        random_blocks (`bool`, *optional*, defaults to `False`):
            Whether to shuffle the order of the middle blocks during training.
        repeat_model (`int`, *optional*, defaults to 1):
            Number of times to repeat the full middle stack of blocks.
        random_repeat (`bool`, *optional*, defaults to `False`):
            Whether to sample the repeat count from the curriculum range.
        loss (`str`, *optional*, defaults to `"StandardLoss"`):
            Name of the loss function defined in this module.
        reduce_loss (`str`, *optional*, defaults to `"mean"`):
            Reduction to apply to the loss. Can be `"mean"` or `"sum"`.
        aligned_labels (`bool`, *optional*, defaults to `False`):
            Whether labels are already aligned with logits instead of requiring next-token shifting.
        curriculum (`bool`, *optional*, defaults to `True`):
            Whether repeat counts are ramped up during training.
        use_pos_emb_2d (`bool`, *optional*, defaults to `False`):
            Whether to add learned 2D positional embeddings from `position_idx`.
        use_rot_emb_2d (`bool`, *optional*, defaults to `False`):
            Whether to build 2D rotary embeddings from `position_idx`.
        all_repeat_losses (`bool`, *optional*, defaults to `False`):
            Whether to compute extra losses at intermediate repeated blocks.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to share input embedding weights with the output LM head.

    Notes:
        The standalone variant stores and serializes only the repo's preferred
        field names: `context_len`, `embed_size`, `num_layers`, and `heads`.
    """

    model_type = "custom_gptv0_standalone"
    keys_to_ignore_at_inference = ["past_key_values"]
    legacy_keys = {"max_position_embeddings", "n_positions", "n_embd", "n_head", "n_layer"}

    def __init__(
        self,
        vocab_size=50257,
        context_len=1024,
        embed_size=768,
        num_layers=12,
        heads=12,
        bias=True,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        selfatt_class="SelfAttention",
        selfatt_class_kwargs={},
        megatron_init=True,
        rotary_emb=0,
        repeat_block=1,
        random_blocks=False,
        repeat_model=1,
        random_repeat=False,
        loss="StandardLoss",
        reduce_loss="mean",
        aligned_labels=False,
        curriculum=True,
        use_pos_emb_2d=False,
        use_rot_emb_2d=False,
        all_repeat_losses=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        legacy_keys = self.legacy_keys.intersection(kwargs)
        if legacy_keys:
            raise ValueError(
                f"Legacy config keys are no longer supported: {sorted(legacy_keys)}. "
                "Use context_len, embed_size, num_layers, and heads instead."
            )
        self.loss = loss
        self.aligned_labels = aligned_labels
        self.reduce_loss = reduce_loss
        assert reduce_loss in ["mean", "sum"]
        self.repeat_model = repeat_model
        self.random_repeat = random_repeat
        self.random_blocks = random_blocks
        self.repeat_block = repeat_block
        self.rotary_emb = rotary_emb
        self.megatron_init = megatron_init
        self.bias = bias
        self.selfatt_class = selfatt_class
        self.selfatt_class_kwargs = dict(selfatt_class_kwargs)
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.curriculum = curriculum
        self.use_pos_emb_2d = use_pos_emb_2d
        self.use_rot_emb_2d = use_rot_emb_2d
        self.all_repeat_losses = all_repeat_losses
        self.tie_word_embeddings = tie_word_embeddings
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"

    def to_dict(self):
        """Serialize the config to a plain Python dict."""

        data = self._custom_params_dict()
        for key, value in self.__dict__.items():
            if key.startswith("_") or callable(value):
                continue
            if key not in data:
                data[key] = deepcopy(value)
        data["model_type"] = self.model_type
        return data

    @classmethod
    def from_dict(cls, config_dict):
        config_dict = dict(config_dict)
        config_dict.pop("model_type", None)
        return cls(**config_dict)

    def save_pretrained(self, save_directory):
        """Save the config as `config.json` inside `save_directory`."""

        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a config from a directory or a direct path to `config.json`."""

        config_path = pretrained_model_name_or_path
        if os.path.isdir(config_path):
            config_path = os.path.join(config_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        config_dict.update(kwargs)
        return cls.from_dict(config_dict)

    def update_from_string(self, update_str: str):
        """Update config attributes from `update_str`.

        The expected format is a relaxed JSON-like string such as:
        `"selfatt_class":"SelfAttentionDist","selfatt_class_kwargs":{"att_type":"dot"}`

        The keys to change must already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated.
        """

        update_str = convert_string_format_to_json_like(update_str)
        print(update_str)
        update_dict = json.loads(update_str)
        return self.update_from_dict(update_dict)

    def update_from_dict(self, update_dict):
        """Update config attributes from `update_dict`.

        Returns:
            dict: Only the parameters defined directly in `CustomGPTConfig`.
        """

        assert isinstance(update_dict, dict)
        for key, value in update_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"key {key} isn't in the original config dict")
            setattr(self, key, value)
        return self._custom_params_dict()

    def _custom_params_dict(self):
        """Return only the parameters defined directly on CustomGPTConfig."""

        param_names = [
            name
            for name, param in inspect.signature(self.__class__.__init__).parameters.items()
            if name != "self"
            and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        ]
        return {name: deepcopy(getattr(self, name)) for name in param_names}


def update_dict(d, d2):
    """Return a new dict with `d2` layered on top of `d`."""

    new_dict = d.copy()
    new_dict.update(d2)
    return new_dict


def min_max_repeat_curriculum(
    base_value: int,
    progress: float,
    min_progress: float = 0.1,
    target_progress: float = 0.7,
    curriculum: bool = True,
) -> int:
    """Scale the maximum repeat count linearly with training progress.

    Args:
        base_value: Upper bound to reach by `target_progress`.
        progress: Fraction of training completed in `[0, 1]`.
        target_progress: Fraction of training when `base_value` is fully reached.
        curriculum: Whether to apply the ramp or just return the full range.

    Returns:
        Tuple `(min_repeat, max_repeat)` derived from the current progress.
    """

    if not curriculum:
        return max(1, base_value // 2), base_value
    progress = min(1.0, max(0.0, progress))
    if base_value <= 1:
        return base_value
    if target_progress <= 0:
        ramp_factor = 1.0
    else:
        ramp_factor = min((min_progress + progress) / (target_progress + min_progress), 1.0)
    scaled = int(math.ceil(max(1.0, ramp_factor * base_value)))
    scaled = min(base_value, scaled)
    if random.random() < 0.00001:
        print(
            f"ramped_max: base_value {base_value} progress {progress:.4f} range {max(1, scaled // 2)} {scaled}"
        )
    return max(1, scaled // 2), scaled


def unfold_input(input, combine_tokens):
    """Create a sliding-window unfolded view across the last dimension.

    Expects input with shape `(B, T, C, E)` where `E == combine_tokens`. Pads on the
    left by `combine_tokens` zeros, then creates windows of size `combine_tokens`
    with step 1 over the last dimension and returns a tensor shaped like the
    original `(B, T, C, E)` but with concatenated windowed features.

    Args:
        input: 4D tensor `(batch, time, channels, embed)` with `embed == combine_tokens`.
        combine_tokens: Size of the sliding window.

    Returns:
        Tensor shaped `(B, T, C, E)` containing unfolded features.
    """

    assert input.ndim == 4
    assert input.shape[-1] == combine_tokens
    input = nn.functional.pad(input, (combine_tokens, 0), mode="constant", value=0)
    input_tensor_unfold = rearrange(input, "b t c e -> (b t) c e", b=input.shape[0], t=input.shape[1])
    input_tensor_unfold = input_tensor_unfold.contiguous()
    input_tensor_unfold = input_tensor_unfold.unfold(dimension=2, size=combine_tokens, step=1)
    input_tensor_unfold = input_tensor_unfold.permute(0, 2, 1, 3)
    input_tensor_unfold = input_tensor_unfold.reshape(
        input_tensor_unfold.size(0), input_tensor_unfold.size(1), -1
    )
    input_tensor_unfold = rearrange(
        input_tensor_unfold, "(b t) c e -> b t c e", b=input.shape[0], t=input.shape[1]
    )
    return input_tensor_unfold


class CustomGPTmodel(nn.Module):
    config_class = CustomGPTConfig

    def __init__(self, config: CustomGPTConfig):
        """Initialize the GPT-like model from a `CustomGPTConfig`.

        Sets up token and position embeddings, transformer blocks (potentially
        custom attention variants), optional rotary embeddings, output head,
        and loss function. Also applies weight initialization and optionally
        ties input and output embeddings.
        """

        super().__init__()
        self.config = config
        self.word_emb = nn.Embedding(config.vocab_size, config.embed_size)
        if config.rotary_emb > 0:
            self.register_buffer(
                "freqs",
                precompute_freqs(
                    qk_rope_dim=config.rotary_emb if not config.use_rot_emb_2d else config.rotary_emb // 2,
                    max_seq_len=config.context_len,
                    original_seq_len=config.context_len,
                ),
                persistent=False,
            )
        else:
            self.freqs = None
            self.pos_emb = nn.Embedding(config.context_len, config.embed_size)
        if config.use_pos_emb_2d:
            self.pos_emb_x = nn.Embedding(48, config.embed_size)
            self.pos_emb_y = nn.Embedding(48, config.embed_size)
        self.drop = nn.Dropout(config.dropout)
        selfatt_class = getattr(sys.modules[__name__], config.selfatt_class)
        selfatt_class_kwargs = dict(config.selfatt_class_kwargs)
        if selfatt_class_kwargs.get("softmax_replacement", None) is not None:
            selfatt_class_kwargs["softmax_replacement"] = getattr(
                sys.modules[__name__],
                selfatt_class_kwargs["softmax_replacement"],
            )
        if selfatt_class_kwargs.get("norm_before_softmax", None) is not None:
            selfatt_class_kwargs["norm_before_softmax"] = getattr(
                sys.modules[__name__],
                selfatt_class_kwargs["norm_before_softmax"],
            )
        self.transf = nn.ModuleList(
            [
                selfatt_class(
                    config.heads,
                    config.embed_size,
                    config.bias,
                    config.context_len,
                    config.dropout,
                    eps_norm=config.layer_norm_epsilon,
                    **update_dict(selfatt_class_kwargs, {"layer_id": layer_id}),
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_size, bias=config.bias, eps=config.layer_norm_epsilon)
        self.lin_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)
        self.loss_function = getattr(sys.modules[__name__], self.config.loss)
        self.apply(self._init_weights)

        # Standalone models do not inherit HF's automatic weight tying logic,
        # so input/output sharing is handled explicitly here.
        self.tie_weights()

    def get_input_embeddings(self):
        """Return the token embedding module used by the model."""

        return self.word_emb

    def set_input_embeddings(self, new_embeddings):
        """Set the token embedding module used by the model."""

        self.word_emb = new_embeddings
        if self.config.tie_word_embeddings:
            self.tie_weights()

    def get_output_embeddings(self):
        """Return the output embedding (LM head) module."""

        return self.lin_head

    def set_output_embeddings(self, new_embeddings):
        """Set the output embedding (LM head) module."""

        self.lin_head = new_embeddings
        if self.config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        """Tie output projection weights to input embedding weights when requested.

        When `tie_word_embeddings` is disabled after being enabled, the LM head is
        re-materialized as an independent parameter tensor.
        """

        if self.config.tie_word_embeddings:
            self.lin_head.weight = self.word_emb.weight
        elif self.lin_head.weight is self.word_emb.weight:
            self.lin_head.weight = nn.Parameter(self.word_emb.weight.detach().clone())
        return self

    def _init_weights(self, module):
        """Initialize module weights following Transformer conventions.

        - Linear/Conv/Embedding: Normal(0, initializer_range)
        - LayerNorm: weight=1, bias=0
        - Optionally apply Megatron-style scaled init to attention proj ('c_proj').
        """

        if isinstance(module, LowRankLinear):
            module.reset_parameters(init_std=self.config.initializer_range)
            return

        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
            if getattr(module, "_skip_init", False):
                return
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                print("PADDING", module.__class__)
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/sqrt(N) where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        if self.config.megatron_init:
            scale = 1.0 / math.sqrt(2 * self.config.num_layers)
            # Apply Megatron-style scaled init to residual projections named `c_proj`.
            # We do this structurally (by inspecting `module.c_proj`) rather than walking
            # `named_parameters()` recursively, to avoid repeatedly touching the same params.
            c_proj = getattr(module, "c_proj", None)
            if c_proj is not None:
                logger.debug(
                    f"Applying Megatron-style init scaling to {module.__class__}.c_proj with scale {scale:.4f}"
                )
                if isinstance(c_proj, nn.Linear):
                    c_proj.weight.data.normal_(mean=0.0, std=(self.config.initializer_range * scale))
                elif isinstance(c_proj, LowRankLinear):
                    # Reinitialize low-rank c_proj so the fused weight matches the scaled std.
                    c_proj.reset_parameters(init_std=(self.config.initializer_range * scale))

    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings and the LM head while preserving existing weights.

        Args:
            new_num_tokens: New vocabulary size.

        Returns:
            The resized input embedding module.
        """

        old_num_tokens, embed_dim = self.word_emb.weight.shape
        if new_num_tokens is None or new_num_tokens == old_num_tokens:
            return self.word_emb

        old_input_weight = self.word_emb.weight.data.clone()
        old_output_weight = self.lin_head.weight.data.clone()

        new_embeddings = nn.Embedding(new_num_tokens, embed_dim)
        self._init_weights(new_embeddings)
        num_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_to_copy] = old_input_weight[:num_to_copy]
        self.word_emb = new_embeddings

        new_lm_head = nn.Linear(embed_dim, new_num_tokens, bias=False)
        self._init_weights(new_lm_head)
        new_lm_head.weight.data[:num_to_copy] = old_output_weight[:num_to_copy]
        self.lin_head = new_lm_head

        self.config.vocab_size = new_num_tokens
        self.tie_weights()
        return self.word_emb

    def save_pretrained(
        self,
        save_directory,
        is_main_process=True,
        save_function=torch.save,
        state_dict=None,
        **kwargs,
    ):
        """Save config and weights in a lightweight HF-like directory layout.

        Args:
            save_directory: Output directory.
            is_main_process: Skip saving on non-main processes.
            save_function: Function used to serialize the state dict.
            state_dict: Optional precomputed state dict. Defaults to `self.state_dict()`.
        """

        del kwargs
        if not is_main_process:
            return
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        if state_dict is None:
            state_dict = self.state_dict()
        save_function(state_dict, os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        config=None,
        map_location="cpu",
        strict=True,
        output_loading_info=False,
        **kwargs,
    ):
        """Load a standalone model from a directory or direct weight file path.

        Args:
            pretrained_model_name_or_path: Directory containing `config.json` and weights,
                or a direct path to a `.bin` file.
            config: Optional config instance. Loaded from disk when omitted.
            map_location: Device mapping passed to `torch.load`.
            strict: Whether `load_state_dict` should enforce an exact key match.
            output_loading_info: Whether to return missing/unexpected key info.

        Returns:
            `CustomGPTmodel`, or `(model, loading_info)` when `output_loading_info=True`.
        """

        del kwargs
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        weight_path = pretrained_model_name_or_path
        if os.path.isdir(pretrained_model_name_or_path):
            candidate_paths = [
                os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
                os.path.join(pretrained_model_name_or_path, "model.bin"),
            ]
            candidate_paths.extend(sorted(glob.glob(os.path.join(pretrained_model_name_or_path, "*.bin"))))
            weight_path = next((path for path in candidate_paths if os.path.exists(path)), None)
            if weight_path is None:
                raise FileNotFoundError(
                    f"Could not find a .bin weights file in {pretrained_model_name_or_path}"
                )
        state_dict = torch.load(weight_path, map_location=map_location)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if output_loading_info:
            return model, {"missing_keys": missing, "unexpected_keys": unexpected, "mismatched_keys": []}
        return model

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        position_idx=None,
        attention_mask=None,
        k_cache=None,
        v_cache=None,
        return_cache=False,
        eval_mode=False,
        train_progress=1.0,
        **kwargs,
    ):
        """Forward pass for causal LM with optional caching and variants.

        Args:
            input_ids: (B, T) token ids. Required if inputs_embeds is None.
            inputs_embeds: (B, T, E) precomputed embeddings; overrides input_ids.
            labels: (B, T) target token ids for computing training loss.
            attention_mask: Optional mask (B, T) with 1 for real tokens and 0 for padding.
            k_cache, v_cache: Optional per-layer caches for fast decoding.
            return_cache: If True, returns updated caches for incremental decoding.
            train_progress: Optional float in [0, 1] indicating training step / total steps.

        Returns:
            CausalLMOutputWithCrossAttentions with logits (B, T, V), optional loss,
            and optional past_key_values when return_cache=True.
        """

        del kwargs
        if attention_mask is not None and attention_mask.min() == 1:
            # Skip mask handling when the mask is all ones to speed up forward.
            attention_mask = None
        random_repeat = self.config.random_repeat and not return_cache and not eval_mode
        progress = min(1.0, max(0.0, train_progress))

        if inputs_embeds is None:
            inputs_embeds = self.word_emb(input_ids)
        else:
            print("input_embeds", inputs_embeds.shape, inputs_embeds.dtype)
        B, T = input_ids.size()
        features = inputs_embeds
        cache_len = 0
        if self.config.rotary_emb <= 0 and (k_cache is not None or v_cache is not None):
            cache_src = k_cache if k_cache is not None else v_cache
            if isinstance(cache_src, (list, tuple)):
                for entry in cache_src:
                    if entry is not None:
                        cache_len = int(entry.shape[-2])
                        break
            elif hasattr(cache_src, "shape"):
                cache_len = int(cache_src.shape[-2])
        if self.config.rotary_emb <= 0:
            # Offset absolute positions during cached decoding so they match full-sequence indices.
            pos_start = cache_len
            if pos_start + T > self.config.context_len:
                pos_start = max(0, self.config.context_len - T)
                print("WARNING: trimming pos_start to fit within context_len")
            pos = torch.arange(pos_start, pos_start + T, dtype=torch.long, device=input_ids.device)
            features = features + self.pos_emb(pos)
        if self.config.use_pos_emb_2d:
            assert position_idx is not None
            assert position_idx.shape == (B, T, 2)
            position_idx_y = position_idx[:, :, 0]
            position_idx_x = position_idx[:, :, 1]
            pos_x_emb = self.pos_emb_x(position_idx_x)
            pos_y_emb = self.pos_emb_y(position_idx_y)
            features = features + pos_x_emb + pos_y_emb
        if self.config.use_rot_emb_2d:
            assert position_idx is not None
            assert position_idx.shape == (
                B,
                T,
                2,
            ), f"position_idx shape {position_idx.shape} expected {(B, T, 2)}"
            position_idx_y = position_idx[:, :, 0]
            position_idx_x = position_idx[:, :, 1]

            assert self.freqs is not None, "use_rot_emb_2d requires rotary_emb > 0"

            num_freqs = self.freqs.shape[1]
            freqs_x = self.freqs[position_idx_x.reshape(-1)].view(B, T, num_freqs, 2)
            freqs_y = self.freqs[position_idx_y.reshape(-1)].view(B, T, num_freqs, 2)
            freqs_cis = torch.cat([freqs_x, freqs_y], dim=-2)  # (B, T, rotary_emb/2, 2)
        else:
            freqs_cis = self.freqs
        x = self.drop(features)
        if k_cache is None:
            k_cache = [None for _ in range(self.config.num_layers * self.config.repeat_model)]
        if v_cache is None:
            v_cache = [None for _ in range(self.config.num_layers * self.config.repeat_model)]
        is_before_out_idx = []
        if self.config.repeat_block > 1:
            min_repeat, max_repeat = min_max_repeat_curriculum(
                self.config.repeat_block, progress, curriculum=self.config.curriculum
            )
            assert not self.config.random_blocks
            assert self.config.repeat_model == 1
            transf = []
            for layer_id, block in enumerate(self.transf):
                if layer_id == 0 or layer_id == len(self.transf) - 1:
                    repeat_block = 1
                else:
                    if random_repeat:
                        repeat_block = random.randint(min_repeat, max_repeat)
                    else:
                        repeat_block = max_repeat
                for _ in range(repeat_block):
                    transf.append(block)
        elif self.config.random_blocks:
            transf = (
                [self.transf[0]]
                + random.sample(list(self.transf)[1:-1], len(self.transf) - 2)
                + [self.transf[-1]]
            )
        elif self.config.repeat_model > 1:
            min_repeat, max_repeat = min_max_repeat_curriculum(
                self.config.repeat_model, progress, curriculum=self.config.curriculum
            )
            transf = [self.transf[0]]
            is_before_out_idx.append(False)
            if random_repeat:
                num_repeats = random.randint(min_repeat, max_repeat)
            else:
                num_repeats = max_repeat
            for idx_rep in range(num_repeats):
                transf += list(self.transf)[1:-1]
                if idx_rep < min_repeat:
                    is_before_out_idx += [False] * (len(self.transf) - 2)
                else:
                    is_before_out_idx += [False] * (len(self.transf) - 3) + [True]
            transf.append(self.transf[-1])
            is_before_out_idx.append(False)
        else:
            transf = self.transf
        k_cache = k_cache[: len(transf)]
        v_cache = v_cache[: len(transf)]
        additional_logits = []
        for layer_id, block in enumerate(transf):
            if v_cache[layer_id] is not None:
                # Trim cache if total sequence would exceed the model context.
                if v_cache[layer_id].shape[-2] + x.shape[1] > self.config.context_len:
                    num_pos = self.config.context_len - x.shape[1]
                    v_cache[layer_id] = v_cache[layer_id][:, :, -num_pos:]
                    k_cache[layer_id] = k_cache[layer_id][:, :, -num_pos:]
                    assert v_cache[layer_id].shape[-2] + x.shape[1] == self.config.context_len
            x = block(
                x,
                attention_mask=attention_mask,
                freqs=freqs_cis,
                v_cache=v_cache[layer_id],
                k_cache=k_cache[layer_id],
                return_cache=return_cache,
            )
            if (
                labels is not None
                and self.config.all_repeat_losses
                and is_before_out_idx != []
                and is_before_out_idx[layer_id]
                and not return_cache
            ):
                add_out = transf[-1](
                    x,
                    attention_mask=attention_mask,
                    freqs=freqs_cis,
                    v_cache=v_cache[layer_id],
                    k_cache=k_cache[layer_id],
                    return_cache=return_cache,
                )
                add_out = self.norm(add_out)
                additional_logits.append(self.lin_head(add_out))
            if return_cache:
                k_cache[layer_id] = x[1]
                v_cache[layer_id] = x[2]
                x = x[0]
        x = self.norm(x)
        logits = self.lin_head(x)
        check_model(self)
        loss = None
        metrics = {}
        if labels is not None:
            if self.config.reduce_loss == "sum" and self.training:
                num_items_in_batch = 1
            else:
                num_items_in_batch = None
            if self.config.all_repeat_losses and len(additional_logits) > 0:
                additional_logits = torch.stack(additional_logits, dim=0)
                additional_labels = labels.unsqueeze(0).expand_as(additional_logits[:, :, :, 0])
                add_loss, _ = self.loss_function(
                    additional_logits,
                    additional_labels,
                    vocab_size=self.config.vocab_size,
                    num_items_in_batch=num_items_in_batch,
                    aligned=self.config.aligned_labels,
                )
                if self.config.reduce_loss == "sum" and self.training:
                    add_loss /= additional_logits.shape[0]
            loss, metrics = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=num_items_in_batch,
                aligned=self.config.aligned_labels,
            )
            if self.config.all_repeat_losses and len(additional_logits) > 0:
                loss = 0.5 * loss + 0.5 * add_loss
            if self.config.reduce_loss == "sum" and self.training:
                loss *= 1.0 / self.config.context_len / logits.shape[0]

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            metrics=metrics,
            logits=logits,
            past_key_values=(
                tuple(k_cache) if return_cache else None,
                tuple(v_cache) if return_cache else None,
            ),
        )

    def generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        temperature=1.0,
        top_k=None,
        eos_token_id=None,
        pad_token_id=None,
        token_type_ids=[],
        return_cache=False,
        debug=False,
        train_progress=1.0,
        position_idx=None,
    ):
        """Autoregressively extend a prompt, optionally reusing attention caches.

        Args:
            input_ids: Conditioning token ids of shape `(B, T)`.
            attention_mask: Optional attention mask. The standalone path currently expects
                either `None` or an all-ones mask.
            max_length: Number of new tokens to generate.
            temperature: Logit temperature.
            top_k: If set, restrict sampling to the top-k logits. Use `0` for greedy decoding.
            eos_token_id: Optional stop token id.
            pad_token_id: Unused, kept for API compatibility.
            token_type_ids: Unused, kept for API compatibility.
            return_cache: Whether to reuse K/V caches across decoding steps.
            debug: Whether to print per-step debugging information.
            train_progress: Forwarded to `forward` for curriculum-aware decoding.
            position_idx: Optional 2D positions of shape `(B, T_total, 2)`.

        Returns:
            Token ids containing the original prompt followed by the generated tokens.
        """

        del pad_token_id, token_type_ids
        if attention_mask is not None and attention_mask.min() == 1:
            attention_mask = None
        assert attention_mask is None
        if position_idx is not None:
            if position_idx.dim() != 3 or position_idx.shape[0] != input_ids.shape[0]:
                raise ValueError(
                    f"position_idx must be (B, T, 2) and match batch size; got {position_idx.shape}"
                )
            # When decoding autoregressively we only need position indices for tokens that
            # are actually fed into the model.
            #
            # For `max_length=1`, the newly generated token is never re-fed (the loop ends),
            # so requiring an additional position slot is unnecessarily strict and can
            # cause an off-by-one failure.
            required_pos_len = input_ids.shape[1] + max(0, int(max_length) - 1)
            if position_idx.shape[1] < required_pos_len:
                raise ValueError(
                    "position_idx length is too short for generation: "
                    f"need >= {required_pos_len}, got {position_idx.shape[1]}"
                )
            if position_idx.device != input_ids.device:
                position_idx = position_idx.to(device=input_ids.device)
        done = torch.zeros((input_ids.size(0), 1), dtype=torch.bool, device=input_ids.device)
        k_cache = None
        v_cache = None
        idx_next = input_ids
        pos_idx_next = None
        for step_id in range(max_length):
            if return_cache:
                idx_cond = idx_next
                if position_idx is not None:
                    if pos_idx_next is None:
                        pos_idx_cond = position_idx[:, : idx_cond.shape[1], :]
                    else:
                        pos_idx_cond = pos_idx_next
                if k_cache is not None:
                    k_cache = [k[:, -self.config.context_len :] for k in k_cache]
                    v_cache = [v[:, -self.config.context_len :] for v in v_cache]
            else:
                # Recompute from the full prefix, cropped to the model context window.
                idx_cond = (
                    input_ids
                    if input_ids.size(1) <= self.config.context_len
                    else input_ids[:, -self.config.context_len :]
                )
                if position_idx is not None:
                    # `position_idx` can contain positions for future decode steps.
                    # First keep only positions for the tokens already generated,
                    # then take the trailing slice that matches the cropped `idx_cond`.
                    pos_idx_full = position_idx[:, : input_ids.shape[1], :]
                    pos_idx_cond = pos_idx_full[:, -idx_cond.shape[1] :, :]
            # Forward the model to get the logits for the index in the sequence.
            out = self(
                idx_cond,
                position_idx=pos_idx_cond if position_idx is not None else None,
                k_cache=k_cache,
                v_cache=v_cache,
                return_cache=return_cache,
                train_progress=train_progress,
            )
            if return_cache and out.past_key_values[0] is not None:
                k_cache = list(out.past_key_values[0])
                v_cache = list(out.past_key_values[1])
            logits = out.logits
            if debug:
                print("logits", logits[:, -1, :])
            # Use the final-step logits as the next-token distribution, then apply temperature.
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options.
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("Inf")
            if top_k == 0:
                # Greedy decoding.
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Apply softmax to convert logits to probabilities before sampling.
                probs = F.softmax(logits, dim=-1)
                # Sample from the distribution.
                idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue.
            if debug:
                print("idx_next", idx_next)
                breakpoint()
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            # Only compute the position indices for the next iteration.
            # The final generated token is not fed back in, so its position is unused.
            if return_cache and position_idx is not None and step_id < max_length - 1:
                pos_idx_next = position_idx[:, input_ids.shape[1] - 1 : input_ids.shape[1], :]
            if eos_token_id is not None:
                assert done.shape == idx_next.shape, f"{done.shape} != {idx_next.shape}"
                done = done | (idx_next == eos_token_id)
                if done.all():
                    break
        return input_ids
