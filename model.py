import time
import os
import torch
import torchvision.utils as vutils
import math
from torch import nn
from einops import rearrange
from dataclasses import dataclass
import sys
import random
import json
from torch.nn import LayerNorm
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.loss.loss_utils import ForCausalLMLoss
from functools import partial
from checks import check_tensors, check_model
from rotary_pos import precompute_freqs, apply_rotary_emb

from blocks import SoftGradHardTanh, SoftGradHardSigmoid
from torch.nn import GELU

from basic_transformer import SelfAttention


class CustomGPTConfig(PretrainedConfig):
    """
    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    ```"""

    model_type = "custom_gptv0"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "embed_size": "n_embd",
        "context_len": "n_positions",
        "heads": "n_head",
        "num_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bias=True,
        n_inner=None,
        activation_function="gelu_new",
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        selfatt_class="SelfAttention",
        selfatt_class_kwargs={},
        random_weight_noise=0.0,
        megatron_init=True,
        norm_loss=0.0,
        rotary_emb=0,
        **kwargs,
    ):
        self.rotary_emb = rotary_emb
        self.norm_loss = norm_loss
        self.megatron_init = megatron_init
        self.random_weight_noise = random_weight_noise
        self.bias = bias
        self.selfatt_class = selfatt_class
        self.selfatt_class_kwargs = selfatt_class_kwargs
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        json like format:
        '"selfatt_class":"SelfAttentionDist","selfatt_class_kwargs":{"att_type":"dot","pos_emb_mode":"sum_mul_x"}'
        The keys to change have to already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

        """
        print("{" + update_str + "}")
        d = json.loads("{" + update_str + "}")
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")
            setattr(self, k, v)


def update_dict(d, d2):
    d.update(d2)
    return d


class CustomGPTmodel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.word_emb = nn.Embedding(config.vocab_size, config.embed_size)
        if config.rotary_emb > 0:
            self.register_buffer(
                "freqs",
                precompute_freqs(
                    qk_rope_dim=config.rotary_emb,
                    max_seq_len=config.context_len,
                    original_seq_len=config.context_len,
                ),
                persistent=False,
            )
        else:
            self.freqs = None
            self.pos_emb = nn.Embedding(config.context_len, config.embed_size)
        self.drop = nn.Dropout(config.dropout)
        selfatt_class = getattr(sys.modules[__name__], config.selfatt_class)
        if config.selfatt_class_kwargs.get("softmax_replacement", None) is not None:
            config.selfatt_class_kwargs["softmax_replacement"] = getattr(
                sys.modules[__name__], config.selfatt_class_kwargs["softmax_replacement"]
            )
        if config.selfatt_class_kwargs.get("norm_before_softmax", None) is not None:
            config.selfatt_class_kwargs["norm_before_softmax"] = getattr(
                sys.modules[__name__], config.selfatt_class_kwargs["norm_before_softmax"]
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
                    **update_dict(config.selfatt_class_kwargs, {"layer_id": layer_id}),
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_size, bias=config.bias, eps=config.layer_norm_epsilon)
        self.lin_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)
        self.loss_function = ForCausalLMLoss
        self.apply(self._init_weights)

        self.tie_weights()

    def save_pretrained(
        self,
        save_directory,
        is_main_process=True,
        state_dict=None,
        save_function=torch.save,
        push_to_hub=False,
        max_shard_size="5GB",
        safe_serialization=False,
        variant=None,
        token=None,
        save_peft_format=True,
        **kwargs,
    ):
        return super().save_pretrained(
            save_directory,
            is_main_process,
            state_dict,
            save_function,
            push_to_hub,
            max_shard_size,
            safe_serialization,
            variant,
            token,
            save_peft_format,
            **kwargs,
        )

    def tie_weights(self):
        self.lin_head.weight = self.word_emb.weight

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
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
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        if self.config.megatron_init:
            for name, p in module.named_parameters():
                if name == "c_proj.weight":
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, attention_mask=None):
        if attention_mask is not None and attention_mask.min() == 1:
            attention_mask = None
        if attention_mask is not None:
            print("attention_mask used!!")
            print(attention_mask)
        if self.config.random_weight_noise > 0 and self.training and random.random() < 0.01:
            for p in self.parameters():
                p.data += (torch.rand_like(p) < 0.001) * torch.randn_like(p) * self.config.random_weight_noise
        if inputs_embeds is None:
            inputs_embeds = self.word_emb(input_ids)
        else:
            print("input_embeds", inputs_embeds.shape, inputs_embeds.dtype)
        B, T = input_ids.size()
        features = inputs_embeds
        if self.config.rotary_emb <= 0:
            pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            features = features + self.pos_emb(pos)
        x = self.drop(features)
        for layer_id, block in enumerate(self.transf):
            x = block(x, attention_mask=attention_mask, freqs=self.freqs)
        if self.config.norm_loss > 0:
            x_not_norm = x
        check_tensors(x, "x before norm")
        x = self.norm(x)
        check_tensors(x, "x after norm")
        logits = self.lin_head(x)
        check_tensors(x, "logits")

        check_model(self)
        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                # **kwargs,
            )
            if self.training and self.config.norm_loss > 0:
                norm_loss = self.config.norm_loss * (x_not_norm**2).mean()
                loss += norm_loss
                if random.random() < 0.001:
                    print(f"norm_loss {norm_loss.detach().item():.3f}")

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            # past_key_values=transformer_outputs.past_key_values,
            # hidden_states=transformer_outputs.hidden_states,
            # attentions=transformer_outputs.attentions,
            # cross_attentions=transformer_outputs.cross_attentions,
        )

    def get_input_embeddings(self):
        return self.word_emb

    @torch.no_grad()
    def generate(self, idx, max_length, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_length):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.n_positions else idx[:, -self.config.n_positions :]
            # forward the model to get the logits for the index in the sequence
            out = self(idx_cond)
            logits = out.logits
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def custom_config(name_config):
    if name_config == "custom:gptv0":
        return CustomGPTConfig()
    # elif name_config == "custom:rnnv0":
    #     import rnn
    #     return rnn.CustomRNNConfig()
    else:
        raise NotImplementedError()


def custom_model(config):
    if isinstance(config, CustomGPTConfig):
        return CustomGPTmodel(config)
    # elif isinstance(config, rnn.CustomRNNConfig):
    #     return rnn.CustomRNNmodel(config)
    else:
        raise NotImplementedError()
