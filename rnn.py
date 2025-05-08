"""
RNN models
work in progress!
to test and check!
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.loss.loss_utils import ForCausalLMLoss
import json
import math
import random
import sys
from einops import rearrange

from blocks import SoftGradHardTanh, SoftGradHardSigmoid
from utils import convert_string_format_to_json_like

class RNNCell(nn.Module):
    def __init__(
        self, input_size, hidden_size, activation, norm=False, sum_residual=False, out_layer=False, dropout=0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.transition = nn.Sequential(nn.Linear(input_size + hidden_size, hidden_size), activation)
        if norm:
            self.norm = nn.LayerNorm(hidden_size)
        else:
            self.norm = nn.Identity()
        if out_layer:
            self.out_layer = nn.Linear(hidden_size, hidden_size)
        else:
            self.out_layer = nn.Identity()
        self.sum_residual = sum_residual
        if self.sum_residual:
            assert self.hidden_size == input_size, "sum_residual only works if hidden_size == input_size"
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, prev_hidden):
        if prev_hidden is None:
            prev_hidden = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        combined = torch.cat((x, prev_hidden), 1)
        hidden = self.transition(combined)
        out = self.out_layer(self.norm(hidden))
        if self.sum_residual:
            out = out + x
        out = self.dropout(out)
        return out, hidden


class LSTMCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        act_sigmoid,
        act_tanh,
        sum_residual=False,
        out_layer=False,
        norm=False,
        norm_c=False,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.act_sigmoid = act_sigmoid
        self.act_tanh = act_tanh
        self.norm = nn.LayerNorm(hidden_size) if norm else nn.Identity()
        self.norm_c = nn.LayerNorm(hidden_size) if norm_c else nn.Identity()
        self.transition = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        self.sum_residual = sum_residual
        if self.sum_residual:
            assert self.hidden_size == input_size, "sum_residual only works if hidden_size == input_size"
        if out_layer:
            self.out_layer = nn.Linear(hidden_size, hidden_size)
        else:
            self.out_layer = nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, prev):
        if prev is None:
            prev_hidden = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            prev_cell = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        else:
            prev_hidden, prev_cell = prev

        gates = self.transition(torch.cat((x, prev_hidden), 1))
        inp_gate, cell_gate, out_gate = gates.chunk(3, 1)
        inp_gate = self.act_sigmoid(inp_gate)
        cell_gate = self.act_tanh(cell_gate)
        out_gate = self.act_sigmoid(out_gate)
        cell = (1 - inp_gate) * prev_cell + inp_gate * cell_gate
        hidden = out_gate * self.act_tanh(self.norm_c(cell))
        out = self.out_layer(self.norm(hidden))
        if self.sum_residual:
            out = out + x
        out = self.dropout(out)
        return out, (hidden, cell)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_cell, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.rnn_cells = [rnn_cell(input_size, hidden_size, **kwargs)] + [
            rnn_cell(hidden_size, hidden_size, **kwargs) for _ in range(num_layers - 1)
        ]
        self.rnn = nn.ModuleList(self.rnn_cells)
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None, detach_steps=10000):
        # x is of shape (batch_size, seq_len, input_size)
        if hidden is None:
            hidden = [None for _ in range(self.num_layers)]
        outputs = []
        for i in range(x.size(1)):
            out, hidden[0] = self.rnn[0](x[:, i], hidden[0])
            for j in range(1, self.num_layers):
                out, hidden[j] = self.rnn[j](out, hidden[j])
            outputs.append(out)
            if (i + 1) % detach_steps == 0:
                hidden = [h.detach() for h in hidden]
        out = torch.stack(outputs, dim=1)
        assert out.size() == (x.size(0), x.size(1), self.hidden_size)
        return out, hidden


class CustomRNNConfig(PretrainedConfig):
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
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.
    ```"""

    model_type = "custom_rnnv0"
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
        activation_function="standard",
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
        pos_emb=False,
        rnn_cell_class="LSTMCell",
        rnn_kwargs={},
        **kwargs,
    ):
        self.bias = bias
        self.pos_emb = pos_emb
        self.rnn_cell_class = rnn_cell_class
        self.rnn_kwargs = rnn_kwargs
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.activation_function = activation_function
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
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
        if '"' not in update_str:
            update_str = convert_string_format_to_json_like(update_str)
        update_str = "{" + update_str + "}"
        print(update_str)
        d = json.loads(update_str)
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")
            setattr(self, k, v)


class CustomRNNmodel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_emb = nn.Embedding(config.vocab_size, config.embed_size)
        if config.pos_emb:
            self.pos_emb = nn.Embedding(config.context_len, config.embed_size)
        self.drop = nn.Dropout(config.dropout)
        rnn_cell_class = getattr(sys.modules[__name__], config.rnn_cell_class)
        rnn_kwargs = config.rnn_kwargs
        rnn_kwargs["dropout"] = config.dropout
        if config.activation_function == "standard":
            if config.rnn_cell_class == "LSTMCell":
                rnn_kwargs["act_sigmoid"] = torch.sigmoid
                rnn_kwargs["act_tanh"] = torch.tanh
            else:
                rnn_kwargs["activation"] = torch.tanh
        elif config.activation_function == "softgrad_hardtanh":
            if config.rnn_cell_class == "LSTMCell":
                rnn_kwargs["act_sigmoid"] = SoftGradHardSigmoid()
                rnn_kwargs["act_tanh"] = SoftGradHardTanh()
            else:
                rnn_kwargs["activation"] = SoftGradHardTanh()
        else:
            raise ValueError(f"Unknown activation function: {config.activation_function}")
        self.rnn_model = RNN(
            config.embed_size,
            config.embed_size,
            config.num_layers,
            rnn_cell_class,
            **rnn_kwargs,
        )
        self.norm = nn.LayerNorm(config.embed_size, bias=config.bias, eps=config.layer_norm_epsilon)
        self.lin_head = nn.Linear(config.embed_size, config.vocab_size, bias=False)
        self.loss_function = ForCausalLMLoss
        self.apply(self._init_weights)

        # tie weights
        self.lin_head.weight = self.word_emb.weight
        self.step_count = 0

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
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
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))

    def forward(self, input_ids, labels=None, attention_mask=None, hidden=None):
        detach_steps = 100000
        if self.training:
            self.step_count += 1
            B, T = input_ids.size()

            max_length = min(T, int(self.step_count // 20 // 12) + 2)
            detach_steps = 50
            max_examples = T // max_length * max_length
            input_ids = input_ids[:, :max_examples]
            input_ids = rearrange(input_ids, "B (E T) -> (B E) T", T=max_length)
            labels = labels[:, :max_examples]
            labels = rearrange(labels, "B (E T) -> (B E) T", T=max_length)
            if attention_mask is not None:
                raise NotImplementedError("Attention mask not implemented")
                attention_mask = attention_mask[:, :max_examples]
                attention_mask = rearrange(attention_mask, "B (E T) -> (B E) T", T=max_length)
            if random.random() < 0.01:
                print("length", max_length)

        B, T = input_ids.size()
        if self.config.pos_emb:
            pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            features = self.word_emb(input_ids) + self.pos_emb(pos)
        else:
            features = self.word_emb(input_ids)
        features = self.drop(features)
        x, hidden = self.rnn_model(features, hidden=hidden, detach_steps=detach_steps)
        x = self.norm(x)
        logits = self.lin_head(x)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                # **kwargs,
            )
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits, hidden_states=hidden)

    def get_input_embeddings(self):
        return self.word_emb

    @torch.no_grad()
    def generate(self, idx, max_length, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        hidden = None
        for _ in range(max_length):
            # forward the model to get the logits for the index in the sequence
            out = self(idx, hidden=hidden)
            logits = out.logits
            hidden = out.hidden_states
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
            print(idx.shape, idx_next.shape)
            idx = idx_next
        return idx
