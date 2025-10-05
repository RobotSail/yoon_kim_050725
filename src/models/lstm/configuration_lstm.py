# -*- coding: utf-8 -*-

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class LSTMConfig(PretrainedConfig):

    model_type = 'lstm'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        # residuals
        residual_connection: bool = False,
        residual_scale: float = 1.0,
        use_residual_layernorm: bool = False,
        # attention (optional)
        use_self_attention: bool = False,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.0,
        use_attention_layernorm: bool = True,
        attn_residual_scale: float = 1.0,
        # norms
        norm_eps: float = 1e-6,
        elementwise_affine: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        vocab_size: int = 32000,
        proj_size: Optional[int] = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.proj_size = proj_size
        # residuals
        self.residual_connection = residual_connection
        self.residual_scale = residual_scale
        self.use_residual_layernorm = use_residual_layernorm
        # attention
        self.use_self_attention = use_self_attention
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.use_attention_layernorm = use_attention_layernorm
        self.attn_residual_scale = attn_residual_scale
        # norms
        self.norm_eps = norm_eps
        self.elementwise_affine = elementwise_affine

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
