# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_lstm import LSTMConfig

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


logger = logging.get_logger(__name__)


class LSTMPreTrainedModel(PreTrainedModel):

    config_class = LSTMConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = False
    _no_split_modules = []

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)


class LSTMModel(LSTMPreTrainedModel):

    def __init__(self, config: LSTMConfig) -> LSTMModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Standard multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            batch_first=True,
            dropout=config.dropout if config.num_hidden_layers > 1 else 0.0,
        )

        # Optional residual post-LSTM projection to match dims (kept identity when same size)
        self.use_residual = bool(getattr(config, 'residual_connection', False))
        self.residual_scale = float(getattr(config, 'residual_scale', 1.0))
        self.use_residual_layernorm = bool(getattr(config, 'use_residual_layernorm', False))
        if self.use_residual_layernorm:
            self.residual_ln = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'norm_eps', 1e-6), elementwise_affine=getattr(config, 'elementwise_affine', True))

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # LSTM forward
        if past_key_values is not None:
            lstm_out, (h_n, c_n) = self.lstm(inputs_embeds, past_key_values)
        else:
            lstm_out, (h_n, c_n) = self.lstm(inputs_embeds)

        # Optional residual connection: add input embeddings to LSTM output
        if self.use_residual:
            # Ensure shapes are compatible; both are (batch, seq, hidden)
            residual = inputs_embeds
            lstm_out = lstm_out + self.residual_scale * residual
            if self.use_residual_layernorm:
                lstm_out = self.residual_ln(lstm_out)

        # Collect hidden states if requested
        all_hidden_states = None
        if output_hidden_states:
            # For standard LSTM, we only have input embeddings and final output
            all_hidden_states = (inputs_embeds, lstm_out)

        next_cache = (h_n, c_n) if use_cache else None

        if not return_dict:
            return tuple(v for v in [lstm_out, next_cache, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=lstm_out,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
        )


class LSTMForCausalLM(LSTMPreTrainedModel):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LSTMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            # We want to predict the next token, so shift labels left
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
