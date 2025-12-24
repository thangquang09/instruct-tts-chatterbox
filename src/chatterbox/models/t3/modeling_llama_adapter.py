import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaModel as HFLlamaModel,
    LlamaDecoderLayer as HFLlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.utils import (
    logging,
)

logger = logging.get_logger(__name__)

class AdaRMSNormAdapter(nn.Module):
    def __init__(self, hidden_size: int, instruction_dim: int):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(instruction_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2) # cho gamma vs beta
        )
        
        # Zero Initialization (zeros init last layer)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        
    def forward(
        self, 
        x: torch.Tensor,
        instruction_emb: torch.Tensor,
        original_norm_layer: nn.Module 
    ):
        """
        x: Hidden states [Batch, Seq, Dim],
        instruction_emb: Instruction Vector [Batch, Instr_Dim],
        original_norm_layer: Backbone's freezed norm layer
        """
        
        normed_x = original_norm_layer(x) # (x / RMS(x)) * W_old
        style_params = self.adapter(instruction_emb) # [Batch, Dim * 2]
        style_params = style_params.unsqueeze(1)
        
        gamma_ada, beta_ada = style_params.chunk(2, dim=-1)
        
        output = normed_x * (1 + gamma_ada) + beta_ada
        return output

class CustomLlamaDecoderLayer(HFLlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, adapter_config: dict):
        super().__init__(config, layer_idx)
        
        instr_dim = adapter_config.get("instruction_dim", 1024)
        self.input_adapter = AdaRMSNormAdapter(config.hidden_size, instr_dim)
        self.post_attention_adapter = AdaRMSNormAdapter(config.hidden_size, instr_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        instruction_emb: torch.Tensor = None, # <--- [NEW INPUT]
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        # Thay input_layernorm bằng input_adapter ---
        hidden_states = self.input_adapter(hidden_states, instruction_emb, self.input_layernorm)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        
        # Thay post_attention_layernorm bằng post_attention_adapter ---
        hidden_states = self.post_attention_adapter(hidden_states, instruction_emb, self.post_attention_layernorm)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    
class CustomLlamaModel(HFLlamaModel):
    def __init__(self, config: LlamaConfig, adapter_config: dict):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        self.layers = nn.ModuleList(
            [
                CustomLlamaDecoderLayer(config, layer_idx, adapter_config) 
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        instruction_emb: Optional[torch.Tensor] = None, # <--- [NEW INPUT]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated..."
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        
        hidden_states = inputs_embeds

        # Rotary Pos Emb
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Containers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Lưu ý: Gradient Checkpointing cần xử lý thêm nếu muốn truyền instruction_emb.
                # Tạm thời gọi trực tiếp nếu ko dùng checkpointing, hoặc dùng partial nếu cần.
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    instruction_emb, # <--- [PASS]
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    instruction_emb=instruction_emb, # <--- [PASS]
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )