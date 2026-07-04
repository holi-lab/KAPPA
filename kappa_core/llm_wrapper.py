import torch as t
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Optional
from pathlib import Path

from kappa_core.utils.helpers import add_vector_from_position, find_instruction_end_postion, get_model_path
from kappa_core.utils.tokenize import get_special_tokens

LOGGER = logging.getLogger(__name__)

class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()            
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.attn_activations = None
        self.mlp_activations = None

        self.add_activations = None
        self.from_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

        self.steering_module = None
        self.multiplier = 1.0

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped block"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        if isinstance(output, tuple):
            hidden, *rest = output
            out_kind = "tuple"
        elif isinstance(output, list):
            hidden, *rest = output
            out_kind = "list"
        else:
            hidden, rest, out_kind = output, [], "tensor"

        self.activations = hidden

        if self.steering_module is not None:
            vector = self.multiplier * self.steering_module(hidden)
            augmented_hidden = add_vector_from_position(
                matrix=hidden,
                vector=vector,
                position_ids=kwargs["position_ids"],
                from_pos=self.from_position,
            )
            if out_kind == "tuple":
                output = (augmented_hidden, *rest)
            elif out_kind == "list":
                output = [augmented_hidden, *rest]
            else:
                output = output.clone()
                output[0] = augmented_hidden
 
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        attn_output = self.block.self_attn.activations
        self.attn_activations = attn_output
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_activations = mlp_output
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        self.layer_activations = args[0] + output[0]

        return output

    def add(self, activations):
        self.add_activations = activations
    
    def set_steering(self, steering_module, multiplier=1.0):
        self.steering_module = steering_module
        self.multiplier = multiplier

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.attn_activations = None
        self.mlp_activations = None
        self.layer_activations = None
        self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []

        self.steering_module = None
        self.multiplier = 1.0
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

class LlmWrapper:
    def __init__(
        self,
        hf_token: str,
        size: str = "7b",
        use_chat: bool = True,
        override_model_weights_path: Optional[str] = None,
        model_name: str = "Llama-2",
        torch_dtype: Optional[t.dtype] = None,
        adapter_path: Path = None
    ):
        self.device = "cuda:0" if t.cuda.is_available() else "cpu"
        self.use_chat = use_chat
        self.model_name_path = get_model_path(size, not use_chat, model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, token=hf_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path, token=hf_token, device_map=self.device,
            torch_dtype=t.bfloat16
        )
        self.adapter_path = adapter_path
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))

        if size in ["8B","13b", "70b", "70B"]:
            self.model = self.model.half()
        
        special_token = get_special_tokens(self.model_name_path)["ADD_FROM_POS_BASE"]
        if use_chat:
            special_token = get_special_tokens(self.model_name_path)["ADD_FROM_POS_CHAT"]
        
        self.END_STR = t.tensor(self.tokenizer.encode(special_token)[1:]).to(
            self.device
        )

        self._wrap_layers()

    def _wrap_layers(self):
        if self.adapter_path is None:
            for i, layer in enumerate(self.model.model.layers):
                self.model.model.layers[i] = BlockOutputWrapper(
                    layer, self.model.lm_head, self.model.model.norm, self.tokenizer
                )
        else:
            for i, layer in enumerate(self.model.base_model.model.model.layers):
                self.model.base_model.model.model.layers[i] = BlockOutputWrapper(
                    layer, self.model.base_model.model.lm_head, self.model.base_model.model.model.norm, self.tokenizer
                )
    
    def set_save_internal_decodings(self, value: bool):
        if self.adapter_path is None:
            for layer in self.model.model.layers:
                layer.save_internal_decodings = value
        else:
            for layer in self.model.base_model.model.model.layers:
                layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        if self.adapter_path is None:
            for layer in self.model.model.layers:
                layer.from_position = pos
        else:
            for layer in self.model.base_model.model.model.layers:
                layer.from_position = pos

    def generate(self, input_ids, attention_mask, max_new_tokens=100, **kwargs):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(input_ids[0], self.END_STR)
            self.set_from_positions(instr_pos)
            generated = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, top_k=1, **kwargs
            )
            return self.tokenizer.batch_decode(generated)[0] if kwargs is None else generated

    def get_logits(self, tokens):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            logits = self.model(tokens).logits
            return logits
    
    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations
    
    def get_last_activations_many(self, layer):
        if self.adapter_path is None:
            return (
                self.model.model.layers[layer].activations,
                self.model.model.layers[layer].attn_activations,
                self.model.model.layers[layer].mlp_activations,
                self.model.model.layers[layer].layer_activations
            )
        else:
            return (
                self.model.base_model.model.model.layers[layer].activations,
                self.model.base_model.model.model.layers[layer].attn_activations,
                self.model.base_model.model.model.layers[layer].mlp_activations,
                self.model.base_model.model.model.layers[layer].layer_activations
            )
    
    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations)
    
    def set_steering(self, layer, steering_module, multiplier=1.0):
        self.model.model.layers[layer].set_steering(steering_module, multiplier)
    
    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()
