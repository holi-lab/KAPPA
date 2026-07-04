from typing import List
from transformers import PreTrainedTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"

ADD_FROM_POS_CHAT = E_INST
ADD_FROM_POS_BASE = BASE_RESPONSE


def get_special_tokens(model_name: str) -> dict:
    if "llama-2" in model_name.lower():
        return {
            "B_INST": "[INST]",
            "E_INST": "[/INST]",
            "B_SYS": "<<SYS>>\n",
            "E_SYS": "\n<</SYS>>\n\n",
            "BASE_INPUT": "Input:",
            "BASE_RESPONSE": "\nResponse:",
            "ADD_FROM_POS_CHAT": "[/INST]",
            "ADD_FROM_POS_BASE": "\nResponse:",
        }
    elif 'llama-3' in model_name.lower():
        return {
            "BOT": "<|begin_of_text|>", 
            "EOT": "<|end_of_text|>",
            "EOT_ID": "<|eot_id|>",
            "START_HEADER_ID": "<|start_header_id|>",
            "END_HEADER_ID": "<|end_header_id|>",
            "ADD_FROM_POS_CHAT": "<|start_header_id|>assistant<|end_header_id|>",
            "ADD_FROM_POS_BASE": "\nResponse:",
        }
    elif 'qwen' in model_name.lower():
        return {
            "ADD_FROM_POS_BASE" :"<|im_end|>",
            "ADD_FROM_POS_CHAT" :"<|im_end|>"
        }
    elif 'mistral' in model_name.lower():
        return {
            "ADD_FROM_POS_BASE" :"[/INST] assistant message</s> [INST]",
            "ADD_FROM_POS_CHAT" :"[/INST] assistant message</s> [INST]"
        }
    elif 'gemma' in model_name.lower():
        return {
            "ADD_FROM_POS_BASE" :"<start_of_turn>model",
            "ADD_FROM_POS_CHAT" :"<start_of_turn>model"
        }



def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    input_prompt = []
    if system_prompt is not None:
        input_prompt.append({"role": "system", "content": system_prompt.strip()})
    
    input_prompt.append({"role": "user", "content": user_input.strip()})

    if model_output is not None:
        input_prompt.append({"role": "assistant", "content": model_output.strip()})

    return tokenizer.apply_chat_template(input_prompt)[:-2] if '2' in tokenizer.name_or_path else tokenizer.apply_chat_template(input_prompt)[:-1] 

def tokenize_llama_base(
    tokenizer, user_input: str, model_output: str = None
) -> List[int]:
    input_content = ""
    input_content += f"{BASE_INPUT} {user_input.strip()}"
    if model_output is not None:
        input_content += f"{BASE_RESPONSE} {model_output.strip()}"
    return tokenizer.encode(input_content)
