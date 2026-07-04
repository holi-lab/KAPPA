import torch as t

def add_vector_from_position(matrix, vector, position_ids, from_pos=None):
    from_id = from_pos
    if from_id is None:
        from_id = position_ids.min().item() - 1

    mask = position_ids >= from_id
    matrix += mask.float().T * vector
    return matrix


def find_last_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if t.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1

def get_model_path(size: str, is_base: bool, model: str = "Llama-2", path: str = None):
    if path:
        return path

    if model == "Llama-2":
        if is_base:
            return f"meta-llama/Llama-2-{size.lower()}-hf"
        else:
            return f"meta-llama/Llama-2-{size.lower()}-chat-hf"
    elif model == "Llama-3":
        if is_base:
            return f"meta-llama/Meta-Llama-3-{size.upper()}"
        else:
            return f"meta-llama/Meta-Llama-3-{size.upper()}-Instruct"
    elif model == "Llama-3.1":
        if is_base:
            return f"meta-llama/Llama-3.1-{size.upper()}"
        else:
            return f"meta-llama/Llama-3.1-{size.upper()}-Instruct"
    elif model == "Llama-3.2":
        if is_base:
            return f"meta-llama/Llama-3.2-{size.upper()}"
        else:
            return f"meta-llama/Llama-3.2-{size.upper()}-Instruct"
        
    elif "gemma" in model.lower():
        if "gemma3" in model.lower():
            if "12b" in model.lower() or size.lower() == '12b':
                return "google/gemma-3-12b-it"
            
    elif "mistral" in model.lower():
        if "7b" in model.lower() or size.lower() == '7b':
            return "mistralai/Mistral-7B-Instruct-v0.3"

    elif "qwen" in model.lower():
        if "qwen2.5" in model.lower():
            return 'Qwen/Qwen2.5-7B-Instruct'
        elif "qwen3" in model.lower():
            if "32b" in model.lower() or size.lower() == '32b':
                return f'Qwen/Qwen3-{ size.upper() or "32B"}'
            elif '4b' in model.lower() or size.lower() == '4b':
                return f"Qwen/Qwen3-{size.upper() or '4B'}"