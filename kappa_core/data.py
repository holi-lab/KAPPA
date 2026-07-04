from __future__ import annotations
import os
from random import Random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import copy
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kappa_core.exp_config import LLMConfig, DatasetConfig, GenerationConfig

LOGGER = logging.getLogger(__name__)

class TemplateError(RuntimeError):
    """Raised when a prompt template or option type is missing."""


def _load_yaml(path: Path) -> Dict:
    """Load a YAML file exactly once and cache the result."""
    if not hasattr(_load_yaml, "_cache"):
        _load_yaml._cache = {}  # type: ignore[attr-defined]
    if path not in _load_yaml._cache:  # type: ignore[attr-defined]
        try:
            with path.open(encoding="utf-8") as fh:
                _load_yaml._cache[path] = yaml.safe_load(fh)  # type: ignore[attr-defined]
        except FileNotFoundError as exc:
            raise TemplateError(f"Cannot find YAML file: {path}") from exc
    return _load_yaml._cache[path]  # type: ignore[attr-defined]


def _load_json(path: Path) -> Dict:
    """Load a JSON file exactly once and cache the result."""
    if not hasattr(_load_json, "_cache"):
        _load_json._cache = {}  # type: ignore[attr-defined]
    if path not in _load_json._cache:  # type: ignore[attr-defined]
        try:
            with path.open(encoding="utf-8") as fh:
                _load_json._cache[path] = json.load(fh)  # type: ignore[attr-defined]
        except FileNotFoundError as exc:
            raise TemplateError(f"Cannot find JSON file: {path}") from exc
    return _load_json._cache[path]  # type: ignore[attr-defined]


class PromptTemplate:
    _INSTRUCTION_FILE = Path("prompts/instructions.yaml")
    _OPTION_FILE = Path("prompts/option_types.yaml")
    _DATASET_INFO_FILE = Path("prompts/dataset_info.json")

    def __init__(self, dataset_config:DatasetConfig, model_name: str, tokenizer:AutoTokenizer) -> None:
        self.model_name = model_name.lower()

        self.dataset_config = dataset_config

        instruction_cfg = _load_yaml(self._INSTRUCTION_FILE)
        option_cfg = _load_yaml(self._OPTION_FILE)
        self.task_prompts = _load_json(self._DATASET_INFO_FILE)
        if dataset_config.type == 'open_ended':
            self.task_prompts = _load_json(self._DATASET_INFO_FILE.parent / "dataset_info_open_ended.json")

        self.mcq_inst = instruction_cfg.get("instructions", {}).get(dataset_config.prompt_config.mcq_inst_version)
        self.answer_format = instruction_cfg.get("answer_format", {}).get(dataset_config.prompt_config.answer_format_version)
        self.option_symbols = option_cfg.get("option_symbol", {}).get(dataset_config.prompt_config.option_symbol)
        self.option_wrapper = option_cfg.get("option_wrapper", {}).get(dataset_config.prompt_config.option_wrapper)

        if not self.mcq_inst:
            raise TemplateError(f"Unknown instruction version: {self.mcq_inst}")
        if not self.answer_format:
            raise TemplateError(f"Unknown answer format version: {self.answer_format}")
        if not self.option_symbols:
            raise TemplateError(f"Unknown option symbol: {self.option_symbols}")
        if not self.option_wrapper:
            raise TemplateError(f"Unknown option wrapper: {self.option_wrapper}")
        
        self.answer_prefix, self.answer_suffix = self.answer_format.get("prefix", ""), self.answer_format.get("suffix", "")
        self.symbol_prefix, self.symbol_suffix = self.option_wrapper.get("prefix", ""), self.option_wrapper.get("suffix", "")
        self.option_symbols = self.option_symbols[:self.dataset_config.num_option]
        self.wrapped_symbols = [
            f"{self.option_wrapper['prefix']}{sym}{self.option_wrapper['suffix']}"
            for sym in self.option_symbols
        ]

        self.answer_formatted =[
            f"{self.answer_format['prefix']}{sym}"
            for sym in self.wrapped_symbols
        ]

        self.symbol_2_id = self.tokenize_option_symbols(tokenizer) 

        self.instruction = self._build_instruction(self.dataset_config.num_option)


    def tokenize_option_symbols(self, tokenizer: AutoTokenizer) -> None:
        self.symbol_2_id = {}
        for idx, (symbol, wrapped_symbol) in enumerate(zip(self.option_symbols, self.wrapped_symbols)):
            has_wrapper = self.option_wrapper['prefix'] and self.option_wrapper['suffix']
            
            if 'qwen' in self.model_name.lower():
                text_to_encode = " " + wrapped_symbol
            else:
                text_to_encode = wrapped_symbol
            
            symbol_ids = tokenizer.encode(text_to_encode, add_special_tokens=False)
            filtered_ids = list(symbol_ids)

            if has_wrapper:
                if self.option_wrapper['prefix'] and self.option_wrapper['suffix']:
                    if len(filtered_ids) > 0:
                        first_token = tokenizer.convert_ids_to_tokens(filtered_ids[0])
                        if self.option_wrapper['prefix'] in first_token:
                            filtered_ids.pop(0)
                    
                    if len(filtered_ids) > 0:
                        last_token = tokenizer.convert_ids_to_tokens(filtered_ids[-1])
                        if self.option_wrapper['suffix'] in last_token:
                            filtered_ids.pop(-1)

                    if len(filtered_ids) != 1:
                        dummy_ids = tokenizer.encode("." + symbol, add_special_tokens=False)
                        if len(dummy_ids) >= 2:
                            filtered_ids = [dummy_ids[-1]]

            if len(filtered_ids) < 1:
                filtered_ids = tokenizer.encode(symbol, add_special_tokens=False)
            
            if len(filtered_ids) == 0:
                raise ValueError(f"Target symbol {symbol} (wrapped: {wrapped_symbol}) could not be isolated to a single token. IDs: {symbol_ids}")
            
            self.symbol_2_id[symbol] = filtered_ids

        return self.symbol_2_id


    def _build_instruction(self, num_option: int) -> Tuple[str, str, str]:
        """Fill placeholders in the instruction template."""
        instruction = self.mcq_inst

        symbols_text = " or ".join(self.wrapped_symbols[:num_option])
        format_text = " or ".join(self.answer_formatted[:num_option])

        instruction_filled = (
            instruction.replace("{option_text}", symbols_text)
                       .replace("{format_text}", format_text)
        )

        if self.dataset_config.type == "open_ended":
            instruction_filled = 'Provide the answer clearly '
            instruction_filled = ''
        return instruction_filled

    # Public API -------------------------------------------------------------
    def make_prompt(self, item: Dict, dataset_name: str, is_positive: bool, num_option: int) -> Tuple[str, str]:
        """Return (user_input, model_output) strings for one data row."""

        messages = []

        instruction = self.instruction if self.instruction else self._build_instruction(num_option)
            
        question = f"{item['question_no_option']}"
        choices= "Choices:\n"
        try:
            choices += '\n'.join(
                [
                    f"{self.wrapped_symbols[i]}: {item['options'][i]}"
                    for i in range(num_option)
                ]
            )
        except Exception as e:
            LOGGER.warning("Error while building choices string: %s", e)
            return None, None, None, None, None, None

        user_input = f"{instruction}\n{question}"
        user_input += f"\n\n{choices}" if self.dataset_config.type != "open_ended" else ''

        messages.append({"role": "user", "content": user_input})

        task_prompt = self.task_prompts.get(dataset_name, {}).get("task_prompt", "")
        messages[0]["content"] = f"{task_prompt}\n\n{messages[0]['content']}" if task_prompt else messages[0]['content']

        target = item.get('answer', 0) if is_positive else 1 - item.get('answer', 0)
        target_option = item.get('options', [])[target] if item.get('options', []) else None

        target_symbol = self.option_symbols[target] if self.option_symbols else None
        self.target_symbol = target_symbol
        target_symbol_wrapped = self.wrapped_symbols[target] if self.wrapped_symbols else None

        gold_output = self.answer_formatted[target] if self.answer_formatted else None

        messages.append({"role": "assistant", "content": gold_output})

        return messages, gold_output, target, target_option, target_symbol, target_symbol_wrapped
    
    @property
    def system_prompt(self) -> str | None:
        return ''

    def apply_chat_template(self, message: List[Dict[str, str]], reference, add_special_tokens=True):
        input_context = ''

        if self.model_name == 'llama-2' or ('llama' in self.model_name and '2' in self.model_name):
            if add_special_tokens:
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                B_SES, E_SES = "<s>", "</s>"  
            else:
                B_INST, E_INST = "", ""
                B_SYS, E_SYS = "", ""
                B_SES, E_SES = "", ""
            
            if message[0]["role"] == "system":
                message = [
                    {
                        "role": message[1]["role"],
                        "content": B_SYS + message[0]["content"] + E_SYS + message[1]["content"]\
                                if message[0]['content'] != '' else message[1]["content"],
                    }
                ] + message[2:]

            for prompt, answer in zip(message[:-1:2], message[1:-1:2]):
                input_context += f"{B_SES} {B_INST} {(prompt['content'])} {E_INST} {(answer['content'])} {E_SES}"
            input_context += f"{B_SES} {B_INST} {(message[-1]['content'])} {E_INST} "

            input_output_context = input_context + reference 

        elif self.model_name == 'llama-3' or ('llama' in self.model_name and '3' in self.model_name):

            if add_special_tokens:
                BOT, EOTURN = "<|begin_of_text|>", "<|eot_id|>"
                BH, EH = "<|start_header_id|>", "<|end_header_id|>"
            else:
                BOT, EOTURN = "", ""
                BH, EH = "", ""

            input_context += BOT
            for msg in message:
                input_context += BH
                input_context += msg['role'] if add_special_tokens else ''
                input_context += EH + '\n\n'
                input_context += msg['content'].strip()
                input_context += EOTURN
            
            input_context += BH
            input_context += 'assistant' if add_special_tokens else ''
            input_context += EH + '\n\n'

            input_output_context = input_context + reference 


        elif 'qwen' in self.model_name:
            if add_special_tokens:
                START, END = "<|im_start|>", "<|im_end|>"
            else:
                START, END = "", ""

            for msg in message:
                input_context += f"{START}{msg['role']}\n{msg['content']}{END}\n"

            input_context += START + "assistant\n"
            input_output_context = input_context + reference
        
        else:
            for msg in message:
                input_context += f"{msg['role']}: {msg['content']}\n"
            input_context += "assistant: "
            input_output_context = input_context + reference
    
        return input_context, input_output_context


class ComparisonDataset(Dataset):
    """torch.utils.data.Dataset that yields fully-tokenised comparison prompts."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizerBase,
        llm_cfg: LLMConfig,
        dataset_cfg: DatasetConfig,
        generation_cfg: GenerationConfig,
        template_manager: PromptTemplate,
        choice_only: bool = False,
    ) -> None:
        try:
            with Path(data_path).open(encoding="utf-8") as fh:
                self._data: List[Dict] = json.load(fh)
                self._original_data = self._data.copy()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Cannot find data file: {data_path}") from exc

        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm_cfg: LLMConfig = llm_cfg
        self.dataset_cfg: DatasetConfig = dataset_cfg
        self.generation_cfg: GenerationConfig = generation_cfg

        self.template_manager = template_manager
        self.use_chat: bool = not llm_cfg.use_base_model
        self.dataset_name = dataset_cfg.dataset
        self.decoding_mode = generation_cfg.decoding_mode

        self.choice_only = choice_only

        self.preprocess()

    def preprocess(self,) -> None:
        self.data = [] 
        tmp = {}
        for idx in range(len(self._data)):
            item = self._data[idx]
            tmp = copy.deepcopy(item)

            permutations = item.get("permutations", [])
            if not permutations:
                permutations = [{"options": item["options"], "answer": item["answer"]}]

            if isinstance(self.dataset_cfg.num_permutations, list):
                permutations = [permutations[perm] for perm in self.dataset_cfg.num_permutations]
            else:
                rng=Random(self.dataset_cfg.random_seed)
                perm_idx = list(range(len(permutations)))
                rng.shuffle(perm_idx)
                permutations = [permutations[i] for i in perm_idx[:self.dataset_cfg.num_permutations]]

            for perm_idx, permutation in enumerate(permutations):
                tmp['options'] = permutation['options']
                tmp['answer'] = permutation['answer']

                if self.choice_only:
                    tmp["question_no_option"] = ' '

                messages, model_output, target, target_option, target_symbol, target_symbol_wrapped =\
                    self.template_manager.make_prompt(
                        tmp, 
                        self.dataset_name, 
                        self.generation_cfg.is_positive,
                        self.dataset_cfg.num_option
                    )

                if messages is None:
                    continue

                if "open_generate" in self.decoding_mode:
                    model_output = ""

                if "logit_generate" in self.decoding_mode:
                    model_output = model_output.split(f"{self.template_manager.target_symbol}{self.template_manager.symbol_suffix}")[0]
                
                model_output = " "+model_output if model_output else ""

                self.data.append({
                    "item_idx": idx,
                    "perm_idx": perm_idx,
                    "question_no_option": tmp['question_no_option'],
                    "options": tmp['options'],
                    "answer": tmp['answer'],
                    "messages": messages,
                    "model_output": model_output,
                    "target": target,
                    "target_symbol": target_symbol,
                    "target_symbol_wrapped": target_symbol_wrapped,
                    "target_option": target_option,
                })

    # Dataset protocol 
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.data[idx]        
        tokens = self._tokenise(item, self.template_manager.system_prompt)
        
        return tokens.unsqueeze(0)

    # Internal helpers 
    def _tokenise(
        self,
        item: Dict,
        system_prompt: str | None,
    ) -> torch.Tensor:
        """Return a single tensor of token ids representing the full conversation."""
        messages : list = item.get('messages', '')
        model_output = item.get('model_output', '')
        if self.use_chat:
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
    
            messages.pop(-1)

            rendered = self.template_manager.apply_chat_template(
                message=messages, reference=model_output, add_special_tokens=True
            )

            item['model_input'] = rendered[1]
            return self.tokenizer(rendered[1], return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        
