from __future__ import annotations

from pathlib import Path
import os, sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import logging
from typing import Iterable

import torch
from tqdm.auto import tqdm

from kappa_core.llm_wrapper import LlmWrapper
from kappa_core.data import ComparisonDataset, PromptTemplate
from kappa_core.exp_config import ActivationConfig, ExperimentConfig, DatasetConfig, GenerationConfig
from kappa_core.data_path import PathManager
from kappa_core.collector import ActivationCollector
from kappa_core.utils.helpers import find_instruction_end_postion

from itertools import product

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("activation_runner")
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# --------------------------------------------------------------------------- #
# Experiment runner
# --------------------------------------------------------------------------- #
class ActivationExperimentRunner:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.activation_config:ActivationConfig = cfg.activation_config

        LOGGER.info("Using LlmWrapper for model operations.")
        self.model = LlmWrapper(
            model_name=cfg.llm_config.model_name,
            size=cfg.llm_config.model_size,
            use_chat=not cfg.llm_config.use_base_model,
            hf_token=HUGGINGFACE_TOKEN,
        )
        self.model.set_save_internal_decodings(False)

        self.path_manager: PathManager = PathManager(cfg)
        self.template_manager = None
        self.symbol_2_id = None

    def _is_experiment_complete(
        self,
        dataset_config: DatasetConfig,
        generation_config: GenerationConfig
    ) -> bool:
        """
        Return True if metadata.jsonl exists for this (dataset_config, generation_config)
        and its non-blank line count equals the dataset length for that configuration.
        Gracefully handle missing files (return False).
        """
        self.path_manager.setup_configs(
            dataset_config=dataset_config,
            generation_config=generation_config
        )

        metadata_path_str = self.path_manager.get_activation_output(output_type="metadata")
        metadata_path = Path(metadata_path_str)

        if not metadata_path.is_file():
            LOGGER.debug(f"[RESUME] Check: Metadata file not found at {metadata_path}. Experiment is not complete.")
            return False

        if metadata_path.exists():
            LOGGER.debug(f"[RESUME] Check: Metadata file found at {metadata_path}. Experiment is complete. Skip this config.")
            return True
        return False

    def _process_sample(
        self, 
        item:dict, 
        input_ids: torch.Tensor, 
        dataset_config: DatasetConfig,
        activation_config: ActivationConfig,
        generation_config:GenerationConfig
    ) -> None:
        with torch.no_grad():
            input_ids = input_ids.to(self.model.device)

            if "teacher_forced" in generation_config.decoding_mode:
                total_token_ids = input_ids

                # add teacher-forced target as 'pred'
                item["pred"] = item["target"]
                item["pred_option"] = item["target_option"]
                item["pred_symbol"] = item["target_symbol"]
                item["score"] = int(item["pred"] == item["answer"])

            elif "open_gen" in generation_config.decoding_mode:
                model_output = self.model.generate(
                    tokens=input_ids, max_new_tokens=100
                )
                total_token_ids = self.model.tokenizer.encode(
                    model_output, return_tensors="pt", add_special_tokens=False
                ).to(self.model.device)

                instr_end_pos = find_instruction_end_postion(total_token_ids[0], self.model.END_STR)
                pred_output = self.model.tokenizer.decode(
                    total_token_ids[0][instr_end_pos + 1:], 
                    skip_special_tokens=True
                ).strip()

                candidates = self.template_manager.answer_formatted
                matches = [
                    (i, sym) for i, (sym, cand) in enumerate(zip(self.template_manager.option_symbols, candidates)) 
                    if cand in pred_output
                ]
                if len(matches) == 1:
                    item['pred'], item['pred_symbol'] = matches[0]
                    item['pred_output'] = pred_output
                    item['pred_option'] = item['options'][item['pred']]
                    item['score'] = int(item['pred'] == item["answer"])
                else:
                    item['pred'] = None
                    item['pred_option'] = None
                    item['pred_symbol'] = None
                    item['pred_output'] = pred_output
                    item['score'] = 0
                    
            elif "logit_gen" in generation_config.decoding_mode:
                logits = self.model.get_logits(
                    input_ids
                )
                symbol_candidates = self.template_manager.option_symbols[:dataset_config.num_option]

                token_ids = [self.symbol_2_id[symbol] for symbol in symbol_candidates]

                last_token_logits = logits[:, -1, :].squeeze(0)  # (vocab_size,)
                
                # select the token with the highest logit
                token_logits = []

                for token_id in token_ids:
                    logit = last_token_logits[token_id].detach().cpu().item()
                    token_logits.append(logit)

                max_index = int(torch.argmax(torch.tensor(token_logits)))
                pred_token_id = token_ids[max_index]

                prediction = self.model.tokenizer.decode(
                    pred_token_id, skip_special_tokens=True
                ).strip()

                item["logits"] = token_logits 
                item['pred'] = symbol_candidates.index(prediction)
                item['pred_option'] = item['options'][item['pred']]
                item['pred_symbol'] = prediction
                item['score'] = int(item['pred'] == item["answer"])
                
                pred_token_id = torch.tensor([pred_token_id], device=input_ids.device)
                total_token_ids = torch.cat((input_ids, pred_token_id), dim=1)

            else:
                total_token_ids = input_ids

            instr_end_pos = find_instruction_end_postion(total_token_ids[0], self.model.END_STR)
            item["instruction_end_position"] = instr_end_pos
            item["model_output"] = self.model.tokenizer.decode(total_token_ids[0]).strip()

            logits = self.model.get_logits(total_token_ids)
            
            seq_len = total_token_ids.shape[1]
            target_tokens = self.model.tokenizer.convert_ids_to_tokens(total_token_ids[0])
            
            item['token_ids'] = total_token_ids.tolist()[0] 
            item['tokens'] = target_tokens

            self.collector.start_sample(seq_len)

            for layer in activation_config.layers:
                activation_tuple = self.model.get_last_activations_many(layer)  # (1, seq_len, dim)
                self.collector.add(
                    activation_tuple=activation_tuple, layer_idx=layer
                )

        self.collector.finish_sample(item)

    def _run_dataset(
        self, 
        dataset_config: DatasetConfig, 
        generation_config: GenerationConfig
    ) -> None:
        # Setup the path manager
        self.path_manager.setup_configs(
            dataset_config=dataset_config,
            generation_config=generation_config
        )
        if self._is_experiment_complete(dataset_config, generation_config):
            LOGGER.info(
                "[RESUME] Skipping completed experiment for dataset='%s' polarity='%s'. Results found.",
                dataset_config.dataset, getattr(generation_config, "polarity", "N/A"),
            )
            return

        # Setup the prompt template manager
        self.collector: ActivationCollector = ActivationCollector(self.cfg.activation_config)
        self.template_manager = PromptTemplate(
            dataset_config=dataset_config,
            model_name=self.cfg.llm_config.model_name,
            tokenizer=self.model.tokenizer
        )
        self.symbol_2_id = self.template_manager.symbol_2_id
        
        # Load the dataset
        data_path = self.path_manager._dataset_file()
        dset = ComparisonDataset(
            data_path=str(data_path),
            tokenizer=self.model.tokenizer,
            llm_cfg=self.cfg.llm_config,
            dataset_cfg=dataset_config,
            generation_cfg=generation_config,
            template_manager=self.template_manager,
            choice_only=True if self.cfg.experiment_type == 'analysis_control' else False
        )
        LOGGER.info(f"Dataset {dataset_config.dataset} — {len(dset)} samples")

        for sid, token_ids in tqdm(enumerate(dset), total=len(dset), desc=dataset_config.dataset):
            item = dset.data[sid]
            self._process_sample(
                item=item, input_ids=token_ids, 
                dataset_config=dataset_config, 
                activation_config=self.activation_config,
                generation_config=generation_config
            )

        # Save the activations and metadata
        self.collector.flush_to_disk(
            self.path_manager.get_activation_output
        )
        LOGGER.info(
            f"""Saved activations of dataset {dataset_config.dataset} 
            for token positions {self.activation_config.token_positions} to {self.path_manager.path}"""
        )

        # Save the config
        config_path = self.path_manager.get_activation_output(output_type="configs")
        with open(config_path, "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2, default=str)
        LOGGER.info(f"Config saved to {config_path}")

        torch.cuda.empty_cache()

    def run_all(self) -> None:
        num_exp = len(self.cfg.dataset_config_list) * len(self.cfg.generation_config_list)

        for idx, (dataset_config, generation_config) in enumerate(
            product(
                self.cfg.dataset_config_list,
                self.cfg.generation_config_list
            )
        ):
            msg = "========================================================"
            msg += f"\nRunning experiment {idx + 1}/{num_exp}:\n"
            msg += f"==== Dataset: {dataset_config.dataset}, Split: {dataset_config.split}, "
            msg += f"mcq_inst_version: {dataset_config.prompt_config.mcq_inst_version}, "
            msg += f"answer_format_version: {dataset_config.prompt_config.answer_format_version}, "
            msg += f"option_symbol: {dataset_config.prompt_config.option_symbol}, " 
            msg += f"option_wrapper: {dataset_config.prompt_config.option_wrapper} ===="

            msg += f"\n==== Decoding mode: {generation_config.decoding_mode}, "
            msg += f"polarity: {generation_config.polarity}, "
            LOGGER.info(msg)
 
            generation_config.is_positive = (generation_config.polarity == 'pos')
            LOGGER.info(f"==== Positive ===") if generation_config.is_positive else LOGGER.info(f"==== Negative ===")
            self._run_dataset(
                dataset_config=dataset_config,
                generation_config=generation_config
            )

        LOGGER.info("All runs complete.")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_cli(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect hidden activations from a transformer.")
    p.add_argument("--config",
                   type=str,
                   help="Path to optional JSON/YAML file with the same keys.")
    return p.parse_args(argv)

def main() -> None:
    args = parse_cli()
    cfg = ExperimentConfig.from_sources(vars(args), cfg_path=args.config)
    LOGGER.info("ExperimentConfig:\n%s", json.dumps(cfg.to_dict(), indent=2, default=str))
    ActivationExperimentRunner(cfg).run_all()

if __name__ == "__main__":
    main()
