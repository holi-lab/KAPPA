from __future__ import annotations
import os, sys
from itertools import product

import copy
import argparse
import json
import logging
from typing import Dict, Iterable, Literal
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from tqdm.auto import tqdm

from kappa_core.llm_wrapper import LlmWrapper
from kappa_core.data import ComparisonDataset, PromptTemplate
from kappa_core.exp_config import DatasetConfig, ExperimentConfig, GenerationConfig, SteeringConfig
from kappa_core.data_path import PathManager
from kappa_core.collector import ActivationCollector
from run_probe import ProbeExperimentRunner
from kappa_core.steering.model import KAPPAModule
from run_activation_collection import ActivationExperimentRunner
from kappa_core.module_generator import SteeringModuleGenerator


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("steering_runner")
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

Split = Literal["train", "validation", "test"]
ActivationDict = Dict[str, Dict[int, Dict[int, torch.Tensor]]]

# --------------------------------------------------------------------------- #
# Experiment runner
# --------------------------------------------------------------------------- #
class SteeringExperimentRunner(ProbeExperimentRunner, ActivationExperimentRunner):
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

        super().__init__(cfg)

        self.activation_config = self.cfg.activation_config
        self.collector: ActivationCollector = ActivationCollector(self.activation_config)

        self.path_manager: PathManager = PathManager(self.cfg)

        self.steering_module_generator: SteeringModuleGenerator = SteeringModuleGenerator(
            cfg=self.cfg,
            activation_exp_config=self.cfg.activation_exp_config,
            probe_exp_config=self.cfg.probe_exp_config
        )
    
        """ Initialize the model based on the configuration.
        """
        self.model = LlmWrapper(
            model_name=cfg.llm_config.model_name,
            size=cfg.llm_config.model_size,
            use_chat=not cfg.llm_config.use_base_model,
            hf_token=HUGGINGFACE_TOKEN,
        )
        self.model.set_save_internal_decodings(False)

        self.result = None

    def _is_steering_experiment_complete(
        self,
        steering_config: SteeringConfig,
        dataset_config: DatasetConfig,
        generation_config: GenerationConfig
    ) -> bool:
        """
        Checks if a steering experiment is complete by verifying the existence of the metadata output file.
        """
        self.path_manager.setup_configs(
            steering_config=steering_config,
            dataset_config=dataset_config,
            generation_config=generation_config
        )
        metadata_path_str = self.path_manager.get_steering_output(output_type="metadata")
        metadata_path = Path(metadata_path_str)

        if not metadata_path.is_file():
            LOGGER.debug(f"[RESUME] Check: Steering metadata not found at {metadata_path}. Experiment not complete.")
            return False

        LOGGER.debug(f"[RESUME] Check: Found steering metadata at {metadata_path}. Experiment is complete.")
        return True

    def _set_steering(
        self, 
        steering_config: SteeringConfig
    ) -> None:
        # Get the steering vectors based on the steering configuration
        for layer in steering_config.intervention_layers.layers:
            steering_module = self.steering_module_generator.make_steering_module(
                layer, 
                steering_module_class=KAPPAModule, 
                steering_config=steering_config
            )            
            if "qwen" in self.model.model_name_path:
                steering_module = steering_module.to(torch.bfloat16)
            self.model.set_steering(layer, steering_module, steering_config.multiplier)

    def _run_dataset(
        self,
        steering_config: SteeringConfig,
        dataset_config: DatasetConfig,
        generation_config:GenerationConfig
    ) -> None:
        self.path_manager.setup_configs(
            steering_config=steering_config,
            dataset_config=dataset_config,
            generation_config=generation_config
        )
        if self._is_steering_experiment_complete(steering_config, dataset_config, generation_config):
            LOGGER.info(
                "[RESUME] Skipping completed steering experiment for dataset='%s' with steering method '%s'.",
                dataset_config.dataset, steering_config.method
            )
            return

        # Setup the prompt template manager
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
            template_manager=self.template_manager
        )
        LOGGER.info(f"Dataset {dataset_config.dataset} — {len(dset)} samples")

        self.result = copy.deepcopy(dset._original_data)
        
        # Setup steering module
        self._set_steering(steering_config=steering_config)

        LOGGER.info(f"Processing layer {steering_config.intervention_layers.layers} for dataset {dataset_config.dataset}")

        for sid, token_ids in tqdm(enumerate(dset), total=len(dset), desc=dataset_config.dataset):
            item = dset.data[sid]
            self._process_sample(
                item=item, 
                input_ids=token_ids, 
                dataset_config=dataset_config, 
                activation_config=self.activation_config,
                generation_config=generation_config
            )
        
        self.model.reset_all()

        # Save the activations and metadata
        self.collector.flush_to_disk(
            self.path_manager.get_steering_output, save_activation=steering_config.save_activation
        )
        LOGGER.info(
            f"""Saved activations of dataset {dataset_config.dataset} 
            for token positions {self.activation_config.token_positions} to {self.path_manager.path}"""
        )

        # Save the config
        config_path = self.path_manager.get_steering_output(output_type="configs")
        with open(config_path, "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2, default=str)
        LOGGER.info(f"Config saved to {config_path}")

        torch.cuda.empty_cache()

    def run_all(self) -> None:
        for steering_config, dataset_config, generation_config in product(
            self.cfg.steering_config_list, 
            self.cfg.dataset_config_list,
            self.cfg.generation_config_list
        ):
            dataset = dataset_config.dataset
            
            LOGGER.info(f"==================== Running dataset: {dataset} ==================== ")
                        
            self._run_dataset(
                steering_config=steering_config, 
                dataset_config=dataset_config, 
                generation_config=generation_config
            )
            
            # Save the configuration
            config_path = self.path_manager.get_steering_output("configs")
            with open(config_path, "w") as f:
                json.dump(self.cfg.to_dict(), f, indent=2, default=str)
            LOGGER.info(f"Config saved to {config_path}")
        LOGGER.info("All runs complete.")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_cli(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for hidden-activation collection.
    Matches the keys in run_config.yaml / run_config.json.
    """
    p = argparse.ArgumentParser(description="Collect hidden activations from a transformer.")
    # External config file (overrides CLI where keys overlap)
    p.add_argument("--config",
                   type=str,
                   help="Path to optional JSON/YAML file with the same keys.")
    return p.parse_args(argv)

def main() -> None:
    args = parse_cli()
    cfg = ExperimentConfig.from_sources(vars(args), cfg_path=args.config)
    LOGGER.info("ExperimentConfig:\n%s", json.dumps(cfg.to_dict(), indent=2, default=str))
    SteeringExperimentRunner(cfg).run_all()


if __name__ == "__main__":
    main()
