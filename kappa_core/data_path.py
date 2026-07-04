import os
from pathlib import Path
from typing import Literal, Optional
from datetime import datetime

from kappa_core.exp_config import (
    ExperimentConfig, 
    DatasetConfig,
    GenerationConfig,
    ProbeConfig, 
    SteeringConfig
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Path manager
# --------------------------------------------------------------------------- #
class PathManager:
    def __init__(
        self, 
        cfg: ExperimentConfig,
    ) -> None:
        self.cfg = cfg
        self.llm_config = cfg.llm_config

        # State variables
        self.dataset_config: Optional[DatasetConfig] = None
        self.generation_config: Optional[GenerationConfig] = None
        self.probe_config: Optional[ProbeConfig] = None
        self.steering_config: Optional[SteeringConfig] = None

        self.output_dir_name = "outputs"
        self.datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_path = BASE_DIR / self.output_dir_name
        self.path = None

    def setup_configs(
        self,
        dataset_config: DatasetConfig = None,
        generation_config: GenerationConfig = None,
        probe_config: Optional[ProbeConfig] = None,
        steering_config: Optional[SteeringConfig] = None,
    ) -> None:
        self.dataset_config = dataset_config
        self.generation_config = generation_config
        self.probe_config = probe_config
        self.steering_config = steering_config

        # Ensure the base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def config_sanity_check(self, exp_type: str = None) -> None:
        if exp_type == 'activations':
            if (not self.generation_config) or (not self.dataset_config) or (not self.llm_config):
                raise ModuleNotFoundError(
                    "For Activation experiment, " \
                    "dataset_config and generation_config and llm_config should be set"
                )
            
        elif exp_type == 'probe':
            if (not self.generation_config) or (not self.probe_config):
                raise ModuleNotFoundError(
                    "For Probe experiment, " \
                    "both probe_config and generation_config should be set"
                )
        
        elif exp_type == 'steering':
            if (not self.steering_config) or (not self.generation_config):
                raise ModuleNotFoundError(
                    "For Steering / Steering Training experiment, " \
                    "both steering_config and generation_config should be set"
                )
        else:
            raise ValueError('Unkown experiment type.')
        return
    
    def _make_path(self, *parts: str, mkdir: bool = True) -> Path:
        dir_path = self.base_path.joinpath(*parts)
        if mkdir:
            dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def _get_llm_name(self) -> str:
        if self.llm_config:
            llm_config = self.llm_config
        elif self.cfg.activation_exp_config.llm_config:
            llm_config = self.cfg.activation_exp_config.llm_config
        else:
            raise ValueError("LLM configuration is not set.")
        return f"{llm_config.model_name}_{llm_config.model_size}"

    def _get_filename(self, output_type: str, prefix: str = "", suffix: str = "") -> str:
        model_name = self.probe_config.model_name if self.probe_config else None

        if output_type == "activations":
            return f"{prefix}_{suffix}.pt"
        elif output_type == "metadata":
            return f"{prefix}_{suffix}.json"
        elif output_type == "train_logs":
            return f"{prefix}_{model_name}.pt"
        elif output_type == "model" or output_type == "vectors" or output_type == "biases":
            return f"{prefix}_{model_name}.pt"
        elif output_type == "modules":
            return f"{prefix}_.pt"
        elif output_type == "configs":
            return f"config_{self.datetime_string}.json"
        else:
            raise ValueError(f"Unknown output type: {output_type}")

    def get_activation_output(self, output_type: Literal["activations", "metadata", "configs"]):
        self.config_sanity_check(exp_type='activations')
        polarity = "pos" if self.generation_config.is_positive else "neg"
        dir_path = self._make_path(
            self._get_llm_name(),
            "activations",
            self.dataset_config.dataset,
            f"option_{self.dataset_config.num_option}",
            self.dataset_config.split,
            self.dataset_config.make_prompt_config_id(),
            self.generation_config.decoding_mode,
        )
        fname = self._get_filename(output_type, prefix=output_type, suffix=polarity)
        self.path = dir_path / fname
        return self.path

    def get_probe_output(self, layer:int, token_position:int, output_type: Literal["model", "metadata", "train_logs", "vectors", "biases", "configs"]):
        self.config_sanity_check(exp_type='probe')
        dir_path = self._make_path(
            self._get_llm_name(),
            "probes",
            self.probe_config.save_name,
            self.generation_config.decoding_mode,
            self.probe_config.objective,
            f"{self.probe_config.component}_layer_{layer}_token_{token_position}",
            f"lr_{self.probe_config.training_config.learning_rate}_bs_{self.probe_config.training_config.batch_size}_ep_{self.probe_config.training_config.epochs}"
        )
        fname = self._get_filename(output_type, prefix=output_type)
        self.path = dir_path / fname
        return self.path

    def get_steering_output(self, output_type: Literal["metadata", "activations", "configs"]):
        self.config_sanity_check(exp_type='steering')
        dir_path = self._make_path(
            self._get_llm_name(),
            "steering",
            self.steering_config.save_name,
            self.dataset_config.dataset,
            self.generation_config.decoding_mode,
            self.steering_config.method,
            self.steering_config.make_steering_modules_id(),
            self.steering_config.component,
            self.steering_config.make_intervention_layers_id(),
            f"multiplier_{self.steering_config.multiplier}_w_{self.steering_config.w}_beta_{self.steering_config.beta}",
        )
        fname = self._get_filename(output_type, prefix=output_type, suffix=self.dataset_config.split)
        self.path = dir_path / fname
        return self.path

    
    def _dataset_file(self) -> Path:
        if (self.dataset_config.num_option) and (self.dataset_config.type != 'open_ended'):
            return (
                BASE_DIR
                / "data"
                / self.dataset_config.dataset
                / self.dataset_config.split
                / f"options_{self.dataset_config.num_option}"
                / f"{self.dataset_config.type}.json"
            )
        else:
            return (
            BASE_DIR
            / "data"
            / self.dataset_config.dataset
            / self.dataset_config.split
            / f"{self.dataset_config.type}.json"
        )
