import os
import torch

import logging
from collections import defaultdict
from typing import List, Literal

from kappa_core.data_path import PathManager
from kappa_core.exp_config import ExperimentConfig, SteeringConfig
from run_activation_collection import ActivationExperimentRunner
from run_probe import ProbeExperimentRunner

Split = Literal["train", "validation", "test"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("steering_module_generator")

class SteeringModuleGenerator(ProbeExperimentRunner, ActivationExperimentRunner):
    def __init__(self, cfg, activation_exp_config, probe_exp_config):
        self.cfg = cfg

        super().__init__(cfg)

        self.activation_exp_config = activation_exp_config
        self.probe_exp_config = probe_exp_config
        self.is_prepared = False

    def prepare_probes(self, probe_exp_config:ExperimentConfig):
        local_manager: PathManager = PathManager(probe_exp_config)
        generation_config = probe_exp_config.activation_exp_config.generation_config_list[0]

        self.all_probe_weights = {}
        self.all_probe_biases = {}
        weights = defaultdict(dict)
        biases = defaultdict(dict)
        for probe_config in probe_exp_config.probe_config_list:
            method: str = probe_config.method
            objective: List[str] = probe_config.objective
            component: List[str] = probe_config.component
            model_name: str = probe_config.model_name
            layer: List[int] = probe_config.layer
            token_positions: str = str(probe_config.token_positions)

            local_manager.setup_configs(
                probe_config=probe_config,
                generation_config=generation_config
            )

            weight_path = local_manager.get_probe_output(
                layer=probe_config.layer,
                token_position=token_positions,
                output_type='vectors'
            )
            bias_path = local_manager.get_probe_output(
                layer=probe_config.layer,
                token_position=token_positions,
                output_type='biases'
            )

            weight = torch.load(weight_path, map_location="cpu")
            if isinstance(weight, torch.Tensor):
                pass
            elif isinstance(weight, (list, tuple)):
                weight = weight[0]
            else:
                raise TypeError(f"Unexpected weight type: {type(weight)}")

            if os.path.exists(bias_path):
                bias = torch.load(bias_path, map_location='cpu')[0]
            else:
                bias = torch.tensor(0.0)

            if isinstance(weight, torch.Tensor):
                weights[(method, component, layer, token_positions, model_name, objective)] = weight
            elif isinstance(weight, dict):
                weights[(method, component, layer, token_positions, model_name, objective)] = weight['linear.weight']
            else:
                raise ValueError(f"Unexpected type for weight: {type(weight)}. Expected torch.Tensor or dict.")
            
            biases[(method, component, layer, token_positions, model_name, objective)] = bias

        self.all_probe_weights = weights
        self.all_probe_biases = biases


    def make_params(self, layer:int, steering_config: SteeringConfig):
        weights = []
        thresholds = []
        for i, vector_config in enumerate(steering_config.vector_config_list):
            method = vector_config.method
            component = vector_config.component
            token_positions = vector_config.token_position if isinstance(vector_config.token_position, str) \
                else "_".join(map(str, vector_config.token_position))
            model_name = vector_config.model_name
            objective = vector_config.objective

            weight = self.all_probe_weights[(method, component, layer, token_positions, model_name, objective)]
            bias = self.all_probe_biases[(method, component, layer, token_positions, model_name, objective)]

            LOGGER.info(f"Norm of original weight: {weight.norm(p=2)}")
            weight = weight / weight.norm(p=2)
            thresh = (- bias) / weight.norm(p=2)
            LOGGER.info(f"Norm of normalized weight: {weight.norm(p=2)}")
            weights.append(weight)
            thresholds.append(thresh)
        w = torch.as_tensor(steering_config.w)
        beta = torch.as_tensor(steering_config.beta)
        return (*weights, *thresholds, w, beta)

    def make_steering_module(self, layer, steering_module_class, steering_config):
        if not self.is_prepared:
            self.prepare_probes(self.probe_exp_config)
            self.is_prepared = True
        params = self.make_params(layer, steering_config)
        return steering_module_class(*params)
