from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
import logging
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from collections import defaultdict
from itertools import product
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from kappa_core.exp_config import ExperimentConfig, ProbeConfig, ProbeTrainingConfig
from kappa_core.data_path import PathManager
from kappa_core.probe.model import LogisticRegression, SoftmaxClassifier
from kappa_core.probe.metric import binary_classification_metric, multi_classification_metric
from kappa_core.activation_loader import ProbeActivationLoader

LOGGER = logging.getLogger("probe_runner")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

Split = Literal["train", "validation", "test"]
ActivationDict = Dict[str, Dict[int, Dict[int, Tensor]]]


class ProbeExperimentRunner():
    """
    Orchestrates activation loading, probe training, evaluation and checkpointing.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg: ExperimentConfig = cfg

        self.probe_config_list: List[ProbeConfig] = self.cfg.probe_config_list
        self.path_manager: PathManager = PathManager(self.cfg)
        self.activation_exp_config: ExperimentConfig = self.cfg.activation_exp_config
        
        # runtime state
        self.input_dim: int | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loaders: Dict[Split, Dict[str, DataLoader]] = {}
        self.train_logs: list[dict[str, Any]] = []
        self.best_weights: list[Tensor] = []
        self.best_biases: list[Tensor] = []
        self.best_model_states: list[dict[str, Tensor]] = []
        self.result_items: list[dict[str, Any]] = []

        self.activations = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.items = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.loaders = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    def _is_experiment_complete(self, probe_config: ProbeConfig) -> bool:
        """
        Checks if a probe experiment is complete by verifying the existence of a key output file.
        """
        # Temporarily set up path manager to get the expected output path
        self.path_manager.setup_configs(
            probe_config=probe_config,
            generation_config=self.activation_exp_config.generation_config_list[0]
        )
        # Check for a final artifact, e.g., the training logs.
        log_path_str = self.path_manager.get_probe_output(
            layer=probe_config.layer,
            token_position=probe_config.token_positions,
            output_type='train_logs'
        )
        log_path = Path(log_path_str)

        if log_path.is_file():
            LOGGER.debug(f"[RESUME] Check: Found log file at {log_path}. Experiment is complete.")
            return True
        
        LOGGER.debug(f"[RESUME] Check: Log file not found at {log_path}. Experiment is not complete.")
        return False

    # --------------------------------------------------------------------- #
    # public entry-point                                                    #
    # --------------------------------------------------------------------- #
    def run(self) -> None:
        """Main pipeline."""
        LOGGER.info("Loading activations & preparing dataloaders …")

        self.prepare_activations(self.activation_exp_config)
        self.prepare_activation_loader(self.probe_config_list)

        for probe_config in self.probe_config_list:
            # Check if this experiment has already been completed
            if self._is_experiment_complete(probe_config):
                LOGGER.info(
                    "[RESUME] Skipping completed probe: method='%s', component='%s', layer=%s, pos=%s",
                    probe_config.method, probe_config.component, probe_config.layer, probe_config.token_positions
                )
                continue

            method: str = probe_config.method
            objective: List[str] = probe_config.objective
            component: List[str] = probe_config.component
            model_name: str = probe_config.model_name
            layer: List[int] = probe_config.layer
            token_position: List[int] = probe_config.token_positions

            epochs: int = probe_config.training_config.epochs
            learning_rate: float = probe_config.training_config.learning_rate

            self.setup_experiment(probe_config)

            self._run_experiment(
                method=method,
                model_name=model_name,
                objective=objective,
                component=component,
                layer=layer,
                token_position=token_position,
                epochs=epochs,
                learning_rate=learning_rate
            )

    # --------------------------------------------------------------------- #
    # training / evaluation                                                 #
    # --------------------------------------------------------------------- #
    def _run_experiment(
        self, 
        method: str,
        model_name: str,
        objective: List[str],
        component: List[str],
        layer: List[int],
        token_position: List[int],
        epochs: int,
        learning_rate: float
    ) -> None:

        LOGGER.info(
            f"Running probe for objective: {objective}, component: {component}"
        )
        LOGGER.info(
            "=============================\n"
            f"Processing layer {layer}, position {token_position}"
            "\n============================="
        )

        train_dl:DataLoader = self.loaders["train"][objective][component][layer][token_position]
        val_dl:DataLoader = self.loaders["validation"][objective][component][layer][token_position]
        test_dl:DataLoader = self.loaders["test"][objective][component][layer][token_position]

        LOGGER.info(
            f"Train: {len(train_dl.dataset)} samples, Val: {len(val_dl.dataset)} samples, Test: {len(test_dl.dataset)} samples",
        )

        self.n_classes = train_dl.dataset.n_classes

        # Load the model
        model = self._build_model(model_name)

        # Train and validate the model
        best_w, best_bias, best_state, log = self._train_validate(
            model=model, 
            epochs=epochs, 
            learning_rate=learning_rate,
            train_dl=train_dl, 
            val_dl=val_dl
        )
        
        # Get the best weight and model
        self.best_weights.append(best_w)
        self.best_biases.append(best_bias)
        self.best_model_states.append(best_state)

        # Evaluate the best model
        test_loss, test_acc = self._evaluate(
            model, best_state, test_dl
        )

        log["test_loss"] = test_loss
        log["test_acc"] = test_acc
        self.train_logs.append(log)

        self._save_checkpoints(layer=layer, pos=token_position)

    def prepare_activations(self, activation_exp_config: ExperimentConfig) -> None:
        """
        Load all activations for the given experiment configuration.
        This method iterates over all dataset and generation configurations,
        loads the activations and items, and stores them in the `self.activations` and
        `self.items` dictionaries.
        """
        
        local_manager : PathManager = PathManager(activation_exp_config)
        
        activations: Dict[Split, List[dict]] = defaultdict(dict)
        items: Dict[Split, List[dict]] = defaultdict(dict)

        # Load all the individual activaton varies by the dataset config and generation config
        for ds_cfg, gen_cfg in product(
            activation_exp_config.dataset_config_list, 
            activation_exp_config.generation_config_list
        ):
            local_manager.setup_configs(
                generation_config=gen_cfg,
                dataset_config=ds_cfg
            )
         
            self.activation_loader: ProbeActivationLoader = ProbeActivationLoader(
                generation_config=gen_cfg,
                dataset_config=ds_cfg,
                activation_config=activation_exp_config.activation_config,
                path_manager=local_manager
            )
            
            split = ds_cfg.split
            acts, its = self.activation_loader.load_activation()

            for key in acts:
                if key not in activations[split]:
                    activations[split][key] = []
                if key not in items[split]:
                    items[split][key] = []

                activations[split][key].extend(acts[key])
                items[split][key].extend(its[key])

        # Combine all the activations from different dataset and generation settings
        # Only consider the component, layer and token position
        for split in activations:
            for key in activations[split]:
                comp, layer, pos = key

                act_list = activations[split][key]
                item_list = items[split][key]

                self.activations[split][comp][layer][pos] = torch.stack(act_list)
                self.items[split][comp][layer][pos] = item_list

        self.input_dim = self.activation_loader.input_dim

    def prepare_activation_loader(self, probe_config_list: List[ProbeConfig]) -> None:
        """
        Prepare DataLoaders for each probe configuration.
        This method creates DataLoaders for each combination of objective, component, layer, and token
        position based on the loaded activations and items.
        It iterates over the probe configurations and sets up the DataLoaders for each split (train, validation, test).
        """

        for split in ("train", "validation", "test"):
            
            for probe_config in probe_config_list:
                obj: List[str] = probe_config.objective
                comp: List[str] = probe_config.component
                layer: List[int] = probe_config.layer
                token_position: List[int] = probe_config.token_positions

                activations = []
                items = []
                for pos in token_position.split('_'):
                    activations.extend(
                        self.activations[split][comp][layer][int(pos)] 
                    )
                    items.extend(
                        self.items[split][comp][layer][int(pos)]
                    )
                
                if activations:
                    self.loaders[split][obj][comp][layer][token_position] = self.activation_loader.make_dataloaders(
                        activations=torch.stack(activations), 
                        items=items,
                        objective=probe_config.objective,
                        batch_size=probe_config.training_config.batch_size,
                    )

    # --------------------------------------------------------------------- #
    # setup configs                                                         #
    # --------------------------------------------------------------------- #      
    def setup_experiment(self, probe_config:ProbeConfig):
        self.train_logs.clear()
        self.best_weights.clear()
        self.best_model_states.clear()
        self._setup_criterion(probe_config.training_config)
        self._setup_metric(probe_config.training_config)
        self.path_manager.setup_configs(
            probe_config=probe_config,
            generation_config=self.activation_exp_config.generation_config_list[0]
        )  
    
    def _setup_criterion(self, training_config:ProbeTrainingConfig) -> None:
        if training_config.loss_function == "BCELoss":
            self.criterion = nn.BCELoss()
        elif training_config.loss_function == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(
                f"Unknown loss function: {training_config.loss_function}"
            )

    def _setup_metric(self, training_config: ProbeTrainingConfig) -> None:
        if training_config.metric == "binary_classification_metric":
            self.metric = binary_classification_metric
        elif training_config.metric == "multi_classification_metric":
            self.metric = multi_classification_metric
        else:
            raise NotImplementedError(
                f"Unknown metric: {training_config.metric}"            
            )        

    # --------------------------------------------------------------------- #
    # model helpers                                                         #
    # --------------------------------------------------------------------- #
    def _build_model(self, model_name) -> nn.Module:
        if model_name == "LogisticRegression":
            model = LogisticRegression(input_dim=self.input_dim)
        elif model_name == "SoftmaxClassifier":
            model = SoftmaxClassifier(input_dim=self.input_dim, n_classes=self.n_classes)
        else:
            raise NotImplementedError(f"Unknown probe model: {model_name}")

        return model.to(self.device)

    # --------------------------------------------------------------------- #
    # evaluation                                                            #
    # --------------------------------------------------------------------- #
    def _evaluate(
        self, model: nn.Module, best_state: dict[str, Tensor], test_dl: DataLoader
    ) -> Tuple[float, float]:  # first arg just to match call-site
        model.load_state_dict(best_state)
        model.eval()
        test_loss, test_acc = self._run_one_epoch(model, test_dl, split="test", verbose=True)
        LOGGER.info("Final Test  Loss %.4f | Acc %.4f", test_loss, test_acc)
        return test_loss, test_acc

    # --------------------------------------------------------------------- #
    # training / validation helpers                                         #
    # --------------------------------------------------------------------- #
    def _train_validate(
        self, model: nn.Module, epochs:int, train_dl: DataLoader, val_dl: DataLoader, learning_rate: float
    ) -> Tuple[Tensor, dict[str, Tensor], dict[str, list[float]]]:
        """Train `model`; return best weight, state-dict and log history."""
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_acc = -1.0
        best_weight: Tensor | None = None
        best_bias: Tensor | None = None
        best_state: dict[str, Tensor] | None = None

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(epochs):
            tloss, tacc = self._run_one_epoch(model, train_dl, opt, split="train", verbose=False)
            vloss, vacc = self._run_one_epoch(model, val_dl, split="validation", verbose=False)

            history["train_loss"].append(tloss)
            history["train_acc"].append(tacc)
            history["val_loss"].append(vloss)
            history["val_acc"].append(vacc)

            if vacc > best_acc:
                best_acc = vacc
                best_weight = model.get_weight().detach()
                best_bias = model.get_bias()
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if (epoch + 1) % max(1, epochs // 10) == 0:
                LOGGER.info(
                    "[%3d/%d] Train %.4f|%.4f  Val %.4f|%.4f",
                    epoch + 1,
                    epochs,
                    tloss,
                    tacc,
                    vloss,
                    vacc,
                )

        LOGGER.info("Best Val Acc %.4f", best_acc)
        assert best_weight is not None and best_bias is not None and best_state is not None  # mypy
        return best_weight, best_bias, best_state, history

    def _run_one_epoch(
        self,
        model: nn.Module,
        dl: DataLoader,
        opt: torch.optim.Optimizer | None = None,
        split: str = "train",
        verbose: bool = False
    ) -> Tuple[float, float]:
        total_loss, correct = 0.0, 0
        model.train(split=="train")

        for idx, x, y in dl:
            x, y = x.to(self.device), y.to(self.device)
            y = y.float() if isinstance(self.criterion, nn.BCELoss) else y.long()
            pred = model(x)
            if pred.ndim == y.ndim:
                pred = pred.view_as(y)

            loss = self.criterion(pred, y)
            if split=="train":
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_loss += loss.item() * len(x)
            correct += self.metric(pred, y).sum().item()
        
            if split == "test":
                for i in range(len(x)):
                    item = dl.dataset.items[idx[i]].copy()
                    item["logits"] = pred[i].detach().cpu().numpy().tolist()
                    item["pred"] = pred[i].argmax().item()
                    item["answer"] = y[i].item()
                    item["score"] = self.metric(pred[i], y[i]).item()
                    self.result_items.append(item)

                    if verbose:
                        LOGGER.info(f"Finished item {item['item_idx']} with perm {item['perm_idx']}")
                        LOGGER.info(f"Prediction: {item['pred']}, Score: {item['score']}")
                        scores = [item['score'] for item in self.result_items]
                        avg_score = sum(scores) / len(scores) if scores else 0
                        LOGGER.info(f"Accumulated Accuracy: {avg_score:.4f}")

        n = len(dl.dataset)
        return total_loss / n, correct / n

    # --------------------------------------------------------------------- #
    # checkpointing                                                         #
    # --------------------------------------------------------------------- #
    def _save_checkpoints(self, layer: int, pos: int) -> None:
        artefacts = {
            "vectors": self.best_weights,
            "biases": self.best_biases,
            "model": self.best_model_states,
            "train_logs": self.train_logs,
            "metadata": self.result_items,
            "configs": self.cfg.to_dict(),
        }
        for kind, obj in artefacts.items():
            self._save_artifact(
                kind,
                obj,
                layer=layer,
                pos=pos
            )
        # Clear the state for the next run
        self.best_weights.clear()
        self.best_biases.clear()
        self.best_model_states.clear()
        self.result_items.clear()

    def _save_artifact(
        self,
        kind: str,
        obj: Any,
        layer: Optional[int] = None,
        pos: Optional[int] = None
    ) -> None:
        save_path = self.path_manager.get_probe_output(
            layer=layer,
            token_position=pos if pos else -1,
            output_type=kind
        )
        if kind in ["train_logs", "metadata", "configs"]:
            with open(save_path, "w", encoding="utf-8") as fp:
                json.dump(obj, fp, indent=2)
        else:
            torch.save(obj, save_path)
        LOGGER.info("Saved %s → %s", kind, save_path)

    # --------------------------------------------------------------------- #
    # CLI helpers                                                           #
    # --------------------------------------------------------------------- #
    @staticmethod
    def parse_cli(argv: Iterable[str] | None = None) -> argparse.Namespace:
        p = argparse.ArgumentParser(description="Run probe experiment.")
        p.add_argument("--config", type=str, help="Path to JSON/YAML config.")
        return p.parse_args(argv)


def main() -> None:
    args = ProbeExperimentRunner.parse_cli()
    cfg = ExperimentConfig.from_sources(vars(args), cfg_path=args.config)
    LOGGER.info("ExperimentConfig:\n%s", json.dumps(cfg.to_dict(), indent=2, default=str))
    ProbeExperimentRunner(cfg).run()


if __name__ == "__main__":
    main()
