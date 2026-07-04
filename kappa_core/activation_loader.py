from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from kappa_core.data_path import PathManager
from kappa_core.exp_config import ActivationConfig, DatasetConfig, GenerationConfig
from kappa_core.probe.data import ProbeItemDataset

LOGGER = logging.getLogger(__name__)
Split = Literal["train", "validation", "test"]
Component = Literal["res", "attn", "mlp", "layer"]
COMPONENTS: Tuple[Component, ...] = ("res", "attn", "mlp", "layer")

ActivationDict = Dict[Tuple[Component, int, int], List[Tensor]]

class ProbeActivationLoader:
    """
    Loads sharded activations and their corresponding metadata from disk.

    This class is designed to read the output of the `ActivationCollector`,
    which stores activations in sharded `.pt` files and metadata in a `.jsonl` file.
    It filters and structures the data into a format suitable for training probe models.
    """

    def __init__(
        self,
        generation_config: GenerationConfig,
        dataset_config: DatasetConfig,
        activation_config: ActivationConfig,
        path_manager: PathManager,
    ):
        self.path_manager = path_manager
        self.generation_config = generation_config
        self.dataset_config = dataset_config
        self.activation_config = activation_config

        # Unpack config for easier access
        self.layers = set(activation_config.layers)
        self.token_positions = activation_config.token_positions
        self.components = set(activation_config.components)

        # Caching to avoid re-reading files within the same instance
        self.cached_activations: Dict[str, Dict] = {}
        self.cached_metadata: Dict[str, List[Dict]] = {}
        self.input_dim: int | None = None

    def load_activation(self) -> Tuple[ActivationDict, List[dict[str, Any]]]:
        """
        Load all activations and metadata for a given path from path manager.

        This method reads all activation shards and the metadata file, filters them
        according to the experiment configuration, and structures them for probing.
        """
        LOGGER.info("Loading activations for probe from %s", Path(self.path_manager.get_activation_output(output_type="activations").parent))

        activations: Dict[Tuple, List] = defaultdict(list)
        items: Dict[Tuple, List] = defaultdict(list)

        # 1. Load all metadata and create an efficient lookup map.
        try:
            metadata_list = self._read_metadata()
            metadata_map = {
                (m["item_idx"], m["perm_idx"]): m for m in metadata_list
            }
        except FileNotFoundError as e:
            LOGGER.warning("Could not load activations: %s", e)
            return {}, {}
        
        # 2. Iterate through all activation shards.
        for shard in self._read_activations():
            for sample_key, activations_for_sample in shard.items():
                # 3. Reconstruct sample identifiers from the key.
                try:
                    # Key format: "item_{item_idx}_perm_{perm_idx}"
                    _, item_idx_str, _, perm_idx_str = sample_key.split("_")
                    item_idx = int(item_idx_str)
                    perm_idx = int(perm_idx_str)
                except (ValueError, IndexError):
                    _, item_idx_str = sample_key.split("_")
                    item_idx = int(item_idx_str)

                # 4. Process each activation vector for the sample.
                for key, tensor in activations_for_sample.items():
                    if isinstance(key, str):
                        _, perm_idx_str = key.split("_")
                        perm_idx = int(perm_idx_str)
                    
                    # 5. Find corresponding metadata.
                    item_meta = metadata_map.get((item_idx, perm_idx))
                    if not item_meta:
                        continue  # This sample's metadata was not found; skip.

                    seq_len = len(item_meta["tokens"])
                    resolved_config_positions = self._resolve_positions(seq_len)


                    if isinstance(key, tuple):
                        comp, layer, pos = key
                        # 6. Filter based on the loader's configuration.
                        if comp not in self.components:
                            continue
                        if layer not in self.layers:
                            continue
                        if pos not in resolved_config_positions:
                            continue

                        if self.input_dim is None:
                            self.input_dim = tensor.shape[-1]

                        # convert absolute position
                        # to a negative index for use in the final key.
                        final_pos_key = pos - seq_len if pos >= 0 else pos

                        key = (comp, layer, final_pos_key)
                        activations[key].append(tensor)
                        items[key].append(item_meta)

                        if self.input_dim is None:
                            self.input_dim = tensor.shape[0]

                    elif isinstance(key, str):
                        for comp, pos_layer_dict in tensor.items():
                            if comp not in self.components:
                                continue
                            for token_pos_idx, layer_act in pos_layer_dict.items():
                                pos = int(token_pos_idx.split("_")[1])  # e.g., "token_0" -> 0
                                resolve_pos = pos + seq_len if pos < 0 else pos  # convert negative to positive index
                                if resolve_pos not in resolved_config_positions:
                                    continue
                                
                                # Adjust position to be negative
                                final_pos_key = pos - seq_len if pos >= 0 else pos

                                for layer_idx, act in layer_act.items():
                                    layer = int(layer_idx.split("_")[1])  # e.g., "layer_12" -> 12
                                    if layer in self.layers:
                                        activations[(comp, layer, final_pos_key)].append(act)
                                        items[(comp, layer, final_pos_key)].append(item_meta)

                                    if self.input_dim is None:
                                        self.input_dim = act.shape[0]

        return dict(activations), dict(items)

    def _read_activations(self) -> Generator[Dict[str, Any], None, None]:
        """Finds all activation shards, loads them, and yields them one by one."""
        
        base_path = Path(self.path_manager.get_activation_output(output_type="activations"))

        if base_path.exists():
            shard_files = [base_path]
        else:
            # The new format uses a predictable suffix, so glob is reliable.
            shard_files = sorted(base_path.parent.glob(f"{base_path.stem}.shard_*.pt"))
            if not shard_files:
                raise FileNotFoundError(f"No activation shards found matching pattern in '{base_path.parent}'")

        for shard_path in shard_files:
            shard_path_str = str(shard_path)
            if shard_path_str in self.cached_activations:
                yield self.cached_activations[shard_path_str]
                continue
            try:
                loaded_shard = torch.load(shard_path, map_location="cpu")
                self.cached_activations[shard_path_str] = loaded_shard
                yield loaded_shard
                
            except (IOError, pickle.UnpicklingError) as e:
                LOGGER.error("Failed to load or read activation shard '%s': %s", shard_path, e)
    
    def _read_metadata(self) -> List[Dict[str, Any]]:
        """
        Reads a metadata file (.json or .jsonl) into a list of dictionaries.
        Handles both standard JSON and JSON Lines formats.
        """
        meta_path = Path(self.path_manager.get_activation_output(output_type="metadata"))
        meta_path_str = str(meta_path)

        if meta_path_str in self.cached_metadata:
            return self.cached_metadata[meta_path_str]

        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        data = []
        try:
            # Try reading as a single JSON object first
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            
            except:
                try:
                    with meta_path.open("r", encoding="utf-8") as f:
                        data = [json.loads(line) for line in f if line.strip()]
                
                except json.JSONDecodeError:
                    # If it fails, reset file pointer and try reading as concatenated JSON
                    data = load_malformat_json(meta_path)

            if isinstance(data, list):
                metadata_list = data
            elif isinstance(data, dict):
                # Handle nested dictionary format
                metadata_list = []
                for perm_dict in data.values():
                    for item in perm_dict.values():
                        metadata_list.append(item)
            self.cached_metadata[meta_path_str] = metadata_list
            return metadata_list
        except Exception as e:
            LOGGER.error("Failed to read metadata from '%s': %s", meta_path, e)
            raise

    def _resolve_positions(self, seq_len: int) -> List[int]:
        """
        Calculates the absolute token indices to load based on the config.
        Handles negative indices (from the end of the sequence).
        """
        if self.token_positions is None or self.token_positions == [None]:
            return list(range(seq_len))
        else:
            return [
                seq_len + pos if pos < 0 else pos for pos in self.token_positions
            ]

    # --------------------------------------------------------------------- #
    # dataloader construction                                               #
    # --------------------------------------------------------------------- #
    def make_dataloaders(
        self, 
        activations: ActivationDict, 
        items: List[dict[str, Any]],
        objective: str,
        batch_size: int,
    ):
        """
        Create ProbeItemDataset & DataLoader hierarchy:
        loaders[objective][comp][layer][pos] = DataLoader
        """
        dataset = ProbeItemDataset(
            activations=activations,
            items=items,
            objective=objective,
        )

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    @staticmethod
    def _load_json(path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def load_malformat_json(meta_path):
    data = []
    f = meta_path.open("r", encoding="utf-8")
    f.seek(0)
    content = f.read()
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        content_trimmed = content[pos:].lstrip()
        if not content_trimmed:
            break
        
        offset = len(content[pos:]) - len(content_trimmed)

        try:
            obj, end_pos = decoder.raw_decode(content_trimmed)
            data.append(obj)
            pos += offset + end_pos
        except json.JSONDecodeError:
            # Fallback to bracket counting for malformed JSON
            open_brackets = 0
            start_index = -1
            found_object = False
            
            search_content = content[pos:]

            for i, char in enumerate(search_content):
                if char == '{':
                    if open_brackets == 0:
                        start_index = i
                    open_brackets += 1
                elif char == '}':
                    if open_brackets > 0:
                        open_brackets -= 1
                    if open_brackets == 0 and start_index != -1:
                        json_str = search_content[start_index : i+1]
                        try:
                            data.append(json.loads(json_str))
                            found_object = True
                        except json.JSONDecodeError:
                            pass
                        
                        pos += i + 1
                        start_index = -1
                        break
            
            if not found_object:
                break
    if len(data) > 1:
        data = data[0]

    return data
