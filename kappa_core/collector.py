from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from kappa_core.exp_config import ActivationConfig

# --------------------------------------------------------------------------- #
# Constants & Type Aliases
# --------------------------------------------------------------------------- #
LOGGER = logging.getLogger("activation_collector")
Component = Literal["res", "attn", "mlp", "layer"]
COMPONENTS: tuple[Component, ...] = ("res", "attn", "mlp", "layer")


# --------------------------------------------------------------------------- #
# Data Structures
# --------------------------------------------------------------------------- #
@dataclass
class ActivationSample:
    """A container for activations and metadata for a single sample."""

    # Key: (component, layer_idx, token_pos)
    activations: Dict[Tuple[Component, int, int], torch.Tensor] = field(
        default_factory=dict
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Activation Collector
# --------------------------------------------------------------------------- #
class ActivationCollector:
    """
    Stores model activations and metadata for a dataset run.
    """

    def __init__(
        self, activation_config: ActivationConfig
    ) -> None:
        """
        Initializes the collector with the experiment's activation configuration.
        """
        self.config = activation_config

        # Configuration parameters
        self.components: Sequence[Component] = self.config.components
        self.layers: list[int] = list(self.config.layers)
        self.token_positions: list[int | None] = (
            list(self.config.token_positions) if self.config.token_positions else [None]
        )

        # In-memory buffer for finished samples
        self._finished_samples: List[ActivationSample] = []
        self._current_sample: Optional[ActivationSample] = None

        # State for resolving token positions for the current sample
        self._resolved_pos_indices: List[int] = []

        # Running metrics
        self._sum_scores: float = 0.0
        self._n_samples: int = 0

        # Sharding and I/O configuration
        self.activation_shard_size: int = 50
        self.log_progress_every_n: int = 100

    # ------------------- Per-Sample API ------------------- #

    def start_sample(self, seq_len: int) -> None:
        """
        Prepares the collector for a new sample. Must be called before `add`.

        Args:
            seq_len: The sequence length of the new sample's tokens.
        """
        self._current_sample = ActivationSample()
        self._resolve_token_positions(seq_len)

    def add(
        self, *, activation_tuple: Tuple[torch.Tensor, ...], layer_idx: int
    ) -> None:
        """
        Adds activations from a single layer for the current sample.

        Args:
            activation_tuple: A tuple of tensors, one for each component,
                              with shape (1, seq_len, dim).
            layer_idx: The layer index from which activations were extracted.
        """
        if self._current_sample is None:
            LOGGER.error("`add` called before `start_sample`. Skipping.")
            return
        if layer_idx not in self.layers:
            return

        for component, all_tokens_activations in zip(COMPONENTS, activation_tuple):
            if component not in self.components:
                continue

            # Squeeze to remove batch dimension, assuming it's always 1
            activations_at_pos = all_tokens_activations.squeeze(0)
            for token_pos in self._resolved_pos_indices:
                key = (component, layer_idx, token_pos)
                vector = activations_at_pos[token_pos]
                self._current_sample.activations[key] = vector.detach().cpu()

    def finish_sample(self, item_metadata: dict) -> None:
        """
        Finalizes the current sample by attaching metadata and adding it to the buffer.
        """
        if self._current_sample is None:
            LOGGER.error("`finish_sample` called before `start_sample`. Skipping.")
            return

        # Finalize and store the sample
        self._current_sample.metadata = self._prepare_metadata(item_metadata)
        self._finished_samples.append(self._current_sample)
        self._current_sample = None

        # Update and log running metrics
        self._n_samples += 1
        score = item_metadata.get("score", 0.0)
        self._sum_scores += score

        if self._n_samples % self.log_progress_every_n == 0:
            avg_score = self._sum_scores / self._n_samples
            LOGGER.info(
                f"Processed {self._n_samples} samples... "
                f"Running average score: {avg_score:.4f}"
            )

    # ----------------- Dataset-Level API ----------------- #

    def flush_to_disk(self, path_maker: Callable[[str], str], save_activation=True) -> None:
        """
        Writes all buffered activations and metadata to disk.

        Activations are saved in sharded `.pt` files, and metadata is saved
        in a `.jsonl` file. The internal buffer is cleared after flushing.

        Args:
            path_maker: A function that takes an `output_type` string
                        ("activations" or "metadata") and returns a file path.
        """
        if not self._finished_samples:
            LOGGER.warning("No samples to flush.")
            return

        LOGGER.info(f"Flushing {len(self._finished_samples)} samples to disk.")

        if save_activation:
            self._flush_activations(path_maker)
        self._flush_metadata(path_maker)

        # Log final accuracy for the flushed batch
        avg_score = (self._sum_scores / self._n_samples) if self._n_samples > 0 else 0.0
        LOGGER.info(
            f"Flush complete. Total samples: {self._n_samples}. "
            f"Final average score: {avg_score:.4f}"
        )
        # Proportion of samples with pred=1
        LOGGER.info(
            f"Proportion of samples with pred=0: "
            f"{sum(1 for sample in self._finished_samples if sample.metadata.get('pred', -1) == 0) / self._n_samples:.4f}"
        )
        # Proportion of samples with pred=1
        LOGGER.info(
            f"Proportion of samples with pred=1: "
            f"{sum(1 for sample in self._finished_samples if sample.metadata.get('pred', -1) == 1) / self._n_samples:.4f}"
        )

        self._reset_state()

    # ------------------- Private Helpers ------------------- #

    def _resolve_token_positions(self, seq_len: int) -> None:
        """Calculates the absolute token indices to save based on the config."""
        if self.token_positions == [None]:  # Special case to save all tokens
            self._resolved_pos_indices = list(range(seq_len))
        else:
            self._resolved_pos_indices = [
                seq_len + pos if pos < 0 else pos for pos in self.token_positions if pos is not None
            ]

    def _prepare_metadata(self, item_metadata: dict) -> dict:
        """Enriches the item metadata with collection-specific info."""
        metadata = item_metadata.copy()
        metadata["saved_token_positions"] = self._resolved_pos_indices
        metadata["saved_tokens"] = [
            item_metadata["tokens"][pos] for pos in self._resolved_pos_indices
        ]
        return metadata

    def _flush_activations(self, path_maker: Callable[[str], str]) -> None:
        """Saves activation tensors to sharded files."""
        base_path = Path(path_maker(output_type="activations"))
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        shard_idx = 0
        samples_iter = iter(self._finished_samples)
        
        while chunk := list(islice(samples_iter, self.activation_shard_size)):
            data_to_save = {
                (
                    f"item_{sample.metadata['item_idx']}_"
                    f"perm_{sample.metadata['perm_idx']}"
                ): sample.activations
                for sample in chunk
            }
            
            shard_path = base_path.with_suffix(f".shard_{shard_idx:05d}.pt")
            LOGGER.info(f"Saving activation shard ({len(chunk)} samples) to '{shard_path}'")
            try:
                torch.save(data_to_save, shard_path)
            except IOError as e:
                LOGGER.error(f"Failed to write activations to '{shard_path}': {e}")
            shard_idx += 1

    def _flush_metadata(self, path_maker: Callable[[str], str]) -> None:
        """Saves all buffered metadata to a single JSONL file."""
        metadata_path = Path(path_maker(output_type="metadata"))
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Writing {len(self._finished_samples)} metadata records to '{metadata_path}'")
        
        try:
            with metadata_path.open("w", encoding="utf-8") as f:
                for sample in self._finished_samples:
                    line = json.dumps(
                        sample.metadata, ensure_ascii=False, separators=(",", ":")
                    )
                    f.write(line + "\n")
        except IOError as e:
            LOGGER.error(f"Failed to write metadata to '{metadata_path}': {e}")

    def _reset_state(self) -> None:
        """Resets the internal state after a flush."""
        self._finished_samples.clear()
        self._current_sample = None
        self._resolved_pos_indices = []
        self._sum_scores = 0.0
        self._n_samples = 0