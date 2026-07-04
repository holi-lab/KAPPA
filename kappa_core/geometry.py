"""Subspace-geometry math for the knowledge/prediction analysis (paper Appendix B).

Principal angles, orthogonal linear CKA, matched random baselines, and probe /
unembedding weight loading. Moved verbatim from the original geometry analysis
utilities; see docs/superpowers/specs/2026-07-01-geometry-analysis-integration-design.md (§0).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


class PairingLogicError(RuntimeError):
    """Raised when experiment pairing or loading logic is inconsistent."""


MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "Llama-3.1_8B": {
        "d": 4096,
        "hf_model_name": "meta-llama/Llama-3.1-8B",
        "model_name": "Llama-3.1",
        "model_size": "8B",
        "use_base_model": False,
    },
    "Qwen2.5_7B": {
        "d": 3584,
        "hf_model_name": "Qwen/Qwen2.5-7B",
        "model_name": "Qwen2.5",
        "model_size": "7B",
        "use_base_model": False,
    },
    "Qwen3_4B": {
        "d": 2560,
        "hf_model_name": "Qwen/Qwen3-4B",
        "model_name": "Qwen3",
        "model_size": "4B",
        "use_base_model": False,
    },
    "Mistral_7B": {
        "d": 4096,
        "hf_model_name": "mistralai/Mistral-7B-v0.1",
        "model_name": "Mistral",
        "model_size": "7B",
        "use_base_model": False,
    },
}

_WU_CACHE: Dict[str, Tuple[Any, torch.Tensor]] = {}


def center_weights(weights: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(weights, dtype=torch.float32)
    if tensor.ndim != 2:
        raise PairingLogicError(f"Expected 2D weight matrix, got shape={tuple(tensor.shape)}")
    return tensor - tensor.mean(dim=0, keepdim=True)


def _row_space_basis(weights: torch.Tensor) -> torch.Tensor:
    """
    Compute an orthonormal basis for the row space of the given weight matrix.
    
    Parameters:
        weights: A 2D tensor of shape (k, d) representing the probe weights. (k = probe dimension, d = model dimension)
    
    Returns:
        A 2D tensor of shape (d, r) where r is the rank of the input matrix, containing an orthonormal basis for the row space. 
    """
    tensor = torch.as_tensor(weights, dtype=torch.float32)
    if tensor.ndim != 2:
        raise PairingLogicError(f"Expected 2D weight matrix, got shape={tuple(tensor.shape)}")
    if tensor.numel() == 0:
        return torch.zeros((tensor.shape[1], 0), dtype=tensor.dtype, device=tensor.device)

    u, s, _ = torch.linalg.svd(tensor.T, full_matrices=False)
    tol = torch.finfo(s.dtype).eps * max(tensor.shape) * float(s.max().item() if s.numel() else 0.0)
    rank = int((s > tol).sum().item())
    if rank == 0:
        return torch.zeros((tensor.shape[1], 0), dtype=tensor.dtype, device=tensor.device)
    return u[:, :rank]


def compute_principal_angles(weights_a: torch.Tensor, weights_b: torch.Tensor) -> Dict[str, float]:
    basis_a = _row_space_basis(weights_a)
    basis_b = _row_space_basis(weights_b)

    if basis_a.shape[1] == 0 or basis_b.shape[1] == 0:
        return {
            "mean_angle_deg": float("nan"),
            "min_angle_deg": float("nan"),
            "max_angle_deg": float("nan"),
            "proj_frob_dist": float("nan"),
        }

    cosines = torch.linalg.svdvals(basis_a.T @ basis_b).clamp(-1.0, 1.0)
    angles_deg = torch.rad2deg(torch.arccos(cosines))

    proj_a = basis_a @ basis_a.T
    proj_b = basis_b @ basis_b.T
    proj_frob_dist = torch.linalg.norm(proj_a - proj_b, ord="fro").item()

    return {
        "mean_angle_deg": float(angles_deg.mean().item()),
        "min_angle_deg": float(angles_deg.min().item()),
        "max_angle_deg": float(angles_deg.max().item()),
        "proj_frob_dist": float(proj_frob_dist),
    }


def get_random_baseline(d: int, k: int, num_samples: int = 100) -> Dict[str, float]:
    angle_vals: List[float] = []
    frob_vals: List[float] = []
    for _ in range(int(num_samples)):
        w1 = center_weights(torch.randn(k, d))
        w2 = center_weights(torch.randn(k, d))
        result = compute_principal_angles(w1, w2)
        angle_vals.append(float(result["mean_angle_deg"]))
        frob_vals.append(float(result["proj_frob_dist"]))
    return {
        "random_mean_angle_deg": float(np.mean(angle_vals)),
        "random_std_mean_angle_deg": float(np.std(angle_vals)),
        "random_mean_frob_dist": float(np.mean(frob_vals)),
        "random_std_frob_dist": float(np.std(frob_vals)),
    }


def _extract_state_dict(payload) -> Dict[str, torch.Tensor]:
    if isinstance(payload, list):
        if not payload:
            raise PairingLogicError("Probe checkpoint list is empty.")
        if isinstance(payload[0], dict):
            return payload[0]
    if isinstance(payload, dict):
        return payload
    raise PairingLogicError(f"Unsupported probe checkpoint payload type: {type(payload).__name__}")


def load_probe_weights(model_path: Path, d: int, k: int) -> torch.Tensor:
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    if "linear.weight" not in state_dict:
        raise PairingLogicError(f"{model_path}: missing 'linear.weight' in probe checkpoint.")
    weight = torch.as_tensor(state_dict["linear.weight"], dtype=torch.float32)
    if tuple(weight.shape) != (int(k), int(d)):
        raise PairingLogicError(
            f"{model_path}: expected probe weight shape {(int(k), int(d))}, got {tuple(weight.shape)}"
        )
    return weight


def load_W_U_for_options(model_key: str, options: List[str]) -> torch.Tensor:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise PairingLogicError(
            "transformers is required to load W_U option vectors. Install dependencies or rerun with --skip-wu."
        ) from exc

    model_cfg = MODEL_CONFIGS.get(model_key)
    if not model_cfg:
        raise PairingLogicError(f"Unknown model key for W_U loading: {model_key}")

    if model_key not in _WU_CACHE:
        from kappa_core.utils.helpers import get_model_path

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        model_path = get_model_path(
            str(model_cfg["model_size"]),
            bool(model_cfg["use_base_model"]),
            str(model_cfg["model_name"]),
        )
        hf_model_name = str(model_cfg["hf_model_name"])

        tokenizer_exc = None
        model_exc = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                token=hf_token,
                local_files_only=True,
            )
        except Exception as exc:
            tokenizer_exc = exc
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    hf_model_name,
                    token=hf_token,
                    local_files_only=True,
                )
            except Exception as fallback_exc:
                raise PairingLogicError(
                    f"{model_key}: failed to load tokenizer from local files only. "
                    f"Tried model_path={model_path!r} and hf_model_name={hf_model_name!r}. "
                    f"Primary error: {tokenizer_exc}. Fallback error: {fallback_exc}. "
                    "Pre-download/cache the model locally, or rerun with --skip-wu."
                ) from fallback_exc

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                token=hf_token,
                local_files_only=True,
            )
        except Exception as exc:
            model_exc = exc
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_name,
                    token=hf_token,
                    local_files_only=True,
                )
            except Exception as fallback_exc:
                raise PairingLogicError(
                    f"{model_key}: failed to load model from local files only. "
                    f"Tried model_path={model_path!r} and hf_model_name={hf_model_name!r}. "
                    f"Primary error: {model_exc}. Fallback error: {fallback_exc}. "
                    "Pre-download/cache the model locally, or rerun with --skip-wu."
                ) from fallback_exc

        output_embeddings = model.get_output_embeddings()
        if output_embeddings is None or not hasattr(output_embeddings, "weight"):
            raise PairingLogicError(f"{model_key}: model has no output embedding weights.")
        _WU_CACHE[model_key] = (tokenizer, output_embeddings.weight.detach().cpu())
        del model

    tokenizer, weight = _WU_CACHE[model_key]

    token_ids: List[int] = []
    for option in options:
        ids = tokenizer.encode(option, add_special_tokens=False)
        if len(ids) != 1:
            raise PairingLogicError(
                f"{model_key}: option {option!r} does not map to exactly one token; got token ids {ids}."
            )
        token_ids.append(int(ids[0]))

    return weight[token_ids, :]


RANDOM_BASELINE_SAMPLES = 32
RANDOM_BASELINE_SEED = 1729
_RANDOM_BASIS_CACHE: Dict[Tuple[int, int, int, int], List[torch.Tensor]] = {}


def _center_features(features: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(features, dtype=torch.float32)
    if tensor.ndim != 2:
        raise PairingLogicError(f"Expected 2D feature matrix, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] == 0:
        raise PairingLogicError("Cannot compute CKA on an empty feature matrix.")
    return tensor - tensor.mean(dim=0, keepdim=True)


def linear_centered_cka(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    x = _center_features(features_a)
    y = _center_features(features_b)
    if x.shape[0] != y.shape[0]:
        raise PairingLogicError(
            f"CKA requires the same number of samples, got {x.shape[0]} and {y.shape[0]}."
        )

    cross = x.T @ y
    gram_x = x.T @ x
    gram_y = y.T @ y

    numerator = torch.linalg.norm(cross, ord="fro") ** 2
    denominator = torch.linalg.norm(gram_x, ord="fro") * torch.linalg.norm(gram_y, ord="fro")
    if not torch.isfinite(denominator) or float(denominator.item()) <= 0.0:
        raise PairingLogicError("CKA denominator is non-positive or non-finite.")

    value = float((numerator / denominator).item())
    if value < -1e-6 or value > 1.0 + 1e-6:
        raise PairingLogicError(f"CKA escaped the valid range [0, 1]: {value}")
    return max(0.0, min(1.0, value))


def _orthonormal_row_space_basis(weights: torch.Tensor, context: str) -> torch.Tensor:
    centered = center_weights(weights)
    if centered.ndim != 2:
        raise PairingLogicError(f"{context}: expected 2D weight matrix, got shape={tuple(centered.shape)}")
    if centered.numel() == 0:
        raise PairingLogicError(f"{context}: centered weight matrix is empty.")

    basis, singular_values, _ = torch.linalg.svd(centered.T, full_matrices=False)
    if singular_values.numel() == 0:
        raise PairingLogicError(f"{context}: SVD returned no singular values for centered probe weights.")

    tol = torch.finfo(singular_values.dtype).eps * max(centered.shape) * float(singular_values.max().item())
    rank = int((singular_values > tol).sum().item())
    if rank <= 0:
        raise PairingLogicError(
            f"{context}: centered probe row space has rank 0, so orthogonal CKA is undefined."
        )
    return basis[:, :rank]


def _project_to_orthonormal_probe_span(
    activations: torch.Tensor,
    weights: torch.Tensor,
    context: str,
) -> torch.Tensor:
    basis = _orthonormal_row_space_basis(weights, context=context)
    projected = torch.as_tensor(activations, dtype=torch.float32) @ basis
    if projected.ndim != 2 or projected.shape[1] <= 0:
        raise PairingLogicError(
            f"{context}: orthonormal probe-span projection produced invalid shape={tuple(projected.shape)}."
        )
    return projected


def _random_orthonormal_basis(d: int, rank: int, *, generator: torch.Generator) -> torch.Tensor:
    if rank <= 0:
        raise PairingLogicError(f"Random orthonormal basis rank must be positive, got {rank}.")
    random_matrix = torch.randn(int(d), int(rank), generator=generator, dtype=torch.float32)
    basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
    if basis.shape[1] != int(rank):
        raise PairingLogicError(
            f"Random orthonormal basis generation failed for d={d}, rank={rank}: shape={tuple(basis.shape)}."
        )
    return basis


def _get_random_bases(d: int, rank: int, num_samples: int, seed: int) -> List[torch.Tensor]:
    key = (int(d), int(rank), int(num_samples), int(seed))
    if key not in _RANDOM_BASIS_CACHE:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        _RANDOM_BASIS_CACHE[key] = [
            _random_orthonormal_basis(d=int(d), rank=int(rank), generator=generator)
            for _ in range(int(num_samples))
        ]
    return _RANDOM_BASIS_CACHE[key]


def _random_baseline_cka(
    raw_acts: torch.Tensor,
    features: torch.Tensor,
    *,
    d: int,
    rank: int,
    num_samples: int,
    seed: int,
    context: str,
) -> Dict[str, float]:
    values: List[float] = []
    for random_basis in _get_random_bases(d=d, rank=rank, num_samples=num_samples, seed=seed):
        random_features = torch.as_tensor(raw_acts, dtype=torch.float32) @ random_basis
        values.append(linear_centered_cka(features, random_features))
    if not values:
        raise PairingLogicError(f"{context}: random baseline produced no samples.")
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "num_samples": int(num_samples),
    }


__all__ = [
    "PairingLogicError",
    "MODEL_CONFIGS",
    "center_weights",
    "compute_principal_angles",
    "get_random_baseline",
    "load_probe_weights",
    "load_W_U_for_options",
    "linear_centered_cka",
    "RANDOM_BASELINE_SAMPLES",
    "RANDOM_BASELINE_SEED",
    "_random_baseline_cka",
    "_orthonormal_row_space_basis",
    "_project_to_orthonormal_probe_span",
]
