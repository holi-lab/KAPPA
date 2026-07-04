from __future__ import annotations
import json
import yaml
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = [
    "BaseConfig",
    "LLMConfig",
    "GenerationConfig",
    "PromptConfig",
    "DatasetConfig",
    "ActivationConfig",
    "ProbeTrainingConfig",
    "ProbeConfig",
    "VectorConfig",
    "SteeringModuleConfig",
    "InterventionLayerConfig",
    "SteeringConfig",
    "ExperimentConfig",
]

@dataclass()
class BaseConfig:
    """Common base class carrying attributes shared across many configurations.

    Parameters
    ----------
    start_time:
        Human readable timestamp when the configuration was instantiated.
        Defaults to the current time in `YYYY‑MM‑DD HH:MM:SS format.
    output_dir:
        Optional directory path where outputs should be written.
    experiment_type:
        Identifier describing the type of experiment (e.g., `"steering",
        `"probe").  When None the caller must set it explicitly.
    splits:
        Optional list of dataset split names applicable at the top level of
        an experiment (e.g., `["train", "validation"]).
    """

    start_time: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    output_dir: Optional[str] = None
    experiment_type: Optional[str] = None
    splits: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert the dataclass and any nested dataclasses into a dict.

        Only fields with non‑`None values are included to prevent noise in
        serialized output.  Nested dataclasses and lists of dataclasses are
        processed recursively.
        """

        def _convert(obj: Any) -> Any:
            if obj is None:
                return None
            if is_dataclass(obj):
                return {k: _convert(v) for k, v in asdict(obj).items() if v is not None}
            if isinstance(obj, list):
                return [_convert(item) for item in obj]
            return obj

        return _convert(self)

    @classmethod
    def resolve_nested(cls, **kwargs: Any) -> List[BaseConfig]:
        """Create a single configuration instance from the given parameters.

        Derived classes override this method to generate multiple
        configurations when parameters are provided as sequences.  In the
        base class the method simply returns a one‑element list containing
        `cls(**kwargs).

        Parameters
        ----------
        **kwargs:
            Arbitrary keyword arguments used to initialize the configuration.

        Returns
        -------
        List[BaseConfig]
            A list containing one initialized instance.
        """

        return [cls(**kwargs)]


@dataclass()
class LLMConfig:
    """Configuration describing a language model.

    Parameters
    ----------
    model_name:
        Name of the model family (e.g. `"Llama-2" or "GPT-3").
    use_base_model:
        Flag indicating whether to use the base model instead of an
        instruction‑tuned variant.
    model_size:
        Variant size (e.g. `"7b", "13b", "70b").  Not all models
        support all sizes.
    """

    model_name: Optional[str] = None
    use_base_model: Optional[bool] = None
    model_size: Optional[str] = None
    model_path: Optional[str] = None


@dataclass()
class GenerationConfig:
    """Configuration controlling text generation.

    Parameters
    ----------
    decoding_mode:
        Decoding strategy for generation (e.g. `"teacher_forced",
        `"open_generate", "logit_generate").
    polarity:
        Polarity of the generated output.  If a sequence is provided the
        :meth:resolve_nested class method will produce a separate
        :class:GenerationConfig for each element.
    is_positive:
        Derived flag indicating whether the polarity is positive or not.
    """

    decoding_mode: Optional[str] = None
    polarity: Optional[str] = None
    is_positive: Optional[bool] = None
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    do_sample: Optional[bool] = False,
    max_new_tokens: Optional[int] = 300,

    @classmethod
    def resolve_nested(
        cls,
        decoding_mode: Optional[str] = None,
        polarity: Optional[Union[str, Sequence[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = False,
        max_new_tokens: Optional[int] = 300,
    ) -> List[GenerationConfig]:
        """Expand generation configurations for multiple polarities.

        Parameters
        ----------
        decoding_mode:
            Generation mode to apply to all returned configurations.  If
            `None a default of "open_generate" is assumed.
        polarity:
            A single string or sequence of strings indicating the
            polarities to generate.  If `None a default polarity of
            `"positive" is used.

        Returns
        -------
        List[GenerationConfig]
            A list of generation configurations, one per polarity value.
        """

        mode = decoding_mode or "open_generate"
        pol_list: Sequence[str]
        if polarity is None:
            pol_list = ["positive"]
        elif isinstance(polarity, str):
            pol_list = [polarity]
        else:
            pol_list = list(polarity)

        configs: List[GenerationConfig] = []
        for p in pol_list:
            is_pos = None
            # Derive boolean if explicit polarity strings are recognized.
            if isinstance(p, str):
                lowered = p.lower()
                if lowered in {"pos", "positive"}:
                    is_pos = True
                elif lowered in {"neg", "negative"}:
                    is_pos = False
            configs.append(
                cls(
                    decoding_mode=mode, 
                    polarity=p, 
                    is_positive=is_pos,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    max_new_tokens=max_new_tokens
                )
            )
        return configs


@dataclass()
class PromptConfig:
    """Configuration controlling the construction of multiple choice prompts.

    The four core fields correspond to specific parts of the prompt:

    * `mcq_inst_version – identifies which instruction template to use.
    * `answer_format_version – selects the answer formatting style.
    * `option_symbol – determines how answer options are labelled.
    * `option_wrapper – defines the wrapper used around option symbols.

    To maintain backward compatibility with earlier versions of the code
    where plural field names were used (`mcq_inst_versions etc.),
    read‑only properties are provided.
    """

    mcq_inst_version: Optional[str]
    answer_format_version: Optional[str]
    option_symbol: Optional[str]
    option_wrapper: Optional[str]

    @property
    def mcq_inst_versions(self) -> str:
        return self.mcq_inst_version

    @property
    def answer_format_versions(self) -> str:
        return self.answer_format_version

    @property
    def option_symbols(self) -> str:
        return self.option_symbol

    @property
    def option_wrappers(self) -> str:
        return self.option_wrapper

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def resolve_nested(
        cls,
        mcq_inst_versions: Union[str, Sequence[str]],
        answer_format_versions: Union[str, Sequence[str]],
        option_symbols: Union[str, Sequence[str]],
        option_wrappers: Union[str, Sequence[str]],
    ) -> List[PromptConfig]:
        """Enumerate the Cartesian product of prompt configuration parameters.

        Accepts either single values or sequences of values for each field
        and produces a list containing one :class:PromptConfig instance per
        combination.  This helper is commonly used to expand YAML list
        values into individual configurations.

        Parameters
        ----------
        mcq_inst_versions:
            Single instruction version or a sequence of versions.
        answer_format_versions:
            Single answer format or a sequence of formats.
        option_symbols:
            Single option symbol or a sequence of symbols.
        option_wrappers:
            Single wrapper type or a sequence of wrappers.

        Returns
        -------
        List[PromptConfig]
            A list of prompt configurations covering all combinations.
        """

        # Normalize each argument to a list for Cartesian product
        def _to_list(value: Union[str, Sequence[str], None]) -> List[Optional[str]]:
            """Normalize a potentially None, string or sequence into a list.

            A `None value produces [None] to mirror the behaviour of
            earlier versions of the code which allowed missing prompt
            parameters.
            """
            if value is None:
                return [None]
            if isinstance(value, str):
                return [value]
            return list(value)

        inst_list = _to_list(mcq_inst_versions)
        ans_list = _to_list(answer_format_versions)
        symbol_list = _to_list(option_symbols)
        wrapper_list = _to_list(option_wrappers)

        configs: List[PromptConfig] = []
        for inst, ans, sym, wrap in product(inst_list, ans_list, symbol_list, wrapper_list):
            configs.append(
                cls(
                    mcq_inst_version=inst,
                    answer_format_version=ans,
                    option_symbol=sym,
                    option_wrapper=wrap,
                )
            )
        return configs


@dataclass()
class DatasetConfig:
    """Configuration describing a dataset split used in an experiment.

    Parameters
    ----------
    dataset:
        Name of the dataset to use.
    split:
        Name of the subset or split (e.g. `"train" or "test").  May be
        `None to indicate that no split information is required.
    num_option:
        Number of answer options to include in multiple choice questions.
    random_seed:
        Random seed used to sample or shuffle the data.
    type:
        Human readable type of dataset (e.g. `"mcqa" for multiple choice
        question answering).  Kept as a string rather than an enum to ease
        extension.
    num_permutations:
        Number of permutations to generate when sampling.
    prompt_config:
        Prompt configuration associated with this dataset.  See
        :class:PromptConfig for details.
    """

    dataset: str
    split: Optional[str]
    num_option: Optional[int]
    random_seed: Optional[int]
    type: Optional[str]
    num_permutations: Optional[int]
    prompt_config: PromptConfig

    def make_prompt_config_id(self) -> str:
        """Compute a compact identifier summarizing the prompt configuration.

        The identifier concatenates abbreviated versions of each prompt
        parameter.  This is primarily used for naming output directories or
        filenames.  The mapping rules mirror the original code but are
        encapsulated here for clarity.  If an unknown option symbol or
        wrapper is encountered the raw value will be used.

        Returns
        -------
        str
            A compact string identifier composed of the prompt elements.
        """

        inst_id = self.prompt_config.mcq_inst_version.replace("instruction_", "inst")
        ans_id = self.prompt_config.answer_format_version.replace(
            "answer_format_", "ans"
        )
        # Map known option symbols to shorter names
        symbol_map = {
            "common_alphabet": "alpha",
            "common_alphabet_upper": "alpha-upper",
            "roman_numerals": "roman",
            "roman_numerals_upper": "roman-upper",
            "ordinal_numbers": "ordinal",
            "ordinal_numbers_upper": "ordinal-upper",
            "common_numbers": "num",
        }
        sym_id = symbol_map.get(self.prompt_config.option_symbol, self.prompt_config.option_symbol)
        # Use only the prefix before the first underscore for the wrapper
        wrap_id = self.prompt_config.option_wrapper.split("_")[0]
        return f"{inst_id}_{ans_id}_{sym_id}_{wrap_id}"

    @classmethod
    def resolve_nested(
        cls,
        datasets: Union[str, Sequence[str]],
        splits: Optional[Union[str, Sequence[Optional[str]]]],
        num_option: Optional[int],
        random_seed: Optional[int],
        type: Optional[str],
        num_permutations: Optional[int],
        prompt_configs: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    ) -> List[DatasetConfig]:
        """Expand dataset configurations for multiple datasets, splits and prompt sets.

        Parameters
        ----------
        datasets:
            Single dataset name or a sequence of dataset names.
        splits:
            Single split name or a sequence of split names.  `None values
            indicate that the dataset has no explicit split.
        num_option:
            Number of options for multiple choice questions.  May be `None.
        random_seed:
            Random seed for sampling operations.  May be `None.
        type:
            Type of dataset (e.g., `"mcqa").  May be None.
        num_permutations:
            Number of sampling permutations.  May be `None.
        prompt_configs:
            Either a single prompt configuration dictionary or a sequence
            thereof.  Each dictionary must contain keys
            `"mcq_inst_versions", "answer_format_versions", "option_symbols" and
            `"option_wrappers" mapping to strings or sequences of strings.

        Returns
        -------
        List[DatasetConfig]
            A list containing one dataset configuration for every
            combination of dataset, split and prompt parameters.
        """

        def _to_list(value: Union[str, Sequence[Any], None]) -> List[Any]:
            if value is None:
                return [None]
            if isinstance(value, str):
                return [value]
            return list(value)

        ds_list = _to_list(datasets)
        split_list = _to_list(splits)

        # Normalize prompt_configs to a list of dicts
        if isinstance(prompt_configs, dict):
            prompt_config_dicts: List[Dict[str, Any]] = [prompt_configs]
        else:
            prompt_config_dicts = list(prompt_configs)

        prompt_instances: List[PromptConfig] = []
        for pc_dict in prompt_config_dicts:
            # Each dictionary may specify lists or single values; use PromptConfig.resolve_nested
            inst_versions = pc_dict.get("mcq_inst_versions")
            ans_versions = pc_dict.get("answer_format_versions")
            option_symbols = pc_dict.get("option_symbols")
            option_wrappers = pc_dict.get("option_wrappers")
            prompt_instances.extend(
                PromptConfig.resolve_nested(
                    mcq_inst_versions=inst_versions,
                    answer_format_versions=ans_versions,
                    option_symbols=option_symbols,
                    option_wrappers=option_wrappers,
                )
            )

        configs: List[DatasetConfig] = []
        for ds, split, prompt in product(ds_list, split_list, prompt_instances):
            configs.append(
                cls(
                    dataset=ds,
                    split=split,
                    num_option=num_option,
                    random_seed=random_seed,
                    type=type,
                    num_permutations=num_permutations,
                    prompt_config=prompt,
                )
            )
        return configs


@dataclass()
class ActivationConfig:
    """Configuration specifying which layers and token positions to record.

    Parameters
    ----------
    layers:
        List of zero‑based layer indices to record.  `None means all
        layers.
    token_positions:
        List of token positions to record.  A value of `-1 typically
        indicates the last token.  `None means all positions.
    """

    layers: Optional[List[int]] = None
    token_positions: Optional[List[int]] = None
    components: Optional[List[str]] = None


@dataclass()
class ProbeTrainingConfig:
    """Configuration controlling probe training hyperparameters."""

    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    epochs: Optional[int] = None
    loss_function: Optional[str] = None
    metric: Optional[str] = None


@dataclass()
class ProbeConfig:
    """Configuration describing a probing classifier to train or evaluate.

    Parameters
    ----------
    method:
        The probing method (e.g., `"softmax").
    model_name:
        Human readable name for the probe classifier.
    objective:
        Objective of the probe (e.g., `"answer", "pred", "error").
    component:
        The component of the activations on which to train (e.g., `"res").
    save_name:
        Identifier used to save the probe outputs.
    training_config:
        Hyperparameters controlling probe training.  See
        :class:ProbeTrainingConfig.
    layer:
        Single layer index for which the probe is defined.  `None
        indicates all layers.
    token_positions:
        String of token positions for which the probe is defined.  `None
        indicates all positions.
    """

    method: str
    model_name: str
    objective: str
    component: str
    save_name: str
    training_config: Optional[ProbeTrainingConfig] = None
    layer: Optional[int] = None
    token_positions: Optional[int] = None

    @classmethod
    def resolve_nested(
        cls,
        method: str,
        model_name: str,
        objectives: Union[str, Sequence[str]],
        components: Union[str, Sequence[str]],
        save_name: str,
        layers: Optional[Union[int, Sequence[Optional[int]]]] = None,
        token_positions: Optional[Union[str, Sequence[str]]] = None,
        training_config: Optional[ProbeTrainingConfig] = None,
    ) -> List[ProbeConfig]:
        """Generate a list of probe configurations from parameter grids.

        For every combination of objective, component, layer and token position
        this method returns a :class:ProbeConfig.  Singular values are
        treated as one‑element sequences.

        Parameters
        ----------
        method:
            The probe method.
        model_name:
            Human readable name of the classifier.
        objectives:
            Single objective or sequence of objectives.
        components:
            Single component or sequence of components.
        save_name:
            Save identifier to reuse for all configurations.
        layers:
            Single layer index or sequence of indices.  `None means no
            layering constraint.
        token_positions:
            String of token positions for which the probe is defined.  `None means
            no position constraint.
        training_config:
            Training hyperparameters.

        Returns
        -------
        List[ProbeConfig]
            One probe configuration per combination.
        """

        def _to_list(value: Union[int, Sequence[Optional[int]], None]) -> List[Optional[int]]:
            if value is None:
                return [None]
            if isinstance(value, int):
                return [value]
            return list(value)

        obj_list = objectives if isinstance(objectives, (list, tuple)) else [objectives]
        comp_list = components if isinstance(components, (list, tuple)) else [components]
        layer_list = _to_list(layers)
        pos_list = _to_list(token_positions)

        configs: List[ProbeConfig] = []
        for obj, comp, layer, pos in product(obj_list, comp_list, layer_list, pos_list):
            configs.append(
                cls(
                    method=method,
                    model_name=model_name,
                    objective=obj,
                    component=comp,
                    save_name=save_name,
                    training_config=training_config,
                    layer=layer,
                    token_positions='_'.join(map(str, pos)) if isinstance(pos, list) else pos,
                )
            )
        return configs


@dataclass()
class VectorConfig:
    """Configuration for vectors used in steering experiments."""

    method: Optional[str] = None
    model_name: Optional[str] = None
    save_name: Optional[str] = None
    objective: Optional[str] = None
    component: Optional[str] = None
    token_position: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "model_name": self.model_name,
            "save_name": self.save_name,
            "objective": self.objective,
            "component": self.component,
            "token_position": self.token_position,
        }

@dataclass()
class SteeringModuleConfig:
    """Configuration of a steering module applied in the intervention."""

    method: str
    save_name: str
    model_name: Optional[str] = None
    component: Optional[str] = None


@dataclass()
class InterventionLayerConfig:
    """Configuration specifying a set of layer indices for interventions."""

    layers: Sequence[int]


@dataclass()
class SteeringConfig:
    """Configuration describing a steering intervention.

    Parameters
    ----------
    method:
        The steering strategy (e.g., `"vector_steering").
    module_name:
        Name identifying the steering module or model.  Some older
        configurations use the key `"model_name" instead; both are
        accepted and normalized to this field.
    save_name:
        Identifier used when saving outputs.
    multiplier:
        Scaling applied to the steering vector or module output.
    intervention_layers:
        An :class:InterventionLayerConfig specifying which layers the
        intervention is applied to.
    save_activation:
        Whether to record hidden activations during steering.  Defaults to
        `False.
    component:
        The component on which to apply steering (e.g. `"res" or "all").
    vector:
        Optional :class:VectorConfig describing a precomputed steering
        vector.  Mutually exclusive with `module.
    module:
        Optional :class:SteeringModuleConfig describing a trained steering
        module.  Mutually exclusive with `vector.
    dataset_config:
        Optional :class:DatasetConfig describing data used in the steering
        experiment.  Provided for convenience when constructing from a
        dictionary.
    """

    method: str
    module_name: Optional[str]
    save_name: str
    multiplier: float
    intervention_layers: InterventionLayerConfig
    w: Optional[float] = None
    beta: Optional[float] = None
    target_projection: Optional[float] = None
    save_activation: Optional[bool] = False
    component: Optional[str] = None
    vector_config_list: Optional[Sequence[VectorConfig]] = field(default_factory=list)
    module: Optional[SteeringModuleConfig] = None
    dataset_config: Optional[DatasetConfig] = None

    # ------------------------------------------------------------------
    # Legacy attribute names
    #
    # The original implementation stored an attribute called
    # `intervention_layer on :class:SteeringConfig.  To maintain
    # backwards compatibility we alias that name to the current
    # `intervention_layers field.
    # ------------------------------------------------------------------
    @property
    def intervention_layer(self) -> InterventionLayerConfig:
        return self.intervention_layers

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SteeringConfig":
        """Construct a :class:SteeringConfig from a plain dictionary.

        The dictionary keys may originate from a YAML configuration file and
        therefore might use names that differ slightly from the dataclass
        fields.  This helper resolves those keys and instantiates the
        appropriate nested dataclasses.  Unknown keys are ignored.

        Parameters
        ----------
        data:
            Dictionary containing steering configuration fields.

        Returns
        -------
        SteeringConfig
            The resulting configuration object.

        Raises
        ------
        ValueError
            If neither a vector nor a module configuration is provided.
        """

        # Extract basic fields with backwards compatibility for
        # `model_name → module_name and intervention_layer → intervention_layers.
        method = data.get("method")
        module_name = data.get("module_name") or data.get("model_name")
        save_name = data.get("save_name")
        multiplier = data.get("multiplier", 1.0)
        w = data.get("w", 1.0)
        beta = data.get("beta", 0.0)
        target_projection = data.get("target_projection", None)

        # Intervention layers: accept either a dict {'layers': [...]} or a raw list
        layers_spec = data.get("intervention_layers") or data.get("intervention_layer")
        if isinstance(layers_spec, dict):
            layers_seq = layers_spec.get("layers")
        else:
            layers_seq = layers_spec
        if layers_seq is None:
            raise ValueError("intervention_layers must be specified for SteeringConfig")
        intervention_layers = InterventionLayerConfig(layers=list(layers_seq))

        save_activation = data.get("save_activation", False)
        component = data.get("component")

        # Parse vector(s).  Accept a single dictionary or a list of dicts.
        vector_data = data.get("vector_config_list")
        vector_config_list: Optional[Sequence[VectorConfig]] = None
        if vector_data:
            if isinstance(vector_data, list):
                for vec in vector_data:
                    if isinstance(vec, dict):
                        if vector_config_list is None:
                            vector_config_list = []
                        vector_config_list.append(VectorConfig(**vec))
            elif isinstance(vector_data, dict):
                vector_config_list = [VectorConfig(**vector_data)]
        # Parse module if provided
        module_data = data.get("module") or data.get("module_config")
        module_config: Optional[SteeringModuleConfig] = None
        if module_data:
            module_config = SteeringModuleConfig(**module_data)

        dataset_cfg_data = data.get("dataset_config")
        dataset_config: Optional[DatasetConfig] = None
        if dataset_cfg_data:
            # `dataset_config keys must exactly match DatasetConfig fields
            dataset_config = DatasetConfig(**dataset_cfg_data)

        if not vector_config_list and module_config is None:
            raise ValueError(
                "Either a vector or a module definition must be provided for SteeringConfig"
            )

        return cls(
            method=method,
            module_name=module_name,
            save_name=save_name,
            multiplier=multiplier,
            w=w,
            beta=beta,
            target_projection=target_projection,
            intervention_layers=intervention_layers,
            save_activation=save_activation,
            component=component,
            vector_config_list=vector_config_list,
            module=module_config,
            dataset_config=dataset_config,
        )

    def make_steering_modules_id(self) -> str:
        """Create an identifier summarizing the steering vector or module.

        When a module is present the identifier combines its save name and
        method.  When only a vector is present the identifier combines the
        vector save name, method and objective.  An exception is raised if
        neither is available.

        Returns
        -------
        str
            The composed identifier.

        Raises
        ------
        ValueError
            If neither a module nor a vector is defined.
        """

        if self.module is not None:
            return f"{self.module.save_name}_{self.module.method}"
        if self.vector_config_list:
            vectors_parts = []
            for vec in self.vector_config_list:
                parts = []
                if vec.model_name:
                    parts.append(vec.model_name)
                if vec.save_name:
                    parts.append(vec.save_name)
                if vec.method:
                    parts.append(vec.method)
                if vec.objective:
                    parts.append(vec.objective)
                vectors_parts.append("-".join(parts))
            return "_".join(vectors_parts)
        raise ValueError("No module or vector defined in SteeringConfig")

    def make_intervention_layers_id(self) -> str:
        """Compute a compact identifier summarizing the intervention layers.

        Consecutive layer indices are collapsed into ranges (e.g., `[1,2,3]
        becomes `"layer_1-3") and non‑consecutive indices are separated
        by underscores (e.g., `[1,3,4] becomes "layer_1_layer_3-4").
        An empty sequence produces `"all_layers".

        Returns
        -------
        str
            String representation of the layer selection.
        """

        layers = sorted(self.intervention_layers.layers)
        if not layers:
            return "all_layers"

        ranges: List[str] = []
        start = prev = layers[0]
        for layer in layers[1:]:
            if layer == prev + 1:
                prev = layer
            else:
                if start == prev:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{prev}")
                start = prev = layer
        # Append the last range
        if start == prev:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{prev}")

        return "_".join(f"layer_{r}" for r in ranges)

    @classmethod
    def resolve_nested(
        cls,
        method: str,
        module_name: Optional[str],
        save_name: str,
        multipliers: Union[float, Sequence[float]],
        w: Optional[float],
        beta: Optional[float],
        target_projection: Optional[float],
        intervention_layers: Union[Sequence[int], Sequence[Sequence[int]]],
        save_activation: bool = False,
        components: Optional[List[str]] = None,
        vector_config_list: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
        module: Optional[SteeringModuleConfig] = None,
    ) -> List[SteeringConfig]:
        """Expand steering configurations over multipliers, layers and vectors.

        Parameters
        ----------
        method:
            The steering method.
        module_name:
            Name of the steering module or model family.  May be `None.
        save_name:
            Base identifier used for saving outputs.
        multipliers:
            Single multiplier or a sequence of multipliers.
        intervention_layers:
            A sequence of layer indices or a sequence of such sequences.
        save_activation:
            Whether to save activations.  Defaults to `False.
        component:
            Component of the activations to apply steering on.  May be `None.
        vectors:
            Single vector specification or a sequence of vector specifications.
            Each specification is a dictionary of fields accepted by
            :class:VectorConfig.  If omitted or empty, `module must be
            provided.
        module:
            A preconstructed :class:SteeringModuleConfig.  If provided the
            steering will use this module instead of a vector.  When both
            `vectors and module are provided the vectors take
            precedence.

        Returns
        -------
        List[SteeringConfig]
            A list of steering configurations covering all combinations.
        """

        # Normalize scalar arguments to lists for Cartesian product
        mult_list = [multipliers] if isinstance(multipliers, (int, float)) else list(multipliers)
        # Intervention layers may be a list of ints or a list of lists
        if not intervention_layers:
            raise ValueError("intervention_layers must not be empty in resolve_nested")
        if isinstance(intervention_layers[0], int):  # type: ignore[index]
            layer_groups = [intervention_layers]  # type: ignore[list-item]
        else:
            layer_groups = [list(group) for group in intervention_layers]  # type: ignore[list-item]

        configs: List[SteeringConfig] = []
        for mult, layers, component in product(mult_list, layer_groups, components):
            vector_config_list = [
                VectorConfig(**vec) if isinstance(vec, dict) else vec
                for vec in vector_config_list or []
            ]
            # Determine if we should use the provided module when vector is absent
            module_config = None if vector_config_list else module
            configs.append(
                cls(
                    method=method,
                    module_name=module_name,
                    save_name=save_name,
                    multiplier=float(mult),
                    w=w,
                    beta=beta,
                    target_projection=target_projection,
                    intervention_layers=InterventionLayerConfig(layers=list(layers)),
                    save_activation=save_activation,
                    component=component,
                    vector_config_list=vector_config_list,
                    module=module_config,
                )
            )
        return configs


@dataclass
class ExperimentConfig(BaseConfig):
    """Aggregate configuration describing a single experiment.

    This class extends :class:BaseConfig and aggregates all other
    configurations required to run an experiment.  Fields which may be
    omitted (set to `None) are optional and allow for different types
    of experiments (e.g. steering only, probing only, activation
    collection) to be represented.

    Parameters
    ----------
    experiment_type:
        Type of experiment (e.g., `"steering").  Inherited from
        :class:BaseConfig.
    splits:
        Dataset splits applicable at the experiment level.  Inherited from
        :class:BaseConfig.
    output_dir:
        Output directory.  Inherited from :class:BaseConfig.
    llm_config:
        Language model configuration.
    dataset_config:
        Dataset configuration.  May be `None if not applicable.
    generation_config:
        Generation configuration.  May be `None.
    activation_config:
        Activation recording configuration.  May be `None.
    probe_config:
        Probe configuration.  May be `None.
    steering_config:
        Steering configuration.  May be `None.
    steering_training_config:
    """

    llm_config: Optional[LLMConfig] = None
    activation_config: Optional[ActivationConfig] = None
    generation_config_list: Optional[List[GenerationConfig]] = None
    dataset_config_list: Optional[List[DatasetConfig]] = None
    probe_config_list: Optional[List[ProbeConfig]] = None
    steering_config_list: Optional[List[SteeringConfig]] = None
    steering_training_config: Optional[ProbeTrainingConfig] = None
    probe_exp_config: Optional["ExperimentConfig"] = None
    activation_exp_config: Optional["ExperimentConfig"] = None


    def to_dict(self) -> Dict[str, Any]:
        """Convert the experiment configuration into a dictionary.

        This overrides :meth:BaseConfig.to_dict to include all fields
        defined on :class:ExperimentConfig as well as those inherited
        from :class:BaseConfig.
        """
        return super().to_dict()

    # ------------------------------------------------------------------
    # Class methods for constructing experiments
    # ------------------------------------------------------------------

    @classmethod
    def from_sources(
        cls,
        cli_dict: Dict[str, Any],
        cfg_path: Optional[Union[str, Path]] = None,
    ) -> List[ExperimentConfig]:
        """Combine command‑line and file configurations to produce experiments.

        Parameters
        ----------
        cli_dict:
            Dictionary of configuration values provided via the CLI.  Keys
            with a value of `None are ignored.  The key "config" is
            reserved and ignored.
        cfg_path:
            Optional path to a YAML or JSON configuration file.  If
            provided its contents will override values in `cli_dict.

        Returns
        -------
        ExperimentConfig
            A list of fully expanded experiment configurations.
        """

        merged: Dict[str, Any] = {}

        # Load from file if present.  Unknown file types raise an error.
        if cfg_path:
            path = Path(cfg_path)
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")
            with path.open("r", encoding="utf-8") as f:
                if path.suffix.lower() in {".yaml", ".yml"}:
                    file_cfg = yaml.safe_load(f) or {}
                elif path.suffix.lower() == ".json":
                    file_cfg = json.load(f) or {}
                else:
                    raise ValueError(f"Unsupported configuration file type: {path.suffix}")
            merged.update(file_cfg)

        # CLI arguments override file configuration except for None values
        for k, v in cli_dict.items():
            if k == "config" or v is None:
                continue
            merged[k] = v

        # Extract top level fields
        experiment_type = merged.pop("experiment_type", None)
        output_dir = merged.pop("output_dir", None)

        # Build lower level configuration combinations
        config_dict = cls.build_config(merged)

        exp_config_instance = cls(
            experiment_type=experiment_type,
            output_dir=output_dir,
        )

        for key in config_dict:
            setattr(exp_config_instance, key, config_dict[key])

        return exp_config_instance

    @classmethod
    def build_config(cls, raw_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand nested experiment configuration sections into combinations.

        This method interprets the provided dictionary as containing
        configuration subsections (e.g., `llm_config, generation_config,
        `dataset_configs) and enumerates all combinations of those
        subsections.  Any additional keys encountered in the dictionary are
        assumed to refer to nested experiment configurations and will be
        processed recursively via :meth:from_sources.

        Parameters
        ----------
        raw_cfg:
            A dictionary representing the merged configuration after
            combining CLI and file inputs.  The dictionary may contain a
            nested section under the key `"exp_config" – if present
            configurations will be read from this subdictionary instead of
            the top level.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries where each dictionary contains a unique
            combination of configuration objects.  The keys correspond to
            the names of the configuration sections (e.g., `"llm_config").
        """
        cfg = dict(raw_cfg)

        # Parse the LLM configuration (single instance)
        llm_data = cfg.pop("llm_config", {})
        llm_config = LLMConfig(**llm_data) if llm_data else None

        # Parse generation configurations (potentially multiple instances)
        gen_data = cfg.pop("generation_config", {})
        gen_configs: List[GenerationConfig]
        if gen_data:
            gen_configs = GenerationConfig.resolve_nested(
                decoding_mode=gen_data.get("decoding_mode"),
                polarity=gen_data.get("polarity"),
                temperature=gen_data.get("temperature"),
                top_p=gen_data.get("top_p"),
                do_sample=gen_data.get("do_sample"),
                max_new_tokens=gen_data.get("max_new_tokens"),
            )
        else:
            gen_configs = [None]  # type: ignore[list-item]

        # Parse dataset configurations (list of dicts)
        ds_dicts: Sequence[Dict[str, Any]] = cfg.pop("dataset_configs", []) or []
        dataset_configs: List[Optional[DatasetConfig]] = []
        if ds_dicts:
            for ds_dict in ds_dicts:
                dataset_configs.extend(
                    DatasetConfig.resolve_nested(
                        datasets=ds_dict.get("datasets"),
                        splits=ds_dict.get("splits"),
                        num_option=ds_dict.get("num_option"),
                        random_seed=ds_dict.get("random_seed"),
                        type=ds_dict.get("type"),
                        num_permutations=ds_dict.get("num_permutations"),
                        prompt_configs=ds_dict.get("prompt_configs", {}),
                    )
                )
        else:
            dataset_configs = [None]

        # Activation configuration (single instance)
        act_data = cfg.pop("activation_config", {}) or {}
        activation_config = (
            ActivationConfig(
                layers=act_data.get("layers"),
                token_positions=act_data.get("token_positions"),
                components=act_data.get("components")
            )
            if act_data
            else None
        )

        # Probe configurations (potentially multiple instances)
        probe_data = cfg.pop("probe_config", {}) or {}
        probe_configs: List[Optional[ProbeConfig]]
        if probe_data:
            probe_configs = ProbeConfig.resolve_nested(
                method=probe_data.get("method"),
                model_name=probe_data.get("model_name"),
                objectives=probe_data.get("objectives"),
                components=probe_data.get("components"),
                save_name=probe_data.get("save_name"),
                layers=probe_data.get("layers"),
                token_positions=probe_data.get("token_positions"),
                training_config=ProbeTrainingConfig(**probe_data.get("training_config", {})),
            )
        else:
            probe_configs = [None]

        # Steering configurations (potentially multiple instances)
        steering_data = cfg.pop("steering_config", {}) or {}
        steering_configs: List[Optional[SteeringConfig]]
        if steering_data:
            # Prepare an optional SteeringModuleConfig.  Only instantiate when a
            # non‑null mapping is provided under either `module or
            # `module_config.
            module_conf = None
            module_data = None
            if "module" in steering_data:
                module_data = steering_data.get("module")
            elif "module_config" in steering_data:
                module_data = steering_data.get("module_config")
            if isinstance(module_data, dict) and module_data:
                module_conf = SteeringModuleConfig(**module_data)

            steering_configs = SteeringConfig.resolve_nested(
                method=steering_data.get("method"),
                module_name=steering_data.get("module_name") or steering_data.get("model_name"),
                save_name=steering_data.get("save_name"),
                multipliers=steering_data.get("multipliers"),
                w=steering_data.get("w"),
                beta=steering_data.get("beta"),
                target_projection=steering_data.get("target_projection"),
                intervention_layers=[d.get("layers") for d in steering_data.get("intervention_layers", [])],
                save_activation=steering_data.get("save_activation", False),
                components=steering_data.get("components"),
                vector_config_list=steering_data.get("vector_config_list"),
                module=module_conf,
            )
        else:
            steering_configs = [None]

        # Steering training configuration (currently unused but preserved)
        steering_train_conf = None
        if "steering_training_config" in cfg:
            st_data = cfg.pop("steering_training_config")
            steering_train_conf = ProbeTrainingConfig(**st_data)


        # Cross product of all non‑nested configurations
        exp_config: List[Dict[str, Any]] = {
            "llm_config": llm_config,
            "activation_config": activation_config,
            "generation_config_list": gen_configs,
            "dataset_config_list": dataset_configs,
            "probe_config_list": probe_configs,
            "steering_config_list": steering_configs,
            "steering_training_config": steering_train_conf
        }

        # Handle nested experiment configurations recursively
        nested_keys = list(cfg.keys())
        for key in nested_keys:
            sub_dict = cfg.pop(key)
            # Each sub_dict should be interpreted as a separate config
            exp_config[key] = cls.from_sources(sub_dict)

        return exp_config
