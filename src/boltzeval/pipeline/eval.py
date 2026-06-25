# orchestrator for running the sample- and energy-based evaluation

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, asdict

from typing import Any, Literal, Optional, TypeAlias
import warnings
import numpy as np

from boltzeval.utils.histogram import Histogram
from boltzeval.utils.pdf import PdfBuffer, pdf_to_wandb_image
from boltzeval.utils.shape_utils import squeeze_last_dim

import logging

logger = logging.getLogger(__name__)

ValueType = float | int | PdfBuffer | Histogram | Any

EvalField: TypeAlias = Literal[
    "samples_true",
    "samples_pred",
    "true_samples_target_log_prob",
    "pred_samples_target_log_prob",
    "true_samples_model_log_prob",
    "pred_samples_model_log_prob",
    "trajs_true",
    "trajs_pred",
]


@dataclass
class EvalData:
    """
    Container for evaluation inputs.

    Shape conventions
    -----------------
    Samples:
        (B, D)

        where B = number of samples (true and pred may differ),
              D = feature dimension (must be identical across all sample fields).

        Molecular/coordinate samples of shape (B, n_atoms, 3)
        are automatically flattened to (B, n_atoms * 3).

    Log-probabilities:
        (B,)

        Inputs of shape (B, 1) are automatically squeezed to (B,).

        Each log-prob array must match the batch size of its corresponding samples.

    Trajectories:
        (T, B, D)

        where T = number of trajectories,
              B = number of frames per trajectory,
              D = feature dimension (must match sample dimension).

    Attributes
    ----------
    samples_true : np.ndarray, optional
        Shape (B_true, D). Samples from the reference/true distribution.

    samples_pred : np.ndarray, optional
        Shape (B_pred, D). Predicted/generated samples (e.g., from a model).

    true_samples_target_log_prob : np.ndarray, optional
        Shape (B_true,). log p_target(x_true).

    pred_samples_target_log_prob : np.ndarray, optional
        Shape (B_pred,). log p_target(x_pred).

    true_samples_model_log_prob : np.ndarray, optional
        Shape (B_true,). log p_model(x_true).

    pred_samples_model_log_prob : np.ndarray, optional
        Shape (B_pred,). log p_model(x_pred).

    trajs_true : np.ndarray | None
        Shape (T_true, B_true, D). Reference trajectories.

    trajs_pred : np.ndarray | None
        Shape (T_pred, B_pred, D). Predicted trajectories.
        Predicted trajectories for trajectory-based evaluations.

    Notes
    -----
    - Log-probability arrays must match the corresponding sample batch size.
    """

    # This is used internally to provide better error messages
    _restricted_access: bool = False
    _eval_cls: Optional[str] = None

    samples_true: Optional[np.ndarray] = None
    samples_pred: Optional[np.ndarray] = None
    true_samples_target_log_prob: Optional[np.ndarray] = None

    pred_samples_target_log_prob: Optional[np.ndarray] = None
    true_samples_model_log_prob: Optional[np.ndarray] = None
    pred_samples_model_log_prob: Optional[np.ndarray] = None

    trajs_true: Optional[np.ndarray] = None
    trajs_pred: Optional[np.ndarray] = None

    def fits_requirements(self, requirements: list[EvalField]) -> bool:
        return len(self.get_missing_requirements(requirements)) == 0

    def get_missing_requirements(
        self, requirements: list[EvalField]
    ) -> list[EvalField]:
        return [r for r in requirements if getattr(self, r, None) is None]

    def __post_init__(self):
        populated_fields = self._get_populated_fields()

        # Remove potential single trailing ones in the fields
        for k, v in populated_fields.items():
            if "log_prob" in k:
                setattr(self, k, squeeze_last_dim(v))

        # Flatten molecular samples of shape (B, n_atoms, 3)
        for k, v in populated_fields.items():
            if k in ["samples_true", "samples_pred"] and v.ndim == 3:
                setattr(self, k, v.reshape((v.shape[0], -1)))

        # Flatten trajectories if they contain atomic coordinates:
        # (T, B, n_atoms, 3) -> (T, B, n_atoms * 3)
        for k, v in populated_fields.items():
            if k in ["trajs_true", "trajs_pred"] and v.ndim == 4:
                setattr(self, k, v.reshape(v.shape[0], v.shape[1], -1))

        # Fetch potentially updated fields
        populated_fields = self._get_populated_fields()

        self._check_type(populated_fields)
        self._check_same_batch_size()
        self._check_sample_shapes(populated_fields)

    def _get_populated_fields(self) -> dict[str, np.ndarray]:
        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None
            if not k.startswith("_")
        }

    def _check_type(self, populated_fields: dict[str, np.ndarray]):
        invalid = [
            f"{k}: {type(v).__name__}"
            for k, v in populated_fields.items()
            if not isinstance(v, np.ndarray)
        ]

        if invalid:
            raise TypeError(
                f"The following fields must be np.ndarray but are not: {invalid}"
            )

    def _check_same_batch_size(self):
        def _check_pair(
            samples: Optional[np.ndarray], log_probs: Optional[np.ndarray], name: str
        ):
            if samples is None or log_probs is None:
                return
            if samples.shape[0] != log_probs.shape[0]:
                raise ValueError(
                    f"Batch size mismatch for {name}: "
                    f"samples batch={samples.shape[0]}, log_probs batch={log_probs.shape[0]}"
                )

        _check_pair(
            self.samples_true,
            self.true_samples_target_log_prob,
            "true_samples_target_log_prob",
        )
        _check_pair(
            self.samples_true,
            self.true_samples_model_log_prob,
            "true_samples_model_log_prob",
        )

        _check_pair(
            self.samples_pred,
            self.pred_samples_target_log_prob,
            "pred_samples_target_log_prob",
        )
        _check_pair(
            self.samples_pred,
            self.pred_samples_model_log_prob,
            "pred_samples_model_log_prob",
        )

    def _check_sample_shapes(self, populated_fields: dict[str, np.ndarray]):
        sample_shapes = {
            k: v.shape
            for k, v in populated_fields.items()
            if k in ["samples_true", "samples_pred"]
        }
        if not all([len(shape) == 2 for shape in sample_shapes.values()]):
            invalid = {k: s for k, s in sample_shapes.items() if len(s) != 2}
            raise ValueError(
                f"All sample arrays must be 2D (batch_size, dim). "
                f"Found invalid shapes: {invalid}"
            )

        cond1 = len(sample_shapes) > 0
        cond2 = len(set([shape[1] for shape in sample_shapes.values()])) != 1
        if cond1 and cond2:
            raise ValueError(
                f"Dimension mismatch: All samples must have the same dimension (index 1). "
                f"Detected dimensions at index 1: { {k: s[1] for k, s in sample_shapes.items()} }"
            )

    def get_required_fields(self, requirements: list[str]):
        populated_fields = self._get_populated_fields()
        required_fields = {k: populated_fields[k] for k in requirements}
        return required_fields

    def copy_required(self, requirements: list[str], eval_cls: type["EvaluationNode"]):
        required_fields = self.get_required_fields(requirements)
        data = EvalData(**required_fields, _eval_cls=eval_cls)
        data._restricted_access = True
        return data

    def __getattribute__(self, name):
        # Prevent access to attributes not listed in the requirements.
        # This avoids subtle bugs and makes dependencies explicit.

        restricted_access = object.__getattribute__(self, "_restricted_access")

        if restricted_access:
            try:
                value = object.__getattribute__(self, name)
            except AttributeError:
                raise

            if value is None:
                eval_cls: EvaluationNode = object.__getattribute__(self, "_eval_cls")
                eval_cls_name = eval_cls.__name__
                requirements = eval_cls.requirements

                if name not in requirements:
                    raise AttributeError(
                        f"The attribute '{name}' was requested by '{eval_cls_name}' without being an explicit requirement, which is not allowed in restricted access mode. Consider adding '{name}' to '{eval_cls_name}.requirements'."
                    )
                else:
                    raise AttributeError("Something went fatally wrong")

        # IMPORTANT: delegate to the base implementation
        return super().__getattribute__(name)


class EvaluationNode(ABC):
    requirements: list[EvalField] = []

    def __init__(self):
        super().__init__()
        # Verify that all requirements are actually valid keys in the dataclass
        valid_fields = {f.name for f in fields(EvalData)}
        if not set(self.requirements).issubset(valid_fields):
            raise TypeError(f"Invalid requirement in {self.__class__.__name__}")

    @abstractmethod
    def _eval(self, data: EvalData) -> dict[str, ValueType]:
        raise NotImplementedError

    def eval(self, data: EvalData, skip_on_missing_data: bool = False):
        missing = data.get_missing_requirements(self.requirements)
        if missing:
            if skip_on_missing_data:
                infill = "this field is" if len(missing) == 1 else "these fields are"
                warnings.warn(
                    f"Evaluation Skipped: '{self.__class__.__name__}' requires [{', '.join(missing)}], "
                    f"but {infill} missing from the provided EvalData.",
                    UserWarning,
                    stacklevel=2,
                )
                return None
            else:
                raise ValueError(
                    f"Evaluation failed: {self.__class__.__name__} requires the following "
                    f"fields to be populated, but they are currently None: {', '.join(missing)}."
                )

        # prevent access to attribute that are not explicitely required.
        # This avoids subtle bugs and makes dependencies explicit.
        safe_data = data.copy_required(self.requirements, type(self))
        return self._eval(safe_data)


def _to_list(
    obj: (
        list[EvaluationNode | tuple[EvaluationNode, str]]
        | EvaluationNode
        | tuple[EvaluationNode, str]
    ),
) -> list[tuple[EvaluationNode, str | None]]:
    if isinstance(obj, EvaluationNode):
        l = [(obj, None)]
        return l

    if isinstance(obj, tuple):
        l = [obj]
        return l

    l = []
    for o in obj:
        if isinstance(o, EvaluationNode):
            l.append((o, None))
        elif isinstance(o, tuple):
            l.append(o)
        else:
            raise ValueError(f"Invalid evaluation node of type '{type(o).__name__}'")
    return l


def _prefix_dict(metrics: dict[str, ValueType], prefix: str | None):
    if prefix:  # None or empty string
        prefix = prefix + "/"
    else:
        prefix = ""

    return {(prefix + k): v for k, v in metrics.items()}


def update_dict_with_id(target: dict, new_data: dict, idx: int) -> dict:
    """
    Update `target` with `new_data`.
    If a key already exists in `target`, append `unique_id` to the key.
    """
    for key, value in new_data.items():
        if key in target:
            new_key = f"{key}_eval_idx_{idx}"
            # Ensure uniqueness even if that key also exists
            i = 1
            while new_key in target:
                new_key = f"{key}_eval_idx_{idx}_{i}"
                i += 1
            target[new_key] = value
        else:
            target[key] = value

    return target


def run_eval(
    data: EvalData,
    *,
    pipeline: list[EvaluationNode | tuple[EvaluationNode, str]] = [],
    skip_on_missing_data: bool = True,
    skip_on_fail: bool = False,
) -> dict[str, ValueType]:
    if len(pipeline) == 0:
        logger.warning("Empty evaluation pipeline -> running no evaluations")

    eval_list = _to_list(pipeline)

    all_metrics = {}

    logger.info(f"Start evaluation with {len(eval_list)} nodes...")
    for i, (eval, prefix) in enumerate(eval_list):
        logger.info(f"Run eval node {i} ({eval.__class__.__name__})...")

        try:
            metrics = eval.eval(data, skip_on_missing_data=skip_on_missing_data)
        except Exception as e:
            if skip_on_fail:
                metrics = None
                warnings.warn(
                    f"Evaluation Skipped: '{eval.__class__.__name__}' failed to evaluate (error msg: {e})",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                raise e

        if metrics is None:
            continue
        metrics = _prefix_dict(metrics, prefix)

        update_dict_with_id(all_metrics, metrics, i)

    logger.info(f"Finished evaluation.")

    return all_metrics


def make_wandb_compatible(
    data: dict[str, ValueType], dpi: int = 100, update_keys: bool = True
):
    """
    Convert all elements in the dict into wandb-compatible items (e.g., pdf (in the form of a binary buffer) -> wandb.Image).
    This function requires the installation of the pip `wandb` package.
    """

    def transform(k: str, v):
        if isinstance(v, PdfBuffer):
            if update_keys:
                k = k.replace("pdf", "vis")
            v = pdf_to_wandb_image(v, dpi=dpi)

        return k, v

    def is_valid(v):
        return isinstance(v, (float, int, PdfBuffer))

    return dict((transform(k, v) for k, v in data.items() if is_valid(v)))


def get_scalar_metrics(data: dict[str, ValueType]):
    def _is_float_like(v):
        try:
            float(v)
        except:
            return False
        return True

    return {k: float(v) for k, v in data.items() if _is_float_like(v)}


def get_histograms(
    data: dict[str, ValueType],
) -> dict[str, Histogram]:
    return {k: v for k, v in data.items() if isinstance(v, Histogram)}


def get_pdfs(data: dict[str, ValueType]) -> dict[str, PdfBuffer]:
    return {k: v for k, v in data.items() if isinstance(v, PdfBuffer)}
