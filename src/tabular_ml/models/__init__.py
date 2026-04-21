"""Model training, tuning, and ensemble utilities.

This module intentionally avoids importing heavy optional ML dependencies at
package import time. Callers can import lightweight helpers such as
``tabular_ml.models.evaluation`` on machines where native tree-library runtime
dependencies are not installed.
"""

from importlib import import_module
from typing import Any


__all__ = [
    "BlendingEnsemble",
    "StackingEnsemble",
    "build_model",
    "compute_metrics",
    "find_optimal_threshold",
    "format_results_table",
    "run_ensemble",
    "run_full_training",
    "train_and_evaluate",
    "tune_model",
]

_EXPORT_MAP = {
    "BlendingEnsemble": ("tabular_ml.models.ensemble", "BlendingEnsemble"),
    "StackingEnsemble": ("tabular_ml.models.ensemble", "StackingEnsemble"),
    "build_model": ("tabular_ml.models.tuning", "build_model"),
    "compute_metrics": ("tabular_ml.models.evaluation", "compute_metrics"),
    "find_optimal_threshold": (
        "tabular_ml.models.evaluation",
        "find_optimal_threshold",
    ),
    "format_results_table": ("tabular_ml.models.evaluation", "format_results_table"),
    "run_ensemble": ("tabular_ml.models.run_ensemble", "run_ensemble"),
    "run_full_training": ("tabular_ml.models.train_all", "run_full_training"),
    "train_and_evaluate": ("tabular_ml.models.trainer", "train_and_evaluate"),
    "tune_model": ("tabular_ml.models.tuning", "tune_model"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported attributes lazily."""
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORT_MAP[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports in interactive environments."""
    return sorted(set(globals()) | set(__all__))
