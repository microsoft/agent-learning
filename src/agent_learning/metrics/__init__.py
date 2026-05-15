"""Judge-based evaluation metrics for native RL reward shaping.

Each metric is a thin wrapper around an evaluator from
``azure-ai-evaluation``. The wrapper normalises the raw judge score
into the ``[0, 1]`` range expected by the reward shaper.
"""

from .base import MetricEvaluator, MetricRequest
from .intent_resolution import IntentResolutionMetric
from .task_adherence import TaskAdherenceMetric
from .task_completion import TaskCompletionMetric
from .registry import default_metrics, evaluate_all

__all__ = [
    "IntentResolutionMetric",
    "MetricEvaluator",
    "MetricRequest",
    "TaskAdherenceMetric",
    "TaskCompletionMetric",
    "default_metrics",
    "evaluate_all",
]
