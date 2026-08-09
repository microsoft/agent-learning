"""Score-based evaluation metrics for native RL reward shaping.

Metrics use on-device stdlib scorers by default. Configured Azure AI
evaluators remain available for remote LLM scoring.
"""

from .base import MetricEvaluator, MetricRequest
from .intent_resolution import IntentResolutionMetric
from .local import LocalScorerMetric
from .task_adherence import TaskAdherenceMetric
from .task_completion import TaskCompletionMetric
from .registry import default_metrics, evaluate_all

__all__ = [
    "IntentResolutionMetric",
    "LocalScorerMetric",
    "MetricEvaluator",
    "MetricRequest",
    "TaskAdherenceMetric",
    "TaskCompletionMetric",
    "default_metrics",
    "evaluate_all",
]
