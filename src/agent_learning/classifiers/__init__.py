"""Pure-Python classifiers for RL reward shaping pipelines.

The reward shaper in :mod:`agent_learning.rewards.shaping` combines
multiple classifier outputs into a scalar reward. This sub-package
ships the classifier implementations themselves:

- :class:`agent_learning.classifiers.router.RouterClassifier` —
  multi-class classifier over a context vector. Used to route an
  incoming request to one of a known set of class ids before the
  policy chooses an action.
- :class:`agent_learning.classifiers.scorers.intent.IntentScorer`,
  :class:`~agent_learning.classifiers.scorers.adherence.AdherenceScorer`,
  and
  :class:`~agent_learning.classifiers.scorers.completion.CompletionScorer`
  — binary ``{pass, fail}`` classifiers over a ``(context, action)``
  pair. Drop-in replacements for LLM-based scorers with the same
  call surface.

All classes here are deterministic, dependency-free (stdlib only),
and JSON-serialisable via ``to_snapshot`` / ``from_snapshot``. The
fit loop is a plain mini-batch logistic regression so the
classifiers are reproducible across environments and easy to ship
through the same image as the rest of the SDK.
"""

from .base import Classifier, ClassifierResult
from .router import RouterClassifier
from .scorers import AdherenceScorer, CompletionScorer, IntentScorer

__all__ = [
  "AdherenceScorer",
    "Classifier",
    "ClassifierResult",
    "CompletionScorer",
    "IntentScorer",
    "RouterClassifier",
]
