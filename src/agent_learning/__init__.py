"""Native reinforcement learning SDK for AI agents.

Replaces the agent-lightning LLM fine-tuning loop with a fully
native, in-process learner. The SDK is organised into five layers:

- ``agent_learning.types``    - durable record types (``Episode``,
  ``Reward``, ``PolicySnapshot``, ...).
- ``agent_learning.storage``  - pluggable persistence (Cosmos DB,
  local file system, and in-memory).
- ``agent_learning.metrics``  - judge-based metrics that wrap the
  Azure AI Evaluation evaluators for Intent Resolution, Task
  Adherence, and Task Completion.
- ``agent_learning.rewards``  - reward shaping + persistence.
- ``agent_learning.policy``   - discrete softmax bandit policy.
- ``agent_learning.learners`` - REINFORCE-with-baseline learner.
- ``agent_learning.training`` - end-to-end :class:`LearningRunner`.

Quick start::

    from agent_learning import (
        Action,
        EpisodeCapture,
        LearningRunner,
        SoftmaxPolicy,
    )

    actions = [Action(id="prompt_A"), Action(id="prompt_B")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="dq")

    # Capture
    capture = EpisodeCapture()
    decision = policy.choose()
    ctx = capture.start(
        "Tell me my Q3 sales summary",
        policy_id=policy.snapshot().id,
        policy_version=policy.snapshot().version,
        action_id=decision.action.id,
        action_logprob=decision.logprob,
    )
    # ... agent runs, records tool calls, produces output ...
    episode = capture.end(ctx, assistant_output="...")

    # Train
    runner = LearningRunner(policy=policy)
    run = runner.run_offline_batch("dq", episode_limit=200)
"""

from ._version import __version__
from .capture import CaptureContext, EpisodeCapture, get_capture
from .classifiers import (
    AdherenceJudge,
    Classifier,
    ClassifierResult,
    CompletionJudge,
    IntentJudge,
    RouterClassifier,
)
from .config import (
    CaptureConfig,
    CosmosConfig,
    JudgeConfig,
    LearnerConfig,
    ShapingConfig,
)
from .learners import Learner, LearnerResult, ReinforceLearner
from .metrics import (
    IntentResolutionMetric,
    MetricEvaluator,
    MetricRequest,
    TaskAdherenceMetric,
    TaskCompletionMetric,
    default_metrics,
    evaluate_all,
)
from .policy import ContextualSoftmaxPolicy, Policy, SoftmaxPolicy
from .rewards import RewardShaper, RewardWriter, shape_episode_reward
from .storage import (
    CosmosStore,
    InMemoryStore,
    LearningStore,
    LocalFileStore,
    get_default_store,
)
from .training import LearningRunner
from .types import (
    Action,
    AgentInfo,
    AgentTaskInfo,
    Episode,
    MetricName,
    MetricResult,
    PolicySnapshot,
    Reward,
    RewardSource,
    ToolCall,
    TrainingRun,
    TrainingStatus,
)

__all__ = [
    "Action",
    "AdherenceJudge",
    "AgentInfo",
    "AgentTaskInfo",
    "CaptureConfig",
    "CaptureContext",
    "Classifier",
    "ClassifierResult",
    "CompletionJudge",
    "ContextualSoftmaxPolicy",
    "CosmosConfig",
    "CosmosStore",
    "Episode",
    "EpisodeCapture",
    "InMemoryStore",
    "IntentJudge",
    "IntentResolutionMetric",
    "JudgeConfig",
    "Learner",
    "LearnerConfig",
    "LearnerResult",
    "LearningRunner",
    "LearningStore",
    "LocalFileStore",
    "MetricEvaluator",
    "MetricName",
    "MetricRequest",
    "MetricResult",
    "Policy",
    "PolicySnapshot",
    "ReinforceLearner",
    "Reward",
    "RewardShaper",
    "RewardSource",
    "RewardWriter",
    "RouterClassifier",
    "ShapingConfig",
    "SoftmaxPolicy",
    "TaskAdherenceMetric",
    "TaskCompletionMetric",
    "ToolCall",
    "TrainingRun",
    "TrainingStatus",
    "__version__",
    "default_metrics",
    "evaluate_all",
    "get_capture",
    "get_default_store",
    "shape_episode_reward",
]
