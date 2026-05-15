"""Quick-start example: native RL loop on an in-memory store.

Run this file with ``python examples/quickstart.py``. It uses
the in-memory store and a stubbed metric evaluator so it works
without any Azure credentials.
"""

from __future__ import annotations

import random

from agent_learning.config import LearnerConfig, ShapingConfig
from agent_learning.learners import ReinforceLearner
from agent_learning.metrics.base import MetricRequest
from agent_learning.policy import SoftmaxPolicy
from agent_learning.rewards import RewardShaper, RewardWriter
from agent_learning.storage import InMemoryStore
from agent_learning.training import LearningRunner
from agent_learning.types import (
    Action,
    Episode,
    MetricName,
    MetricResult,
)


def main() -> None:
    rng = random.Random(42)
    store = InMemoryStore()

    # Define two candidate prompt strategies as actions
    actions = [
        Action(id="concise", description="Short, direct prompt"),
        Action(id="detailed", description="Verbose chain-of-thought prompt"),
    ]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="demo", rng=rng)
    store.store_policy(policy.snapshot())

    # Simulate one round of agent interactions
    for i in range(40):
        decision = policy.choose()
        # Pretend the "detailed" action returns better answers on this task.
        success = 1.0 if decision.action.id == "detailed" else 0.2
        ep = Episode(
            id=f"ep-{i}",
            agent_id="demo",
            user_input="Summarize Q3 KPIs",
            assistant_output="...summary...",
            policy_id=policy.snapshot().id,
            policy_version=policy.snapshot().version,
            action_id=decision.action.id,
            action_logprob=decision.logprob,
            request_latency_ms=2000,
            metadata={"_synthetic_reward": success},
        )
        store.store_episode(ep)

    # Stubbed metric evaluator (no Azure calls)
    def synthetic_metrics(episode: Episode):
        score = episode.metadata.get("_synthetic_reward", 0.0)
        return [
            MetricResult(metric=MetricName.INTENT_RESOLUTION, score=score, normalized=score, status="completed"),
            MetricResult(metric=MetricName.TASK_ADHERENCE, score=score, normalized=score, status="completed"),
            MetricResult(metric=MetricName.TASK_COMPLETION, score=score, normalized=score, status="completed"),
        ]

    class _DemoRunner(LearningRunner):
        def evaluate_episode(self, episode: Episode):
            return synthetic_metrics(episode)

    runner = _DemoRunner(
        store=store,
        policy=policy,
        metrics=[],
        shaper=RewardShaper(ShapingConfig()),
        writer=RewardWriter(store),
        learner=ReinforceLearner(LearnerConfig(learning_rate=0.4, entropy_bonus=0.01)),
    )

    before_probs = policy.probabilities()
    run = runner.run_offline_batch("demo", episode_limit=200)
    after_probs = policy.probabilities()

    print("Run status:        ", run.status.value)
    print("Episodes used:     ", run.metrics["episodes_used"])
    print("Mean reward:       ", round(run.metrics["mean_reward"], 4))
    print("Probs before:      ", {a.id: round(p, 3) for a, p in zip(policy.actions(), before_probs)})
    print("Probs after:       ", {a.id: round(p, 3) for a, p in zip(policy.actions(), after_probs)})


if __name__ == "__main__":
    main()
