"""Tests for the REINFORCE learner.

We assert two important properties:

1. A consistently positive reward for one action shifts its
   logit upwards (and the other action's logit down).
2. Episodes without an aggregate reward or an unknown action are
   skipped without errors.
"""

from __future__ import annotations

import random

from agent_learning.config import LearnerConfig
from agent_learning.learners import ReinforceLearner
from agent_learning.policy import SoftmaxPolicy
from agent_learning.types import Action, Episode, Reward, RewardSource


def _make_episode(agent_id: str, action_id: str, logprob: float = None) -> Episode:
    return Episode(agent_id=agent_id, user_input="x", assistant_output="y", action_id=action_id, action_logprob=logprob)


def test_positive_reward_shifts_logit_up() -> None:
    actions = [Action(id="a"), Action(id="b")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="nba", rng=random.Random(0))
    learner = ReinforceLearner(LearnerConfig(learning_rate=0.5, entropy_bonus=0.0))

    episodes = []
    rewards = []
    for _ in range(20):
        ep = _make_episode("nba", "a")
        episodes.append(ep)
        rewards.append(
            Reward(
                episode_id=ep.id,
                agent_id="nba",
                source=RewardSource.AGGREGATE,
                value=0.8,
            )
        )

    before = policy.snapshot()
    result = learner.update(policy, episodes, rewards)
    after = policy.snapshot()

    assert result.episodes_used == 20
    assert after.logits["a"] > before.logits["a"]
    assert after.logits["b"] < before.logits["b"]
    assert after.version == before.version + 1
    assert after.episodes_seen == 20
    # Mean reward should be exactly 0.8 because every aggregate is identical.
    assert abs(result.mean_reward - 0.8) < 1e-9


def test_no_aggregate_reward_yields_noop() -> None:
    actions = [Action(id="a"), Action(id="b")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="nba")
    learner = ReinforceLearner(LearnerConfig())

    # Pass non-aggregate rewards only - learner must skip them.
    ep = _make_episode("nba", "a")
    rewards = [
        Reward(
            episode_id=ep.id,
            agent_id="nba",
            source=RewardSource.METRIC,
            value=0.9,
        )
    ]
    result = learner.update(policy, [ep], rewards)
    assert result.episodes_used == 0


def test_unknown_action_is_skipped() -> None:
    actions = [Action(id="a")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="nba")
    learner = ReinforceLearner(LearnerConfig())

    ep = _make_episode("nba", "missing")
    rewards = [
        Reward(
            episode_id=ep.id,
            agent_id="nba",
            source=RewardSource.AGGREGATE,
            value=1.0,
        )
    ]
    result = learner.update(policy, [ep], rewards)
    assert result.episodes_used == 0
