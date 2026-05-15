"""Tests for the softmax bandit policy."""

from __future__ import annotations

import math
import random

from agent_learning.policy import SoftmaxPolicy
from agent_learning.types import Action


def test_uniform_initial_distribution() -> None:
    actions = [Action(id="a"), Action(id="b"), Action(id="c")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="dq")
    probs = policy.probabilities()
    assert len(probs) == 3
    for p in probs:
        assert math.isclose(p, 1 / 3, abs_tol=1e-9)


def test_apply_update_changes_logits_and_version() -> None:
    actions = [Action(id="a"), Action(id="b")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="dq")
    before = policy.snapshot()
    policy.apply_update({"a": 1.0, "b": -1.0}, baseline=0.1, episodes_seen=5)
    after = policy.snapshot()
    assert after.version == before.version + 1
    assert after.logits["a"] > before.logits["a"]
    assert after.logits["b"] < before.logits["b"]
    assert after.baseline == 0.1
    assert after.episodes_seen == 5


def test_choose_returns_consistent_logprob() -> None:
    actions = [Action(id="a"), Action(id="b")]
    rng = random.Random(0)
    policy = SoftmaxPolicy.from_actions(actions, agent_id="dq", rng=rng)
    decision = policy.choose()
    # logprob must match the probability of the chosen action
    idx = next(i for i, a in enumerate(actions) if a.id == decision.action.id)
    assert math.isclose(math.exp(decision.logprob), decision.probabilities[idx], rel_tol=1e-9)


def test_logit_clipping() -> None:
    actions = [Action(id="a")]
    policy = SoftmaxPolicy.from_actions(actions, agent_id="dq", max_logit_abs=2.0)
    policy.apply_update({"a": 100.0}, baseline=0.0, episodes_seen=1)
    assert policy.snapshot().logits["a"] == 2.0
