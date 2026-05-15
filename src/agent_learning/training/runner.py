"""End-to-end native learning loop.

The :class:`LearningRunner` ties together the four moving parts:

1. ``MetricEvaluator``s score each episode against the three judges.
2. ``RewardShaper`` collapses metrics into a scalar reward.
3. ``RewardWriter`` persists per-metric and aggregate rewards.
4. ``Learner`` consumes recent episodes + their aggregate rewards
   and applies one update to the current ``Policy``.

The runner is intentionally synchronous and stateless: callers
control the cadence (cron, manual, event-driven). Each invocation
of :meth:`run_offline_batch` is one training step.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from ..config import JudgeConfig, LearnerConfig, ShapingConfig
from ..learners.base import Learner, LearnerResult
from ..learners.reinforce import ReinforceLearner
from ..metrics.base import MetricEvaluator
from ..metrics.registry import default_metrics, evaluate_all
from ..policy.base import Policy
from ..rewards.shaping import RewardShaper, shape_episode_reward
from ..rewards.writer import RewardWriter
from ..storage.base import LearningStore
from ..storage.cosmos import get_default_store
from ..types import Episode, MetricResult, Reward, TrainingRun, TrainingStatus

logger = logging.getLogger(__name__)


class LearningRunner:
    """Orchestrates the evaluate → shape → learn pipeline."""

    def __init__(
        self,
        *,
        store: Optional[LearningStore] = None,
        policy: Optional[Policy] = None,
        metrics: Optional[Iterable[MetricEvaluator]] = None,
        shaper: Optional[RewardShaper] = None,
        writer: Optional[RewardWriter] = None,
        learner: Optional[Learner] = None,
        judge_config: Optional[JudgeConfig] = None,
        learner_config: Optional[LearnerConfig] = None,
        shaping_config: Optional[ShapingConfig] = None,
    ) -> None:
        self._store = store or get_default_store()
        self._policy = policy
        self._metrics = list(metrics) if metrics is not None else default_metrics(judge_config)
        self._shaper = shaper or RewardShaper(shaping_config)
        self._writer = writer or RewardWriter(self._store)
        self._learner = learner or ReinforceLearner(learner_config)

    # ------------------------------------------------------------------
    # Per-episode pipeline
    # ------------------------------------------------------------------

    def evaluate_episode(self, episode: Episode) -> List[MetricResult]:
        """Run all metrics over an episode without writing anything."""
        return evaluate_all(episode, self._metrics)

    def score_and_record(self, episode: Episode) -> List[Reward]:
        """Evaluate, shape, and persist rewards for one episode."""
        results = self.evaluate_episode(episode)
        shaped = shape_episode_reward(episode, results, self._shaper.config)
        return self._writer.write(episode, results, shaped)

    # ------------------------------------------------------------------
    # Batched offline learning
    # ------------------------------------------------------------------

    def run_offline_batch(
        self,
        agent_id: str,
        *,
        episode_limit: int = 200,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        score_missing: bool = True,
    ) -> TrainingRun:
        """Score (if missing) and update the policy over recent episodes."""
        if self._policy is None:
            raise RuntimeError("A Policy must be supplied before running a batch update.")

        run = TrainingRun(
            agent_id=agent_id,
            policy_id=self._policy.snapshot().id,
            algorithm=type(self._learner).__name__,
            status=TrainingStatus.RUNNING,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._store.store_run(run)

        try:
            episodes = self._store.query_episodes(
                agent_id,
                limit=episode_limit,
                start_date=start_date,
                end_date=end_date,
            )
            rewards = self._collect_rewards(agent_id, episodes, score_missing=score_missing)

            result = self._learner.update(self._policy, episodes, rewards)
            policy_snapshot = self._policy.snapshot()
            self._store.store_policy(policy_snapshot)

            run.status = TrainingStatus.SUCCEEDED
            run.completed_at = datetime.now(timezone.utc).isoformat()
            run.episode_ids = [ep.id for ep in episodes]
            run.metrics = _summarise_result(result)
            run.metadata.update(
                {
                    "policy_version": policy_snapshot.version,
                    "score_missing": score_missing,
                }
            )
            self._store.store_run(run)
            return run
        except Exception as exc:  # pragma: no cover - defensive path
            run.status = TrainingStatus.FAILED
            run.error_message = str(exc)
            run.completed_at = datetime.now(timezone.utc).isoformat()
            self._store.store_run(run)
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_rewards(
        self,
        agent_id: str,
        episodes: Iterable[Episode],
        *,
        score_missing: bool,
    ) -> List[Reward]:
        rewards: List[Reward] = []
        for episode in episodes:
            existing = self._store.get_rewards_for_episode(episode.id, agent_id)
            if existing:
                rewards.extend(existing)
                continue
            if not score_missing:
                continue
            scored = self.score_and_record(episode)
            rewards.extend(scored)
        return rewards


def _summarise_result(result: LearnerResult) -> Dict[str, object]:
    """Project a :class:`LearnerResult` into a Cosmos-friendly dict."""
    return {
        "episodes_used": result.episodes_used,
        "mean_reward": result.mean_reward,
        "baseline_before": result.baseline_before,
        "baseline_after": result.baseline_after,
        "logit_deltas": result.logit_deltas,
        "extra": result.extra,
    }


__all__ = ["LearningRunner"]
