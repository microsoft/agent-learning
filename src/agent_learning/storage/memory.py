"""In-memory store used for tests and offline experimentation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from ..types import Episode, MetricResult, PolicySnapshot, Reward, TrainingRun
from .base import LearningStore


class InMemoryStore(LearningStore):
    """Thread-unsafe in-memory implementation of :class:`LearningStore`.

    Suitable for unit tests and single-process notebooks. Data does
    not survive process exit.
    """

    def __init__(self) -> None:
        self._episodes: Dict[Tuple[str, str], Episode] = {}
        self._metrics: Dict[Tuple[str, str], List[MetricResult]] = {}
        self._rewards: Dict[Tuple[str, str], List[Reward]] = {}
        self._policies: Dict[Tuple[str, str], PolicySnapshot] = {}
        self._runs: Dict[Tuple[str, str], TrainingRun] = {}

    # ---- Episodes --------------------------------------------------

    def store_episode(self, episode: Episode) -> str:
        self._episodes[(episode.id, episode.agent_id)] = episode
        return episode.id

    def get_episode(self, episode_id: str, agent_id: str) -> Optional[Episode]:
        return self._episodes.get((episode_id, agent_id))

    def query_episodes(
        self,
        agent_id: str,
        *,
        limit: int = 100,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> List[Episode]:
        results = [
            ep
            for (eid, aid), ep in self._episodes.items()
            if aid == agent_id
            and (start_date is None or ep.created_at >= start_date)
            and (end_date is None or ep.created_at <= end_date)
            and (policy_id is None or ep.policy_id == policy_id)
        ]
        results.sort(key=lambda ep: ep.created_at, reverse=True)
        return results[:limit]

    # ---- Metric results -------------------------------------------

    def store_metric_results(
        self, episode_id: str, agent_id: str, results: Iterable[MetricResult]
    ) -> None:
        key = (episode_id, agent_id)
        existing = self._metrics.setdefault(key, [])
        existing.extend(list(results))

    def get_metric_results(self, episode_id: str, agent_id: str) -> List[MetricResult]:
        return list(self._metrics.get((episode_id, agent_id), []))

    # ---- Rewards ---------------------------------------------------

    def store_reward(self, reward: Reward) -> str:
        key = (reward.episode_id, reward.agent_id)
        existing = self._rewards.setdefault(key, [])
        # De-duplicate by id (upsert semantics)
        existing[:] = [r for r in existing if r.id != reward.id]
        existing.append(reward)
        return reward.id

    def get_rewards_for_episode(self, episode_id: str, agent_id: str) -> List[Reward]:
        return list(self._rewards.get((episode_id, agent_id), []))

    def query_rewards(
        self,
        agent_id: str,
        *,
        episode_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Reward]:
        if episode_id is not None:
            return list(self._rewards.get((episode_id, agent_id), []))[:limit]
        results: List[Reward] = []
        for (eid, aid), rewards in self._rewards.items():
            if aid == agent_id:
                results.extend(rewards)
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    # ---- Policy snapshots -----------------------------------------

    def store_policy(self, policy: PolicySnapshot) -> str:
        self._policies[(policy.id, policy.agent_id)] = policy
        return policy.id

    def get_policy(self, policy_id: str, agent_id: str) -> Optional[PolicySnapshot]:
        return self._policies.get((policy_id, agent_id))

    def get_latest_policy(self, agent_id: str) -> Optional[PolicySnapshot]:
        candidates = [p for (pid, aid), p in self._policies.items() if aid == agent_id]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.version)

    # ---- Training runs --------------------------------------------

    def store_run(self, run: TrainingRun) -> str:
        self._runs[(run.id, run.agent_id)] = run
        return run.id

    def get_run(self, run_id: str, agent_id: str) -> Optional[TrainingRun]:
        return self._runs.get((run_id, agent_id))

    def list_training_runs(
        self,
        agent_id: str,
        *,
        limit: int = 100,
    ) -> List[TrainingRun]:
        results = [r for (rid, aid), r in self._runs.items() if aid == agent_id]
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]


__all__ = ["InMemoryStore"]
