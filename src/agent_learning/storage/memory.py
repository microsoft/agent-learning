"""In-memory store used for tests and offline experimentation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from ..types import (
    AgentSummary,
    AgentTaskSummary,
    Episode,
    MetricResult,
    PolicySnapshot,
    Reward,
    TrainingRun,
)
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
        self._active_policies: Dict[Tuple[str, str], str] = {}
        self._runs: Dict[Tuple[str, str], TrainingRun] = {}

    # ---- Discovery -------------------------------------------------

    def list_agents(self) -> List[AgentSummary]:
        names: Dict[str, str] = {}
        for episode in self._episodes.values():
            names.setdefault(episode.agent_id, episode.agent_id)
            if episode.agent_name:
                names[episode.agent_id] = episode.agent_name
        for policy in self._policies.values():
            names.setdefault(policy.agent_id, policy.agent_id)
        return [AgentSummary(id=agent_id, name=names[agent_id]) for agent_id in sorted(names)]

    def list_agent_tasks(self, agent_id: str) -> List[AgentTaskSummary]:
        names: Dict[str, str] = {}
        for episode in self._episodes.values():
            if episode.agent_id != agent_id:
                continue
            names.setdefault(episode.task_id, episode.task_id)
            if episode.task_name:
                names[episode.task_id] = episode.task_name
        for policy in self._policies.values():
            if policy.agent_id == agent_id:
                names.setdefault(policy.task_id, policy.task_id)
        return [AgentTaskSummary(id=task_id, name=names[task_id]) for task_id in sorted(names)]

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
        task_id: Optional[str] = None,
    ) -> List[Episode]:
        results = [
            ep
            for (eid, aid), ep in self._episodes.items()
            if aid == agent_id
            and (start_date is None or ep.created_at >= start_date)
            and (end_date is None or ep.created_at <= end_date)
            and (policy_id is None or ep.policy_id == policy_id)
            and (task_id is None or ep.task_id == task_id)
        ]
        results.sort(key=lambda ep: ep.created_at, reverse=True)
        return results[:limit]

    def count_episodes(
        self,
        agent_id: str,
        *,
        task_id: Optional[str] = None,
        full_only: bool = False,
    ) -> int:
        return sum(
            1
            for episode in self._episodes.values()
            if episode.agent_id == agent_id
            and (task_id is None or episode.task_id == task_id)
            and (not full_only or episode.is_full)
        )

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
        stored = PolicySnapshot.from_dict(policy.to_dict())
        self._policies[(stored.id, stored.agent_id)] = stored
        self._active_policies[(stored.agent_id, stored.task_id)] = stored.id
        return stored.id

    def get_policy(self, policy_id: str, agent_id: str) -> Optional[PolicySnapshot]:
        policy = self._policies.get((policy_id, agent_id))
        return PolicySnapshot.from_dict(policy.to_dict()) if policy else None

    def list_policies(
        self,
        agent_id: str,
        task_id: str,
        *,
        limit: int = 100,
    ) -> List[PolicySnapshot]:
        candidates = [
            policy
            for (_, aid), policy in self._policies.items()
            if aid == agent_id and policy.task_id == task_id
        ]
        candidates.sort(key=lambda policy: (policy.version, policy.created_at), reverse=True)
        return [PolicySnapshot.from_dict(policy.to_dict()) for policy in candidates[:limit]]

    def get_latest_policy(
        self, agent_id: str, task_id: str = "default"
    ) -> Optional[PolicySnapshot]:
        candidates = self.list_policies(agent_id, task_id, limit=1)
        if not candidates:
            return None
        return candidates[0]

    def get_active_policy(
        self, agent_id: str, task_id: str
    ) -> Optional[PolicySnapshot]:
        policy_id = self._active_policies.get((agent_id, task_id))
        if policy_id is None:
            return self.get_latest_policy(agent_id, task_id)
        return self.get_policy(policy_id, agent_id)

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
