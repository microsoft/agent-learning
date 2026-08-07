"""In-memory store used for tests and offline experimentation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from ..types import (
    AgentInfo,
    AgentTaskInfo,
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
        task_id: Optional[str] = None,
        completed_only: bool = False,
    ) -> List[Episode]:
        results = [
            ep
            for (eid, aid), ep in self._episodes.items()
            if aid == agent_id
            and (start_date is None or ep.created_at >= start_date)
            and (end_date is None or ep.created_at <= end_date)
            and (policy_id is None or ep.policy_id == policy_id)
            and (task_id is None or ep.task_id == task_id)
            and (not completed_only or ep.is_complete)
        ]
        results.sort(key=lambda ep: ep.created_at, reverse=True)
        return results[:limit]

    def count_completed_episodes(
        self,
        agent_id: str,
        *,
        task_id: Optional[str] = None,
    ) -> int:
        return sum(
            episode.is_complete and (task_id is None or episode.task_id == task_id)
            for (_episode_id, stored_agent_id), episode in self._episodes.items()
            if stored_agent_id == agent_id
        )

    def list_agents(self) -> List[AgentInfo]:
        latest: Dict[str, PolicySnapshot] = {}
        for (_policy_id, agent_id), policy in self._policies.items():
            current = latest.get(agent_id)
            if current is None or policy.version > current.version:
                latest[agent_id] = policy
        return [
            AgentInfo.from_metadata(agent_id, latest[agent_id].metadata)
            for agent_id in sorted(latest)
        ]

    def list_agent_tasks(self, agent_id: str) -> List[AgentTaskInfo]:
        latest: Dict[str, PolicySnapshot] = {}
        for (_policy_id, stored_agent_id), policy in self._policies.items():
            if stored_agent_id != agent_id or not policy.task_id:
                continue
            current = latest.get(policy.task_id)
            if current is None or (policy.version, policy.created_at) > (
                current.version,
                current.created_at,
            ):
                latest[policy.task_id] = policy
        return [
            AgentTaskInfo.from_metadata(task_id, latest[task_id].metadata)
            for task_id in sorted(latest)
        ]

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

    def list_policies(
        self,
        agent_id: str,
        *,
        task_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[PolicySnapshot]:
        policies = [
            policy
            for (_policy_id, stored_agent_id), policy in self._policies.items()
            if stored_agent_id == agent_id
            and (task_id is None or policy.task_id == task_id)
        ]
        policies.sort(
            key=lambda policy: (policy.version, policy.created_at),
            reverse=True,
        )
        return policies[:limit]

    def get_latest_policy(
        self,
        agent_id: str,
        *,
        task_id: Optional[str] = None,
    ) -> Optional[PolicySnapshot]:
        policies = self.list_policies(agent_id, task_id=task_id, limit=1)
        return policies[0] if policies else None

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
