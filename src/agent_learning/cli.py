"""Command-line interface for the agent-learning SDK."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from .policy.softmax_bandit import SoftmaxPolicy
from .storage.cosmos import get_default_store
from .training.runner import LearningRunner
from .types import Action, MetricName, PolicySnapshot, RewardSource


logger = logging.getLogger(__name__)

_MAX_EPISODES = 500


def _episode_limit(value: str) -> int:
    limit = int(value)
    if limit < 1 or limit > _MAX_EPISODES:
        raise argparse.ArgumentTypeError(f"limit must be between 1 and {_MAX_EPISODES}")
    return limit


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-learn", description="Native RL CLI for AI agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List discovered agent ids and names.")

    tasks = sub.add_parser("tasks-list", help="List tasks for an agent.")
    tasks.add_argument("agent_id")

    count = sub.add_parser(
        "task-episodes-count",
        help="Count full learning episodes for an agent.",
    )
    count.add_argument("agent_id")
    count.add_argument("--task-id")

    episodes = sub.add_parser(
        "task-episodes-list",
        help="Print episodes with score and reward details.",
    )
    episodes.add_argument("agent_id")
    episodes.add_argument("--task-id")
    episodes.add_argument("--limit", type=_episode_limit, default=_MAX_EPISODES)
    episodes.add_argument("--include-incomplete", action="store_true")

    train = sub.add_parser("train", help="Run one offline learning batch.")
    train.add_argument("--agent-id", required=True)
    train.add_argument("--task-id")
    train.add_argument("--limit", type=_episode_limit, default=200)
    train.add_argument("--start-date")
    train.add_argument("--end-date")
    train.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip scoring episodes that have no rewards yet.",
    )

    score = sub.add_parser("score", help="Score episodes but skip the policy update.")
    score.add_argument("--agent-id", required=True)
    score.add_argument("--task-id")
    score.add_argument("--limit", type=_episode_limit, default=100)

    show = sub.add_parser("task-policy", help="Print the active policy for an agent task.")
    show.add_argument("--agent-id", required=True)
    show.add_argument("--task-id", required=True)

    init = sub.add_parser(
        "task-policy-init",
        help="Create and activate the initial policy for an agent task.",
    )
    init.add_argument("--agent-id", required=True)
    init.add_argument("--task-id", required=True)
    init.add_argument(
        "--actions",
        required=True,
        help="Path to a JSON file containing a list of {id, description, parameters} objects.",
    )

    return parser


def _cmd_agents_list(args: argparse.Namespace) -> int:
    del args
    agents = get_default_store().list_agents()
    print(json.dumps([{"id": agent.id, "name": agent.name} for agent in agents], indent=2))
    return 0


def _cmd_agent_tasks_list(args: argparse.Namespace) -> int:
    tasks = get_default_store().list_agent_tasks(args.agent_id)
    print(json.dumps([{"id": task.id, "name": task.name} for task in tasks], indent=2))
    return 0


def _cmd_agents_episodes_count(args: argparse.Namespace) -> int:
    count = get_default_store().count_episodes(
        args.agent_id,
        task_id=args.task_id,
        full_only=True,
    )
    print(count)
    return 0


def _cmd_agents_episodes_list(args: argparse.Namespace) -> int:
    store = get_default_store()
    episodes = store.query_episodes(
        args.agent_id,
        task_id=args.task_id,
        limit=_MAX_EPISODES,
    )
    if not args.include_incomplete:
        episodes = [episode for episode in episodes if episode.is_full]

    payload = []
    for episode in episodes[: args.limit]:
        metrics = store.get_metric_results(episode.id, args.agent_id)
        rewards = store.get_rewards_for_episode(episode.id, args.agent_id)
        aggregate_rewards = [
            reward for reward in rewards if reward.source == RewardSource.AGGREGATE
        ]
        aggregate_rewards.sort(key=lambda reward: reward.created_at, reverse=True)
        completion = next(
            (metric for metric in metrics if metric.metric == MetricName.TASK_COMPLETION),
            None,
        )
        payload.append(
            {
                "episode": episode.to_dict(),
                "score_breakdown": [metric.to_dict() for metric in metrics],
                "final_reward": aggregate_rewards[0].value if aggregate_rewards else None,
                "task_completion_quality": completion.to_dict() if completion else None,
            }
        )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    store = get_default_store()
    task_ids = (
        [args.task_id]
        if args.task_id
        else [task.id for task in store.list_agent_tasks(args.agent_id)]
    )
    selected_episodes = store.query_episodes(
        args.agent_id,
        task_id=args.task_id,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    episode_limits = Counter(episode.task_id for episode in selected_episodes)
    runs: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for task_id in task_ids:
        snapshot = store.get_active_policy(args.agent_id, task_id)
        if snapshot is None:
            skipped.append({"task_id": task_id, "reason": "no active policy"})
            continue
        episode_limit = episode_limits.get(task_id, 0)
        if episode_limit == 0:
            skipped.append({"task_id": task_id, "reason": "no episodes in selected batch"})
            continue
        policy = SoftmaxPolicy.from_snapshot(snapshot)
        runner = LearningRunner(store=store, policy=policy)
        run = runner.run_offline_batch(
            args.agent_id,
            task_id=task_id,
            episode_limit=episode_limit,
            start_date=args.start_date,
            end_date=args.end_date,
            score_missing=not args.skip_scoring,
        )
        runs.append(run.to_dict())

    print(json.dumps({"agent_id": args.agent_id, "runs": runs, "skipped": skipped}, indent=2))
    if runs:
        return 0
    print(
        f"No task policies were trained for agent_id={args.agent_id!r}. "
        "Check the skipped reasons and initialize missing task policies.",
        file=sys.stderr,
    )
    return 2


def _cmd_score(args: argparse.Namespace) -> int:
    store = get_default_store()
    runner = LearningRunner(store=store)
    episodes = store.query_episodes(
        args.agent_id,
        task_id=args.task_id,
        limit=args.limit,
    )
    scored = 0
    for episode in episodes:
        existing = store.get_rewards_for_episode(episode.id, args.agent_id)
        if existing:
            continue
        runner.score_and_record(episode)
        scored += 1
    print(json.dumps({"episodes_seen": len(episodes), "newly_scored": scored}, indent=2))
    return 0


def _policy_payload(snapshot: PolicySnapshot) -> Dict[str, Any]:
    payload = snapshot.to_dict()
    policy = SoftmaxPolicy.from_snapshot(snapshot)
    payload["action_probabilities"] = {
        action.id: probability
        for action, probability in zip(snapshot.actions, policy.probabilities())
    }
    return payload


def _policy_difference(
    current: PolicySnapshot, previous: Optional[PolicySnapshot]
) -> Optional[Dict[str, Any]]:
    if previous is None:
        return None
    current_payload = _policy_payload(current)
    previous_payload = _policy_payload(previous)
    action_ids = sorted(
        set(current_payload["action_probabilities"])
        | set(previous_payload["action_probabilities"])
    )
    return {
        "version": current.version - previous.version,
        "baseline": current.baseline - previous.baseline,
        "episodes_seen": current.episodes_seen - previous.episodes_seen,
        "updates_applied": current.updates_applied - previous.updates_applied,
        "logits": {
            action_id: current.logits.get(action_id, 0.0)
            - previous.logits.get(action_id, 0.0)
            for action_id in action_ids
        },
        "action_probabilities": {
            action_id: current_payload["action_probabilities"].get(action_id, 0.0)
            - previous_payload["action_probabilities"].get(action_id, 0.0)
            for action_id in action_ids
        },
    }


def _cmd_show_task_policy(args: argparse.Namespace) -> int:
    store = get_default_store()
    snapshot = store.get_active_policy(args.agent_id, args.task_id)
    if snapshot is None:
        print(
            f"No active policy found for agent_id={args.agent_id!r}, "
            f"task_id={args.task_id!r}.",
            file=sys.stderr,
        )
        return 2
    previous = next(
        (
            policy
            for policy in store.list_policies(args.agent_id, args.task_id)
            if policy.id != snapshot.id
        ),
        None,
    )
    print(
        json.dumps(
            {
                "current_policy": _policy_payload(snapshot),
                "previous_policy": _policy_payload(previous) if previous else None,
                "difference": _policy_difference(snapshot, previous),
            },
            indent=2,
        )
    )
    return 0


def _cmd_init_task_policy(args: argparse.Namespace) -> int:
    store = get_default_store()
    if store.get_active_policy(args.agent_id, args.task_id) is not None:
        print(
            f"An active policy already exists for agent_id={args.agent_id!r}, "
            f"task_id={args.task_id!r}.",
            file=sys.stderr,
        )
        return 2
    try:
        with open(args.actions, "r", encoding="utf-8") as actions_file:
            action_payloads = json.load(actions_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Unable to read --actions file: {exc}", file=sys.stderr)
        return 2
    if not isinstance(action_payloads, list) or not action_payloads:
        print("--actions file must contain a non-empty JSON list", file=sys.stderr)
        return 2
    try:
        actions = [Action.from_dict(item) for item in action_payloads]
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Invalid action definition: {exc}", file=sys.stderr)
        return 2
    policy = SoftmaxPolicy.from_actions(
        actions,
        agent_id=args.agent_id,
        task_id=args.task_id,
    )
    snapshot = policy.snapshot()
    store.store_policy(snapshot)
    print(json.dumps(_policy_payload(snapshot), indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "list": _cmd_agents_list,
        "tasks-list": _cmd_agent_tasks_list,
        "task-episodes-count": _cmd_agents_episodes_count,
        "task-episodes-list": _cmd_agents_episodes_list,
        "train": _cmd_train,
        "score": _cmd_score,
        "task-policy": _cmd_show_task_policy,
        "task-policy-init": _cmd_init_task_policy,
    }
    handler = dispatch[args.command]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
