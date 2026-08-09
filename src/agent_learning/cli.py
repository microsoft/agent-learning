"""Command-line interface for the agent-learning SDK."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from ._version import __version__
from .policy.softmax_bandit import SoftmaxPolicy
from .storage.cosmos import get_default_store
from .training.runner import LearningRunner
from .types import Action, Episode, MetricName, PolicySnapshot, RewardSource

logger = logging.getLogger(__name__)

_MAX_EPISODES = 500
_DECISION_POLICY_SCOPE = "delegated_decision"


def _episode_limit(value: str) -> int:
    limit = int(value)
    if limit < 1 or limit > _MAX_EPISODES:
        raise argparse.ArgumentTypeError(f"limit must be between 1 and {_MAX_EPISODES}")
    return limit


def _iso_date(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO 8601 date: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-learn",
        description=f"Native RL CLI for AI agents. SDK version {__version__}.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List discovered agent ids and names.")

    tasks = sub.add_parser("tasks-list", help="List tasks for an agent.")
    tasks.add_argument("agent_id")
    tasks.add_argument("--decision-only", action="store_true")

    count = sub.add_parser(
        "task-episodes-count",
        help="Count full learning episodes for an agent.",
    )
    count.add_argument("agent_id")
    count.add_argument("--task-id")
    count.add_argument("--start-date", type=_iso_date)
    count.add_argument("--end-date", type=_iso_date)

    episodes = sub.add_parser(
        "task-episodes-list",
        help="Print episodes with score and reward details.",
    )
    episodes.add_argument("agent_id")
    episodes.add_argument("--task-id")
    episodes.add_argument("--limit", type=_episode_limit, default=_MAX_EPISODES)
    episodes.add_argument("--include-incomplete", action="store_true")
    episodes.add_argument("--start-date", type=_iso_date)
    episodes.add_argument("--end-date", type=_iso_date)

    train = sub.add_parser("train", help="Run one offline learning batch.")
    train.add_argument("--agent-id", required=True)
    train.add_argument("--task-id")
    train.add_argument("--limit", type=_episode_limit, default=200)
    train.add_argument("--min-episodes", type=_episode_limit, default=1)
    train.add_argument("--decision-only", action="store_true")
    train.add_argument("--start-date", type=_iso_date)
    train.add_argument("--end-date", type=_iso_date)
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

    decide = sub.add_parser(
        "task-policy-decide",
        help="Choose a delegated decision action and return learned feedback.",
    )
    decide.add_argument("--agent-id", required=True)
    decide.add_argument("--task-id", required=True)
    decide.add_argument("--history-limit", type=_episode_limit, default=100)
    decide.add_argument("--greedy", action="store_true")
    decide.add_argument("--seed", type=int)

    init = sub.add_parser(
        "task-policy-init",
        help="Create and activate the initial policy for an agent task.",
    )
    init.add_argument("--agent-id", required=True)
    init.add_argument("--task-id", required=True)
    init.add_argument(
        "--decision-context",
        required=True,
        help="Stable description of the delegated choice this policy controls.",
    )
    init.add_argument(
        "--actions",
        required=True,
        help="Path to a JSON file containing a list of {id, description, parameters} objects.",
    )

    register = sub.add_parser(
        "task-episode-register",
        help="Register an agent task episode from a JSON file.",
    )
    register.add_argument("--agent-id", required=True)
    register.add_argument("--task-id", required=True)
    register.add_argument("--require-decision-policy", action="store_true")
    register.add_argument(
        "--episode",
        required=True,
        help="Path to a JSON file containing an Episode object.",
    )

    return parser


def _cmd_agents_list(args: argparse.Namespace) -> int:
    del args
    agents = get_default_store().list_agents()
    print(json.dumps([{"id": agent.id, "name": agent.name} for agent in agents], indent=2))
    return 0


def _cmd_agent_tasks_list(args: argparse.Namespace) -> int:
    store = get_default_store()
    tasks = store.list_agent_tasks(args.agent_id)
    if args.decision_only:
        tasks = [
            task
            for task in tasks
            if _is_decision_policy(store.get_active_policy(args.agent_id, task.id))
        ]
    print(json.dumps([{"id": task.id, "name": task.name} for task in tasks], indent=2))
    return 0


def _cmd_agents_episodes_count(args: argparse.Namespace) -> int:
    count = get_default_store().count_episodes(
        args.agent_id,
        task_id=args.task_id,
        full_only=True,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(count)
    return 0


def _cmd_agents_episodes_list(args: argparse.Namespace) -> int:
    store = get_default_store()
    episodes = store.query_episodes(
        args.agent_id,
        task_id=args.task_id,
        limit=_MAX_EPISODES,
        start_date=args.start_date,
        end_date=args.end_date,
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
    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for task_id in task_ids:
        snapshot = store.get_active_policy(args.agent_id, task_id)
        if snapshot is None:
            skipped.append({"task_id": task_id, "reason": "no active policy"})
            continue
        if args.decision_only and not _is_decision_policy(snapshot):
            skipped.append({"task_id": task_id, "reason": "not a delegated decision policy"})
            continue
        episode_limit = episode_limits.get(task_id, 0)
        if episode_limit == 0:
            skipped.append({"task_id": task_id, "reason": "no episodes in selected batch"})
            continue
        if episode_limit < args.min_episodes:
            skipped.append(
                {
                    "task_id": task_id,
                    "reason": (
                        f"selected batch has {episode_limit} episodes; "
                        f"minimum is {args.min_episodes}"
                    ),
                }
            )
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
        if runner.has_usable_reward(episode):
            continue
        rewards = runner.score_and_record(episode)
        if any(reward.source == RewardSource.AGGREGATE for reward in rewards):
            scored += 1
    print(json.dumps({"episodes_seen": len(episodes), "newly_scored": scored}, indent=2))
    return 0


def _policy_payload(snapshot: PolicySnapshot) -> dict[str, Any]:
    payload = snapshot.to_dict()
    policy = SoftmaxPolicy.from_snapshot(snapshot)
    payload["action_probabilities"] = {
        action.id: probability
        for action, probability in zip(snapshot.actions, policy.probabilities())
    }
    return payload


def _is_decision_policy(snapshot: PolicySnapshot | None) -> bool:
    return bool(
        snapshot
        and snapshot.metadata.get("policy_scope") == _DECISION_POLICY_SCOPE
    )


def _latest_aggregate(store: Any, episode: Episode) -> float | None:
    rewards = [
        reward
        for reward in store.get_rewards_for_episode(episode.id, episode.agent_id)
        if reward.source == RewardSource.AGGREGATE
    ]
    if not rewards:
        return None
    return max(rewards, key=lambda reward: reward.created_at).value


def _decision_feedback(
    store: Any, snapshot: PolicySnapshot, history_limit: int
) -> dict[str, Any]:
    stats = {
        action.id: {
            "attempts": 0,
            "correctness_evaluated": 0,
            "correct": 0,
            "correctness_rate": None,
            "rewarded_episodes": 0,
            "mean_reward": None,
            "recent_outcomes": [],
        }
        for action in snapshot.actions
    }
    reward_totals = {action.id: 0.0 for action in snapshot.actions}
    episodes = store.query_episodes(
        snapshot.agent_id,
        task_id=snapshot.task_id,
        limit=history_limit,
    )
    for episode in episodes:
        action_id = episode.action_id
        if action_id not in stats:
            continue
        action_stats = stats[action_id]
        action_stats["attempts"] += 1
        correct_action_id = episode.metadata.get("correct_action_id")
        was_correct = None
        if correct_action_id:
            was_correct = action_id == correct_action_id
            action_stats["correctness_evaluated"] += 1
            action_stats["correct"] += int(was_correct)
        reward = _latest_aggregate(store, episode)
        if reward is not None:
            action_stats["rewarded_episodes"] += 1
            reward_totals[action_id] += reward
        if len(action_stats["recent_outcomes"]) < 3:
            score_breakdown = {}
            for result in store.get_metric_results(episode.id, episode.agent_id):
                score_breakdown[result.metric.value] = {
                    "normalized": result.normalized,
                    "status": result.status,
                    "reason": result.reason,
                }
            action_stats["recent_outcomes"].append(
                {
                    "created_at": episode.created_at,
                    "was_correct": was_correct,
                    "reward": reward,
                    "execution_status": episode.execution_status,
                    "result_summary": episode.result_summary,
                    "score_breakdown": score_breakdown,
                }
            )
    for action_id, action_stats in stats.items():
        evaluated = action_stats["correctness_evaluated"]
        rewarded = action_stats["rewarded_episodes"]
        if evaluated:
            action_stats["correctness_rate"] = action_stats["correct"] / evaluated
        if rewarded:
            action_stats["mean_reward"] = reward_totals[action_id] / rewarded
    return {"episodes_reviewed": len(episodes), "actions": stats}


def _policy_difference(
    current: PolicySnapshot, previous: PolicySnapshot | None
) -> dict[str, Any] | None:
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


def _cmd_decide_task_policy(args: argparse.Namespace) -> int:
    store = get_default_store()
    snapshot = store.get_active_policy(args.agent_id, args.task_id)
    if snapshot is None:
        print(
            f"No active policy found for agent_id={args.agent_id!r}, "
            f"task_id={args.task_id!r}.",
            file=sys.stderr,
        )
        return 2
    if not _is_decision_policy(snapshot):
        print(
            "The active policy is not marked as a delegated decision policy. "
            "Questions, reporting tasks, and agent-learning automation are not eligible.",
            file=sys.stderr,
        )
        return 2
    rng = random.Random(args.seed) if args.seed is not None else None
    policy = SoftmaxPolicy.from_snapshot(snapshot, rng=rng)
    probabilities = policy.probabilities()
    recommended_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    if args.greedy:
        selected_index = recommended_index
        selected_action = snapshot.actions[selected_index]
        selected_probability = probabilities[selected_index]
        logprob = math.log(max(selected_probability, 1e-12))
        mode = "greedy"
    else:
        decision = policy.choose()
        selected_action = decision.action
        selected_index = next(
            index
            for index, action in enumerate(snapshot.actions)
            if action.id == selected_action.id
        )
        selected_probability = probabilities[selected_index]
        logprob = decision.logprob
        mode = "sampled"
    feedback = _decision_feedback(store, snapshot, args.history_limit)
    selected_stats = feedback["actions"][selected_action.id]
    recommendation = snapshot.actions[recommended_index]
    print(
        json.dumps(
            {
                "agent_id": snapshot.agent_id,
                "task_id": snapshot.task_id,
                "decision_context": snapshot.metadata.get("decision_context"),
                "policy_id": snapshot.id,
                "policy_version": snapshot.version,
                "selection_mode": mode,
                "selected_action": {
                    **selected_action.to_dict(),
                    "probability": selected_probability,
                    "logprob": logprob,
                },
                "recommended_action": {
                    **recommendation.to_dict(),
                    "probability": probabilities[recommended_index],
                },
                "action_probabilities": {
                    action.id: probability
                    for action, probability in zip(snapshot.actions, probabilities)
                },
                "selected_action_feedback": selected_stats,
                "historical_feedback": feedback,
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
    if not isinstance(action_payloads, list) or len(action_payloads) < 2:
        print(
            "--actions file must contain at least two delegated decision actions",
            file=sys.stderr,
        )
        return 2
    try:
        actions = [Action.from_dict(item) for item in action_payloads]
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Invalid action definition: {exc}", file=sys.stderr)
        return 2
    action_ids = [action.id for action in actions]
    if any(not action_id.strip() for action_id in action_ids) or len(set(action_ids)) != len(
        action_ids
    ):
        print("Decision action ids must be non-empty and unique", file=sys.stderr)
        return 2
    policy = SoftmaxPolicy.from_actions(
        actions,
        agent_id=args.agent_id,
        task_id=args.task_id,
    )
    snapshot = policy.snapshot()
    snapshot.metadata.update(
        {
            "policy_scope": _DECISION_POLICY_SCOPE,
            "decision_context": args.decision_context,
        }
    )
    store.store_policy(snapshot)
    print(json.dumps(_policy_payload(snapshot), indent=2))
    return 0


def _cmd_register_task_episode(args: argparse.Namespace) -> int:
    try:
        with open(args.episode, "r", encoding="utf-8") as episode_file:
            payload = json.load(episode_file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Unable to read --episode file: {exc}", file=sys.stderr)
        return 2
    if not isinstance(payload, dict):
        print("--episode file must contain a JSON object", file=sys.stderr)
        return 2

    payload = dict(payload)
    for field, expected in (("agent_id", args.agent_id), ("task_id", args.task_id)):
        actual = payload.get(field)
        if actual is not None and actual != expected:
            print(
                f"Episode {field}={actual!r} does not match --{field.replace('_', '-')}={expected!r}.",
                file=sys.stderr,
            )
            return 2
        payload[field] = expected
    payload.setdefault("id", str(uuid.uuid4()))

    try:
        episode = Episode.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Invalid episode definition: {exc}", file=sys.stderr)
        return 2

    if args.require_decision_policy:
        policy = get_default_store().get_policy(episode.policy_id or "", args.agent_id)
        if not _is_decision_policy(policy) or policy.task_id != args.task_id:
            print(
                "Episode registration requires a delegated decision policy and its policy_id.",
                file=sys.stderr,
            )
            return 2
        action_ids = {action.id for action in policy.actions}
        if episode.action_id not in action_ids:
            print("Episode action_id is not in the delegated decision policy.", file=sys.stderr)
            return 2
        correct_action_id = episode.metadata.get("correct_action_id")
        if correct_action_id is not None and correct_action_id not in action_ids:
            print(
                "Episode metadata.correct_action_id is not in the delegated decision policy.",
                file=sys.stderr,
            )
            return 2

    get_default_store().store_episode(episode)
    print(json.dumps(episode.to_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
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
        "task-policy-decide": _cmd_decide_task_policy,
        "task-policy-init": _cmd_init_task_policy,
        "task-episode-register": _cmd_register_task_episode,
    }
    handler = dispatch[args.command]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
