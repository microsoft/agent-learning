"""Command-line interface for the agent-learning SDK.

Examples::

    # Initialise a durable local policy
    agent-learn policy-init --agent-id dq --actions ./actions.json

    # Ask the policy how to handle a task, then record its result
    agent-learn task-intent --agent-id dq --intent "Summarise Q3 sales"
    agent-learn task-complete --agent-id dq --episode-id <id> --output "..."

    # Judge completed episodes and update the policy
    agent-learn train --agent-id dq --limit 500

The CLI defaults to the local file store so state survives separate command
invocations. Explicit ``AGENT_LEARNING_STORE_BACKEND`` configuration still
selects another SDK backend.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .capture import redact
from .policy.base import Policy
from .policy.contextual_softmax import ContextualSoftmaxPolicy
from .policy.softmax_bandit import SoftmaxPolicy
from .storage.base import LearningStore
from .storage.cosmos import get_default_store
from .storage.local import LocalFileStore
from .training.runner import LearningRunner
from .types import Action, Episode, PolicySnapshot


logger = logging.getLogger(__name__)


def _add_store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store-dir",
        help=(
            "Local store directory. Defaults to AGENT_LEARNING_LOCAL_STORE_DIR "
            "or ./data/agent-learning/store."
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-learn", description="Native RL CLI for AI agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run one offline learning batch.")
    train.add_argument("--agent-id", required=True)
    train.add_argument("--limit", type=int, default=200)
    train.add_argument("--start-date")
    train.add_argument("--end-date")
    train.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip scoring episodes that have no rewards yet.",
    )
    _add_store_argument(train)

    score = sub.add_parser("score", help="Score episodes but skip the policy update.")
    score.add_argument("--agent-id", required=True)
    score.add_argument("--limit", type=int, default=100)
    _add_store_argument(score)

    show = sub.add_parser("policy", help="Print the latest policy snapshot for an agent.")
    show.add_argument("--agent-id", required=True)
    _add_store_argument(show)

    init = sub.add_parser(
        "policy-init",
        aliases=["init-policy"],
        help="Create the initial policy snapshot from a JSON file.",
    )
    init.add_argument("--agent-id", required=True)
    init.add_argument(
        "--actions",
        required=True,
        help="Path to a JSON file containing a list of {id, description, parameters} objects.",
    )
    _add_store_argument(init)

    intent = sub.add_parser(
        "task-intent",
        help="Choose an action for a task and start a locally persisted episode.",
    )
    intent.add_argument("--agent-id", required=True)
    intent.add_argument("--intent", required=True, help="The user's requested outcome.")
    intent.add_argument(
        "--context",
        "--features-json",
        dest="context",
        default="{}",
        help=(
            "JSON context object (or @path to a JSON file). A JSON array is treated "
            "as the contextual policy's phi feature vector."
        ),
    )
    intent.add_argument(
        "--episode-id",
        help="Optional caller-provided episode ID. A UUID is generated when omitted.",
    )
    _add_store_argument(intent)

    complete = sub.add_parser(
        "task-complete",
        help="Complete a task episode with the agent's output.",
    )
    complete.add_argument("--agent-id", required=True)
    complete.add_argument("--episode-id", required=True)
    complete.add_argument(
        "--output",
        "--result",
        "--assistant-output",
        dest="output",
        required=True,
        help="The final task output recorded for judging and training.",
    )
    _add_store_argument(complete)

    return parser


def _get_cli_store(args: argparse.Namespace) -> LearningStore:
    store_dir = getattr(args, "store_dir", None)
    if store_dir is not None or "AGENT_LEARNING_STORE_BACKEND" not in os.environ:
        return LocalFileStore(store_dir)
    return get_default_store()


def _policy_from_snapshot(snapshot: PolicySnapshot) -> Policy:
    if snapshot.metadata.get("policy_kind") == "contextual_softmax":
        return ContextualSoftmaxPolicy.from_snapshot(snapshot)
    return SoftmaxPolicy.from_snapshot(snapshot)


def _load_context(raw: str) -> Dict[str, Any]:
    try:
        if raw.startswith("@"):
            with open(raw[1:], "r", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"--context must be valid JSON or @path: {exc}") from exc

    if isinstance(value, list):
        return {"phi": value}
    if not isinstance(value, dict):
        raise ValueError("--context must contain a JSON object or feature-vector array")
    return value


def _redact_json(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {str(key): _redact_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    return value


def _cmd_train(args: argparse.Namespace) -> int:
    store = _get_cli_store(args)
    snapshot = store.get_latest_policy(args.agent_id)
    if snapshot is None:
        print(
            f"No policy found for agent_id={args.agent_id!r}. "
            "Run `agent-learn policy-init` first.",
            file=sys.stderr,
        )
        return 2

    policy = SoftmaxPolicy.from_snapshot(snapshot)
    runner = LearningRunner(store=store, policy=policy)
    run = runner.run_offline_batch(
        args.agent_id,
        episode_limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
        score_missing=not args.skip_scoring,
    )
    print(json.dumps(run.to_dict(), indent=2))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    store = _get_cli_store(args)
    runner = LearningRunner(store=store)
    episodes = store.query_episodes(args.agent_id, limit=args.limit)
    scored = 0
    for episode in episodes:
        existing = store.get_rewards_for_episode(episode.id, args.agent_id)
        if existing:
            continue
        runner.score_and_record(episode)
        scored += 1
    print(json.dumps({"episodes_seen": len(episodes), "newly_scored": scored}, indent=2))
    return 0


def _cmd_show_policy(args: argparse.Namespace) -> int:
    store = _get_cli_store(args)
    snapshot = store.get_latest_policy(args.agent_id)
    if snapshot is None:
        print(f"No policy found for agent_id={args.agent_id!r}.", file=sys.stderr)
        return 2
    print(json.dumps(snapshot.to_dict(), indent=2))
    return 0


def _cmd_init_policy(args: argparse.Namespace) -> int:
    with open(args.actions, "r", encoding="utf-8") as f:
        action_payloads = json.load(f)
    if not isinstance(action_payloads, list) or not action_payloads:
        print("--actions file must contain a non-empty JSON list", file=sys.stderr)
        return 2
    actions = [Action.from_dict(item) for item in action_payloads]
    policy = SoftmaxPolicy.from_actions(actions, agent_id=args.agent_id)
    store = _get_cli_store(args)
    snapshot = policy.snapshot()
    store.store_policy(snapshot)
    print(json.dumps(snapshot.to_dict(), indent=2))
    return 0


def _cmd_task_intent(args: argparse.Namespace) -> int:
    if not args.intent.strip():
        print("--intent must not be empty", file=sys.stderr)
        return 2

    try:
        context = _redact_json(_load_context(args.context))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    store = _get_cli_store(args)
    snapshot = store.get_latest_policy(args.agent_id)
    if snapshot is None:
        print(
            f"No policy found for agent_id={args.agent_id!r}. "
            "Run `agent-learn policy-init` first.",
            file=sys.stderr,
        )
        return 2

    if args.episode_id and store.get_episode(args.episode_id, args.agent_id) is not None:
        print(
            f"Episode {args.episode_id!r} already exists for agent_id={args.agent_id!r}.",
            file=sys.stderr,
        )
        return 2

    policy = _policy_from_snapshot(snapshot)
    state: Optional[Any] = None
    if isinstance(policy, ContextualSoftmaxPolicy):
        state = context.get("phi")
        if state is None:
            print(
                "Contextual policies require --context with a 'phi' feature vector.",
                file=sys.stderr,
            )
            return 2

    try:
        decision = policy.choose(state)
    except (TypeError, ValueError) as exc:
        print(f"Unable to choose an action: {exc}", file=sys.stderr)
        return 2

    probabilities = {
        action.id: probability
        for action, probability in zip(snapshot.actions, decision.probabilities)
    }
    episode = Episode(
        agent_id=args.agent_id,
        user_input=redact(args.intent) or "",
        policy_id=snapshot.id,
        policy_version=snapshot.version,
        action_id=decision.action.id,
        action_logprob=decision.logprob,
        context_features=context,
        metadata={
            "status": "in_progress",
            "decision": {
                "action": _redact_json(decision.action.to_dict()),
                "probabilities": probabilities,
            },
        },
    )
    if args.episode_id:
        episode.id = args.episode_id
    store.store_episode(episode)

    print(
        json.dumps(
            {
                "episode_id": episode.id,
                "agent_id": episode.agent_id,
                "policy_id": episode.policy_id,
                "policy_version": episode.policy_version,
                "action_id": decision.action.id,
                "action": decision.action.to_dict(),
                "action_logprob": decision.logprob,
                "probabilities": probabilities,
            },
            indent=2,
        )
    )
    return 0


def _cmd_task_complete(args: argparse.Namespace) -> int:
    store = _get_cli_store(args)
    episode = store.get_episode(args.episode_id, args.agent_id)
    if episode is None:
        print(
            f"No episode found for episode_id={args.episode_id!r} "
            f"and agent_id={args.agent_id!r}.",
            file=sys.stderr,
        )
        return 2

    completed_at = datetime.now(timezone.utc)
    episode.assistant_output = redact(args.output) or ""
    episode.metadata["status"] = "completed"
    episode.metadata["completed_at"] = completed_at.isoformat()
    try:
        started_at = datetime.fromisoformat(episode.created_at)
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        episode.request_latency_ms = max(
            0,
            int((completed_at - started_at.astimezone(timezone.utc)).total_seconds() * 1000),
        )
    except ValueError:
        pass
    store.store_episode(episode)

    print(
        json.dumps(
            {
                "episode_id": episode.id,
                "agent_id": episode.agent_id,
                "status": "completed",
                "episode": episode.to_dict(),
            },
            indent=2,
        )
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    dispatch = {
        "train": _cmd_train,
        "score": _cmd_score,
        "policy": _cmd_show_policy,
        "policy-init": _cmd_init_policy,
        "init-policy": _cmd_init_policy,
        "task-intent": _cmd_task_intent,
        "task-complete": _cmd_task_complete,
    }
    handler = dispatch[args.command]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
