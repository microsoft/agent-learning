"""Create an isolated synthetic routing history and demonstrate exact replay.

No model, network, or real operational data is used. Numerical results are
seeded; UUIDs and reward/run timestamps vary between executions.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_learning import (
    Action,
    AuditStatus,
    Episode,
    LearnerConfig,
    LearningRunner,
    LocalFileStore,
    MetricName,
    MetricResult,
    RewardShaper,
    RewardWriter,
    ShapingConfig,
    SoftmaxPolicy,
    ToolCall,
    audit_run,
)


def create_demo(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=False)
    store = LocalFileStore(root)
    catalog = {"red": "handler-1", "green": "handler-2", "blue": "handler-3", "yellow": "handler-4"}
    stale_cache = {"red": "handler-1", "green": "old-handler", "blue": "old-handler", "yellow": "old-handler"}
    policy = SoftmaxPolicy.from_actions(
        [Action(id="cached", description="Use a stale synthetic cache"),
         Action(id="catalog", description="Use the current synthetic catalog")],
        agent_id="audit-demo", task_id="choose-lookup", rng=random.Random(7),
        initial_logits={"cached": 0.15, "catalog": -0.15}, max_logit_abs=3.0,
    )
    before = policy.snapshot()
    before.metadata["decision_authority"] = "low"
    # Set policy metadata before both capture and training.
    policy = SoftmaxPolicy.from_snapshot(before, rng=random.Random(7), max_logit_abs=3.0)
    store.store_policy(policy.snapshot())
    shaper = RewardShaper(ShapingConfig(
        intent_resolution_weight=0.1, task_adherence_weight=0.2, task_completion_weight=0.5,
        latency_penalty_threshold_ms=15000, latency_penalty_value=-0.1,
        route_correct_reward=0.2, route_wrong_penalty=-0.3, hallucinated_class_penalty=-0.25,
    ))
    writer = RewardWriter(store)
    keys = list(catalog)
    for index in range(40):
        key = keys[index % len(keys)]
        choice = policy.choose()
        actual = (stale_cache if choice.action.id == "cached" else catalog)[key]
        correct = actual == catalog[key]
        episode = Episode(
            id=f"lookup-{index:02d}", agent_id="audit-demo", task_id="choose-lookup",
            user_input=f"Find a synthetic handler for {key}", assistant_output=actual,
            action_id=choice.action.id, action_logprob=choice.logprob,
            policy_id=before.id, policy_version=before.version,
            intent_summary="Resolve a synthetic routing key",
            expected_outcome=catalog[key], execution_status="completed",
            result_summary="Correct handler" if correct else "Stale handler",
            tool_calls=[ToolCall(name=choice.action.id, arguments={"key": key}, result=actual)],
            metadata={"synthetic": True},
            created_at=(datetime(2026, 9, 16, tzinfo=timezone.utc) + timedelta(seconds=index)).isoformat(),
        )
        store.store_episode(episode)
        metrics = [
            MetricResult(
                metric=metric, score=score, normalized=score, status="completed",
                evaluator="synthetic-handler-check",
            )
            for metric, score in (
                (MetricName.INTENT_RESOLUTION, 1.0), (MetricName.TASK_ADHERENCE, 1.0),
                (MetricName.TASK_COMPLETION, float(correct)),
            )
        ]
        writer.write(episode, metrics, shaper.shape(metrics))
    run = LearningRunner(
        store=store, policy=policy, metrics=[], learner_config=LearnerConfig(
            learning_rate=0.8, baseline_decay=0.8, entropy_bonus=0.02,
            importance_clip=2.0, max_logit_abs=3.0, min_train_episodes=0,
        ),
    ).run_offline_batch("audit-demo", task_id="choose-lookup", score_missing=False)
    first = audit_run(LocalFileStore(root), "audit-demo", run.id)
    if first.status != AuditStatus.VERIFIED:
        raise RuntimeError(f"Initial replay failed: {first.reason_code}: {first.message}")

    # Deliberate later correction: append new reward IDs, preserving the
    # original selected reward and its recorded decomposition.
    revised_episode = store.get_episode("lookup-00", "audit-demo")
    if revised_episode is None:
        raise RuntimeError("Demo episode is missing.")
    revised = [MetricResult(
        metric=MetricName.TASK_COMPLETION, score=0.0, normalized=0.0,
        status="completed", evaluator="synthetic-later-correction",
    )]
    writer.write(revised_episode, revised, shaper.shape(revised))
    second = audit_run(LocalFileStore(root), "audit-demo", run.id)
    if second.status != AuditStatus.VERIFIED or second.contributions != first.contributions:
        raise RuntimeError("Appending a rescore changed replay of the historical run.")
    return {
        "agent_id": "audit-demo",
        "run_id": run.id,
        "store_dir": str(root.resolve()),
        "before_rescore": first.status.value,
        "after_rescore": second.status.value,
        "input_policy": second.input_policy,
        "output_policy": second.output_policy,
        "synthetic": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True, help="New directory; existing paths are refused.")
    args = parser.parse_args()
    try:
        manifest = create_demo(args.store_dir)
    except FileExistsError:
        parser.error("--store-dir already exists; choose a new directory. Existing data will not be replaced.")
    print(json.dumps(manifest, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
