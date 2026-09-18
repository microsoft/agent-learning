"""Inspect one local training run without modifying the store."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_learning import AuditReport, AuditStatus, LocalFileStore, audit_run

_EXIT_CODES = {
    AuditStatus.VERIFIED: 0,
    AuditStatus.MISMATCH: 1,
    AuditStatus.INSUFFICIENT_LINEAGE: 2,
    AuditStatus.MISSING_RECORD: 3,
}


def format_report(report: AuditReport) -> str:
    lines = [
        f"{report.status.value.upper()} [{report.reason_code}]",
        report.message,
        f"Agent: {report.agent_id} | Task: {report.task_id} | Run: {report.run_id}",
    ]
    if report.input_policy is not None and report.output_policy is not None:
        before, after = report.input_policy, report.output_policy
        lines.extend([
            f"Input snapshot:  {before['id']} (v{before['version']})",
            f"Output snapshot: {after['id']} (v{after['version']})",
            "Action probabilities and logits (before -> recorded after):",
        ])
        for action, probability in before["probabilities"].items():
            lines.append(
                f"  {action}: {probability:.6f} -> {after['probabilities'][action]:.6f}; "
                f"logit {before['logits'][action]:+.6f} -> {after['logits'][action]:+.6f}"
            )
    if report.summary:
        lines.append(
            f"Queried: {report.summary['queried_episodes']} | "
            f"Consumed: {report.summary['consumed_episodes']} | No update: {report.summary['no_update']}"
        )
    if report.contributions:
        lines.append("Largest episode reward contributions (up to 10; JSON contains all, in processing order):")
        ranked = sorted(
            report.contributions, key=lambda item: -sum(abs(v) for v in item["reward_deltas"].values()),
        )
        for item in ranked[:10]:
            delta = ", ".join(f"{aid}={value:+.6f}" for aid, value in item["reward_deltas"].items())
            lines.append(
                f"  {item['episode_id']} -> {item['reward_id']}: action={item['action_id']} "
                f"reward={item['reward']:+.3f} advantage={item['advantage']:+.3f} "
                f"importance={item['importance_weight']:.3f}; {delta}"
            )
            breakdown = item["reward_decomposition"]
            if breakdown is not None:
                parts = [f"{c['metric']}={c['contribution']:+.3f}" for c in breakdown["metric_contributions"]]
                parts.extend(f"{p['kind']}={p['value']:+.3f}" for p in breakdown["penalties"])
                lines.append(f"    Recorded reward composition: {', '.join(parts)}")
        if "entropy_deltas" in report.summary:
            lines.append(f"Entropy contribution: {report.summary['entropy_deltas']}")
        if "attribution_rounding_residuals" in report.summary:
            lines.append(f"Attribution rounding residual: {report.summary['attribution_rounding_residuals']}")
        if "clipping_adjustments" in report.summary:
            lines.append(f"Clipping adjustment: {report.summary['clipping_adjustments']}")
    for difference in report.differences:
        lines.append(
            f"DIFFERENCE {difference['field']}: recorded={difference['recorded']} "
            f"replayed={difference['replayed']}"
        )
    lines.extend(f"WARNING: {warning}" for warning in report.warnings)
    lines.append("Scope: numerical consistency, not immutable provenance or an explanation of model reasoning.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()
    if not args.store_dir.is_dir():
        parser.error("--store-dir must be an existing local store directory")
    try:
        report = audit_run(LocalFileStore(args.store_dir), args.agent_id, args.run_id)
    except OSError as exc:
        print(f"Unable to read the local store: {exc}", file=sys.stderr)
        return 4
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, allow_nan=False))
    else:
        print(format_report(report))
    return _EXIT_CODES[report.status]


if __name__ == "__main__":
    raise SystemExit(main())
