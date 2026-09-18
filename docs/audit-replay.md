# Audit and replay a recorded policy update

An operator should be able to answer: **Which recorded outcomes changed this
policy, and do those inputs reproduce the stored update?**

`audit_run` reconciles a successful native REINFORCE training run without
executing agent actions, calling scorers, writing records, or changing the
active policy. It uses the exact input/output snapshot IDs and ordered
episode/reward references recorded by `LearningRunner`.

## Run the local demo

From this checkout, use Python 3.10+ with the project dependencies installed
(`python -m pip install -e .`). Neither example requires a model, credentials,
Cosmos DB, nor a network service.

```powershell
$store = Join-Path .\data ("audit-demo-" + [guid]::NewGuid().ToString())
$demo = python .\examples\audit_replay_demo.py --store-dir $store | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw "Demo failed" }

python .\examples\audit_run.py `
  --store-dir $store `
  --agent-id $demo.agent_id `
  --run-id $demo.run_id
```

The demo refuses any existing destination, including an empty directory.
It never deletes or resets a store. Each run creates its own local history.

The synthetic agent executes either a stale-cache lookup or a current-catalog
lookup for a routing key. It records 40 sampled choices, actual dictionary
lookup results, fixture-based metric scores, aggregate rewards, and one
training update. These are real executions of small local functions but
**synthetic outcomes**, not evidence about real retrieval systems or agent
quality.

The demo first verifies that the update reconciles. It then appends a later
scoring correction with new reward IDs and verifies that replay of the
original run is unchanged. Numerical results are seeded; UUIDs and
reward/run timestamps vary.

The operator report includes:

- Input and output snapshot IDs, versions, logits, and action probabilities.
- Queried versus consumed episode counts.
- Episode IDs linked to the exact reward IDs used by training.
- Rewards, baseline advantages, behavior probabilities, importance weights,
  and episode contributions to each action's proposed logit change.
- Recorded metric contributions and penalties, when available.
- Separate entropy and clipping adjustments.
- Reconciliation status, per-field differences, and missing-evidence warnings.

The text report shows the ten largest episode contributions by absolute
reward-delta magnitude. JSON preserves **all** contributions in consumption
order:

```powershell
python .\examples\audit_run.py `
  --store-dir $store `
  --agent-id $demo.agent_id `
  --run-id $demo.run_id `
  --format json
```

Use the same operator script for an existing local store and a run ID returned
by `agent-learn train`. There is no new `agent-learn audit-run` subcommand.
Both example scripts load the current checkout's SDK source.

## Python API

```python
from agent_learning import AuditStatus, LocalFileStore, audit_run

store = LocalFileStore(r".\data\my-learning-store")
report = audit_run(store, agent_id="agent", run_id="recorded-run-id")
if report.status is AuditStatus.VERIFIED:
    print(report.summary)
else:
    print(report.reason_code, report.message, report.differences)
```

The operation supports the exact `LocalFileStore` and `InMemoryStore` types.
Other implementations, including Cosmos and subclasses, are rejected before
I/O because their initialization or custom behavior may have side effects.
There is no implicit default-store selection.

Storage reads use existing backend behavior. A local file that cannot be
decoded/read is logged by `LocalFileStore` and reported as unavailable.
Audit validates required local JSON fields from the same read that it decodes,
before ordinary SDK loaders can supply legacy defaults or coerce types.
Malformed fields and numeric decoding overflow produce a structured
`mismatch` with reason `invalid_record`. Ordinary SDK loading remains
backward-compatible. Other storage exceptions
propagate to the caller instead of becoming a successful or neutral audit.

## What permits verification

The run must be successful and record:

- `policy_id`: exact input/target snapshot ID.
- `output_policy_id`: exact resulting snapshot ID.
- `episode_ids`: the original queried batch.
- `consumed_inputs`: ordered `{episode_id, reward_id}` pairs actually used.
- `hyperparameters.learner`: applied learning rate, baseline decay, entropy
  bonus, and importance clip.
- `hyperparameters.policy.max_logit_abs`: the policy's effective clipping
  ceiling, not a similarly named but unused learner setting.
- `metadata.lineage_version: 1` and
  `metadata.replay_implementation: "reinforce_softmax_v1"`.
- The original learner summary and `metadata.policy_version`.

The implementation validates ownership, action definitions, receipt order,
counts, reward source, and finite numeric inputs before replay. Required
snapshot fields (including baseline, version, counters, logits, and action
definitions) and the selected reward's value must be explicitly recorded.
Counters must be nonnegative JSON integers, not booleans, strings, or
fractional numbers. Numeric values must be finite JSON numbers, not strings
or booleans. Missing fields are not replaced with zero to establish
verification. It loads
selected rewards by exact ID, never by "newest now." Repeated episodes may
appear if they are repeated in the queried batch; their order and selected
reward must remain consistent.

Replay invokes the real native learner on a detached input snapshot, with
every configuration field supplied explicitly. Current scoring settings,
default-store settings, and learner environment overrides are not used.
Older behavior-policy IDs are permitted: offline samples need not originate
from the current training snapshot.

An explicitly recorded `action_logprob: null` is reported as an unknown
behavior probability; an omitted local JSON field is incomplete evidence.
The native importance-weight fallback is 1; it does not mean the probability
of the original action was 1. Known probabilities still use the native
probability floor and importance cap.

Replay compares the summary, raw deltas, baseline, final logits,
probabilities, version, and counters. Float tolerances are relative `1e-9`
and absolute `1e-12`; identities and integer counters are exact. Generated
replay UUIDs and timestamps are not compared to the persisted output UUID
and timestamp.

An empty supported receipt (`[]`) is distinct from a missing receipt
(`null`). A successful no-op must reference the same input/output snapshot
and is reported as **no update applied**.

## Status and process exit codes

| Status | Script exit code | Meaning |
|---|---:|---|
| `verified` | 0 | Referenced records reproduce the native update, including a valid no-op. |
| `mismatch` | 1 | Invalid/inconsistent records or numerical disagreement. Inspect reason codes and differences. |
| `insufficient_lineage` | 2 | Old/unsupported format, missing configuration, unsupported store/policy/learner, or non-successful run. |
| `missing_record` | 3 | A referenced run, snapshot, episode, or reward is missing or unreadable. |

Argument errors also exit 2 with a usage message; an unexpected OS-level read
error reaching the script exits 4 with an error message. Successful runs
alone do not imply successful audits.

Old runs remain readable but are **not** replay-verified by guessing
hyperparameters from current defaults or finding a nearby output version.
Contextual policies, custom learners/subclasses, and full-authority reasoned
decisions are not supported by this replay implementation.

## Attribution is numerical, not causal

The report isolates each episode's reward term by running the native learner
on that one episode from the **same original snapshot and baseline**, with
entropy disabled, then scaling to the original batch size. Terms are totaled
using `math.fsum` and reported as `summary.attribution_reward_totals`.
The native reward-only batch accumulates and scales in a different order,
so cancellation can produce different floating-point rounding.

`summary.attribution_rounding_residuals` records the native reward-only
batch delta minus those attribution totals. The difference between the
original update and the native reward-only batch remains the separate
entropy contribution; rounding is not mislabeled as entropy.
If a residual exceeds the report's comparison tolerances,
`summary.attribution_status` is `rounding_residual` and a warning is shown;
otherwise it is `reconciled`. These attribution diagnostics do not change
the main replay status or relax any persisted-state comparisons.

The report's reconciliation is: episode reward totals plus the rounding
residual, entropy contribution, and clipping adjustment equal the recorded
applied delta (subject to floating-point arithmetic).

This reuses the actual algorithm, rather than maintaining a second gradient
formula. It requires one additional small native update per episode and an
additional reward-only batch, all on private copies. The action space is
small; no model inference is involved.

Raw learner deltas are proposed changes. The policy clips resulting logits,
so the actual snapshot difference can be smaller. Clipping is shown as a
batch-level adjustment, not arbitrarily assigned to individual episodes.

The report explains numerical contributions conditional on the recorded
batch. It does not attribute causality among an episode's internal tool
calls and does not establish that the update algorithm itself is correct.

## Interpretability and provenance boundaries

This is **policy-level interpretability**: it explains changes to a small
explicit action distribution. It does not explain the frozen model's
internal reasoning, verify that a response is true, calibrate an evaluator,
prove that logged propensities reflect real sampling, or certify that a
policy is safe for autonomous execution.

The chosen aggregate reward's stored metric contributions and penalties are
shown as **recorded composition**. Current raw metric rows are not joined as
if they were the exact historical evaluation event. Local stores append
metric results while Cosmos upserts them, and raw metrics do not carry
evaluation-batch IDs. If composition metadata is missing or malformed,
the report warns separately; numerical replay can still reconcile.

All current stores permit same-ID upserts. IDs provide references, not
immutability. Persist the input snapshot before training, retain referenced
records, and use new reward IDs when rescoring. An edit that preserves the
numerical result may still produce `verified`; the tool cannot prove the
records were never altered.

The report omits prompts, tool arguments, and arbitrary episode metadata.
IDs, numerical state, and recorded reward components are still operational
data; review reports before sharing them.

Immutable event sourcing, tamper-evident hashes, raw-evaluator replay,
cloud-store audit, policy promotion/rollback, and model explanations remain
outside this example's scope.
