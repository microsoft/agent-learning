---
title: Complexity-proportional autonomy
description: Declare and score intent and decision complexity so agent-learning requires proportionally stronger evidence before autonomous execution.
author: Microsoft
ms.date: 2026-08-10
ms.topic: concept
keywords:
  - agent autonomy
  - decision complexity
  - intent complexity
  - evidence thresholds
  - human approval
estimated_reading_time: 10
---

# Complexity-proportional autonomy

A single autonomy threshold is too permissive for consequential decisions and
too expensive for routine ones. `agent-learning` therefore separates two
questions:

1. How complex is this recurring intent and decision?
2. Has the policy collected enough evidence for that complexity?

Complexity changes the required evidence. It never changes observed rewards,
correctness labels, policy probabilities, or deterministic approval rules.

## Declare complexity; do not infer it from prose

The agent must not grade the complexity of its own answer from free text. That
would make autonomy sensitive to wording and create an incentive to understate
risk. Instead, each decision policy has a validated, persisted complexity
profile. The number of actions is derived by the SDK.

```json
{
  "intent_ambiguity": "medium",
  "context_variability": "variable",
  "outcome_observability": "delayed",
  "decision_impact": "high",
  "reversibility": "costly",
  "requires_human_approval": false,
  "rationale": "Architecture choice affects a shared production integration."
}
```

| Field | Values | Meaning |
|---|---|---|
| `intent_ambiguity` | `low`, `medium`, `high` | How much interpretation is required before alternatives are comparable. |
| `context_variability` | `stable`, `variable`, `dynamic` | How often material conditions change across executions. |
| `outcome_observability` | `direct`, `delayed`, `subjective` | How independently and quickly success can be measured. |
| `decision_impact` | `low`, `medium`, `high`, `critical` | The scope of cost, quality, safety, or downstream consequences. |
| `reversibility` | `reversible`, `costly`, `irreversible` | The cost or feasibility of undoing the selected action. |
| `requires_human_approval` | Boolean | A deterministic override. When true, learned autonomy is never granted. |
| `rationale` | String | Human-readable justification for review and audit. |

Missing profiles use a conservative `standard` default. New integrations
should always declare a profile. Existing policies can be configured without
creating a learned snapshot; changing configuration must not fabricate policy
stability.

## Complexity score

Intent complexity is the sum of three ordinal values:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Ambiguity | low | medium | high |
| Variability | stable | variable | dynamic |
| Observability | direct | delayed | subjective |

Decision complexity adds:

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Impact | low | medium | high | critical |
| Reversibility | reversible | costly | irreversible | — |
| Action-space size | 2 actions | 3–4 actions | 5+ actions | — |

The total ranges from 0 to 13:

| Total | Tier |
|---:|---|
| 0–2 | `low` |
| 3–6 | `standard` |
| 7–9 | `high` |
| 10–13 | `critical` |

Risk floors prevent averaging away a severe dimension:

- critical impact always produces the `critical` tier;
- high-impact irreversible decisions are `critical`;
- high ambiguity plus subjective outcomes are at least `high`.

## Proportional evidence

Every statistical criterion remains conjunctive. A policy earns statistical
autonomy only when all criteria for its tier pass.

| Criterion | Low | Standard | High | Critical |
|---|---:|---:|---:|---:|
| Scored outcomes | 3 | 20 | 50 | 100 |
| 95% Wilson correctness lower bound | 40% | 90% | 95% | 97.5% |
| Mean aggregate reward | > 0.00 | > 0.00 | > 0.10 | > 0.20 |
| Winner probability | Not gated | 60% | 75% | 85% |
| Margin over runner-up | Not gated | 15 points | 30 points | 45 points |
| Consecutive trained snapshots with same winner | 1 | 3 | 4 | 5 |
| Autonomous drift-audit rate | 5% | 10% | 25% | 50% |

The low tier is calibrated for bounded, reversible preferences with direct
feedback: three unanimous, positively rewarded outcomes can earn autonomy after
one policy update, while mixed correctness evidence does not. Higher tiers keep
strong probability and stability gates. Their Wilson thresholds usually require
more correctness labels than the scored-outcome minimum. Global environment
variables can raise or lower numeric thresholds, but the resolved values are
always returned in `autonomy.criteria` for audit.

## Explicit user acceptance

Explicit acceptance is authorization, not a statistical sample. A completed
episode with `metadata.feedback_status: accepted` grants autonomy immediately
for that `(agent_id, task_id)` decision and pins its accepted `action_id`, even
when the policy logits currently prefer another action. The assessment returns
`authorization_basis: user_acceptance`, sets the audit rate to zero, and does
not request routine or drift-audit feedback again.

The newest explicit feedback wins. A later completed episode with
`metadata.feedback_status: rejected` revokes the acceptance and returns the
policy to its statistical evidence gates. `requires_human_approval: true`
continues to block autonomy because it represents a deterministic approval
requirement rather than a reusable policy preference.

## Human approval and execution authority

`requires_human_approval: true` adds a failing autonomy criterion regardless of
probability or outcome history. The SDK reports the block; it never silently
converts approval into a statistical threshold.

The profile does not grant operating-system, cloud, financial, clinical,
security, or destructive-action permission. Existing deterministic approvals
remain authoritative even when the policy is statistically autonomous.

## Lifecycle

1. Initialize a policy with `--complexity-profile`, or configure an existing
   policy with `task-policy-complexity-set`.
2. `task-policy-decide` resolves the profile, derived action-space points, tier,
   and proportional threshold set.
3. The response exposes `autonomy.complexity`, every threshold, every observed
   value, and the resulting mode.
4. Statistically autonomous execution continues recording observable outcomes
   and requesting tier-proportional drift audits. Explicitly accepted policies
   continue recording outcomes without requesting feedback.
5. Negative outcomes, a changed winner, or a stricter profile can immediately
   return the next decision to supervised mode.

Complexity profile changes update policy configuration in place. They do not
increment policy version or count as a stable learned snapshot.