# Next Best Action Example — Summary

## Purpose
A "Next Best Action" agent (inspired by [`azure-agents-control-plane`](https://github.com/microsoft/azure-agents-control-plane/blob/main/src/next_best_action_agent.py)) that recommends the best customer-retention action given a customer's context. It adapts the reference's customer-churn use case to this SDK using a **contextual bandit** policy.

File: [next_best_action.py](next_best_action.py)

## Problem Setup
| Element | Definition |
|---|---|
| **Context** | Customer state: `churn_risk` (0–1) + `segment` (enterprise / smb / consumer) |
| **Features (`phi`)** | Length-5 vector: `[bias, churn_risk, is_enterprise, is_smb, is_consumer]` |
| **Actions** | `no_action`, `email_nudge`, `discount_offer`, `retention_call` |
| **Reward** | Simulated success in `[0,1]` → shaped to `[-1,1]`; optimal action varies by context |

The reward table has **no globally-best action**, so the agent must condition on context to do well.

## How It Maps to the SDK
- **`ContextualSoftmaxPolicy`** — linear softmax over `W · phi` (vs. the marginal `SoftmaxPolicy` in [quickstart.py](quickstart.py)).
- **Custom `ContextualReinforceLearner`** — the built-in `ReinforceLearner` only accepts the marginal policy, so the example implements the contextual policy gradient:

  $$\Delta W_k = \eta \,(R - b)\,(\mathbb{1}[k=\text{chosen}] - \pi_k)\,\phi$$

  with an **entropy bonus** (prevents collapse to a "safe" action) and a **batch-mean baseline** `b` (low-variance credit assignment).
- **Reused pipeline** — `LearningRunner` + `RewardShaper` + `RewardWriter` + `InMemoryStore`. A runner subclass converts the simulated outcome into the three judge metrics, so **no Azure credentials** are needed.

## Training Loop
Each round: sample fresh customers → policy picks an action → simulate outcome → one batched contextual REINFORCE update. Run for 200 rounds × 150 episodes.

## Results (deterministic, seed-fixed)
Optimal-action rate: **25% (random) → 90%**; mean reward **+0.05 → +0.31**.

| Customer context | Learned action | Confidence |
|---|---|---|
| High-risk Enterprise | `retention_call` | 0.51 |
| High-risk SMB | `discount_offer` | 0.68 |
| High-risk Consumer | `email_nudge` | 0.66 |
| Low-risk Enterprise | `no_action` | 0.78 |

## Key Learnings
1. A contextual policy needs a **custom learner** (the built-in one is marginal-only).
2. An **entropy bonus** is required, or the policy collapses onto one globally-safe action.
3. A **batch-mean baseline** converges far more reliably than the lagging EMA baseline for the contextual case.

## Run It
```powershell
python examples/next_best_action.py
```
