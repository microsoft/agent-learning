# Next Best Action Example — Summary

## Purpose
A **config-driven** "Next Best Action" agent (inspired by [`azure-agents-control-plane`](https://github.com/microsoft/azure-agents-control-plane/blob/main/src/next_best_action_agent.py)) that recommends the best action for a given context using a **contextual bandit** policy. The learning engine is generic; each use case is **externalised to a YAML file**, so the same script drives very different domains with no code changes.

- Engine: [next_best_action.py](next_best_action.py)
- Default use case: [next_best_action_retention_risk.yaml](next_best_action_retention_risk.yaml) — the reference's customer-churn use case adapted to this SDK.

## Problem Setup (Retention Risk)
| Element | Definition |
|---|---|
| **Context** | Customer state: `churn_risk` (0–1) + `segment` (enterprise / smb / consumer) |
| **Features (`phi`)** | Length-5 vector: `[bias, churn_risk, is_enterprise, is_smb, is_consumer]` |
| **Actions** | `no_action`, `email_nudge`, `discount_offer`, `retention_call` |
| **Reward** | Simulated success in `[0,1]` → shaped to `[-1,1]`; optimal action varies by context |

The reward table has **no globally-best action**, so the agent must condition on context to do well.

## The YAML Schema
Each use-case file declares five blocks the engine reads at startup:

| Block | Purpose |
|---|---|
| `actions` | The discrete action space (`id` + `description`). |
| `context.variables` | Each context dimension: `continuous` (sampled in `[low, high]`) or `categorical` (one-hot encoded). The feature vector is `phi = [bias] + continuous values + one-hot categoricals`, so `feature_dim` is derived automatically. |
| `environment` | The hidden reward table as ordered `rules`; the **first** rule whose `when` matches wins and the last (no `when`) is the default. `success` is the noise-free P(success) per action; `noise_std` adds Gaussian noise. The policy never sees this. |
| `probes` | Representative contexts printed before/after training. |
| `training` | `seed`, `rounds`, `episodes_per_round`, `learning_rate`, `entropy_bonus`. |

## How It Maps to the SDK
- **`ContextualSoftmaxPolicy`** — linear softmax over `W · phi` (vs. the marginal `SoftmaxPolicy` in [quickstart.py](quickstart.py)).
- **Custom `ContextualReinforceLearner`** — the built-in `ReinforceLearner` only accepts the marginal policy, so the example implements the contextual policy gradient:

  $$\Delta W_k = \eta \,(R - b)\,(\mathbb{1}[k=\text{chosen}] - \pi_k)\,\phi$$

  with an **entropy bonus** (prevents collapse to a "safe" action) and a **batch-mean baseline** `b` (low-variance credit assignment).
- **Reused pipeline** — `LearningRunner` + `RewardShaper` + `RewardWriter` + `InMemoryStore`. A runner subclass converts the simulated outcome into the three metric scores, so **no Azure credentials** are needed.

## Training Loop
Each round: sample fresh contexts → policy picks an action → simulate outcome → one batched contextual REINFORCE update. The retention use case runs 200 rounds × 150 episodes.

## Results — Retention Risk (deterministic, seed-fixed)
Optimal-action rate: **25% (random) → 90%**; mean reward **+0.05 → +0.31**.

| Customer context | Learned action | Confidence |
|---|---|---|
| High-risk Enterprise | `retention_call` | 0.51 |
| High-risk SMB | `discount_offer` | 0.68 |
| High-risk Consumer | `email_nudge` | 0.66 |
| Low-risk Enterprise | `no_action` | 0.78 |

## Other Bundled Use Cases
The same engine runs two more domains — each is a self-contained YAML with the identical five-block structure:

| Use case | Context | Learned mapping (at-risk state → action) |
|---|---|---|
| [next_best_action_patient_care.yaml](next_best_action_patient_care.yaml) | `readmission_risk` + `care_pathway` (post_op / chronic / behavioral) | post-op → `physician_escalation`, chronic → `nurse_call`, behavioral → `educational_message`; stable → `monitor` |
| [next_best_action_game_play.yaml](next_best_action_game_play.yaml) | `frustration` + `playstyle` (explorer / achiever / socializer) | explorer → `offer_hint`, achiever → `spawn_loot`, socializer → `matchmake_coop`; engaged → `stay_course` |

To add your own domain, copy any YAML and edit the five blocks — no Python changes required.

## Key Learnings
1. A contextual policy needs a **custom learner** (the built-in one is marginal-only).
2. An **entropy bonus** is required, or the policy collapses onto one globally-safe action.
3. A **batch-mean baseline** converges far more reliably than the lagging EMA baseline for the contextual case.
4. Externalising the domain to **declarative YAML** lets one learning engine serve many "next best action" problems.

## Run It
```powershell
# Default (retention risk)
python examples/next_best_action.py

# Or pick a use case explicitly
python examples/next_best_action.py examples/next_best_action_patient_care.yaml
python examples/next_best_action.py examples/next_best_action_game_play.yaml
```

Requires PyYAML (`pip install pyyaml`, or `pip install -e .[examples]`).
