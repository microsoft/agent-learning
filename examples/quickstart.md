# Quickstart Example — Summary

## Purpose
The minimal end-to-end demonstration of the SDK's native reinforcement-learning loop. It optimizes a single global choice between two prompt strategies using a **marginal (non-contextual) bandit** policy, entirely in-process with an in-memory store.

File: [quickstart.py](quickstart.py)

## Problem Setup
| Element | Definition |
|---|---|
| **Context** | None — one action is globally best (marginal bandit) |
| **Actions** | `concise` (short, direct prompt) vs. `detailed` (verbose chain-of-thought prompt) |
| **Reward** | Stubbed: `detailed` scores well (1.0), `concise` poorly (0.2); shaped to `[-1,1]` |

## How It Maps to the SDK
- **`SoftmaxPolicy`** — a discrete softmax bandit over the two actions, with per-action logits.
- **Built-in `ReinforceLearner`** — REINFORCE-with-baseline (EMA value baseline + entropy bonus), configured via `LearnerConfig(learning_rate=0.4, entropy_bonus=0.01)`.
- **`LearningRunner` pipeline** — a `_DemoRunner` subclass overrides `evaluate_episode` to return stubbed metric scores (Intent Resolution, Task Adherence, Task Completion), so **no Azure credentials** are needed. `RewardShaper` + `RewardWriter` + `InMemoryStore` handle shaping and persistence.

## Flow
1. Build a `SoftmaxPolicy` from two `Action`s and store its snapshot.
2. Simulate 40 episodes: TaskPolicy chooses an action; a synthetic reward is attached in metadata.
3. `run_offline_batch("demo", episode_limit=200)` scores episodes, shapes rewards, and applies one learner update.
4. Print run status plus the action probabilities before and after learning.

## Results (deterministic, seed-fixed)
```
Run status:         succeeded
Episodes used:      40
Mean reward:        0.256
Probs before:       {'concise': 0.5, 'detailed': 0.5}
Probs after:        {'concise': 0.434, 'detailed': 0.566}
```
The policy shifts toward the higher-reward `detailed` action (0.5 → 0.566) after a single batch update.

## How It Differs From `next_best_action.py`
| | `quickstart.py` | [next_best_action.py](next_best_action.py) |
|---|---|---|
| Policy | `SoftmaxPolicy` (marginal) | `ContextualSoftmaxPolicy` (contextual) |
| Best action | Globally fixed | Depends on customer context |
| Learner | Built-in `ReinforceLearner` | Custom `ContextualReinforceLearner` |

## Run It
```powershell
python examples/quickstart.py
```
