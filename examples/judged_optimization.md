# Judged Optimization Example — Summary

## Purpose
The end-to-end demonstration of the SDK's **judge layer** driving learning. Where [quickstart.py](quickstart.py) stubs the reward and [next_best_action.py](next_best_action.py) simulates an outcome, this example scores every episode with the SDK's *real* Tier 1 judges (pure Python standard library, zero dependencies) and lets those scores — not a hand-written oracle — update the policy.

File: [judged_optimization.py](judged_optimization.py)

## Problem Setup
A support agent must resolve a password-reset ticket in a required format. The policy picks one of three response templates; the judges decide how good each answer is.

| Element | Definition |
|---|---|
| **Contract** | `required_substrings` (ticket id + `password`), `forbidden_substrings` (`I don't know`, `cannot help`), length bounds |
| **Expected tokens** | `reset`, `verify`, `identity`, `link` (completion coverage) |
| **Actions** | `template_rich`, `template_terse`, `template_offtopic` |
| **Reward** | Produced by the stdlib judges, shaped to `[-1, 1]` with routing + hallucination penalties |

## All six SDK building blocks in one file
| Object | How the example uses it |
|---|---|
| **Judges** | `build_judges(JudgeRuntimeConfig(tier="stdlib"))` returns the intent/adherence/completion trio. Adherence and completion are deterministic rule engines; the intent head is `fit()` in-process on a tiny labelled corpus. |
| **Metrics** | A small adapter maps each `JudgeScore` onto the `MetricResult` shape the reward pipeline already understands — the one bridge that lets *any* judge tier feed rewards. |
| **Shaping** | `RewardShaper` folds in the `routing_correct` and `hallucinated_class` penalties that `shape_episode_reward` never passes — so those `ShapingConfig` terms finally fire. |
| **Policy** | A marginal `SoftmaxPolicy` over the three templates. |
| **Learning** | The **built-in** `ReinforceLearner` (not a re-implementation) consumes the shaped reward. |
| **Episodes** | Full `Episode` records carry the contract, expected tokens, `ToolCall`s, conversation history, and context features the judges read. |

## How the judges are wired in (no SDK changes)
The example subclasses `LearningRunner` and overrides two hooks — the same extension pattern the other examples use:

- `evaluate_episode` → score the episode with the three judges instead of Azure evaluators.
- `score_and_record` → call `RewardShaper.shape(..., routing_correct=..., hallucinated_class=...)` so the routing and hallucination penalties contribute.

The core runner, shaper, writer, learner, and storage are all used **unmodified**.

## Results (deterministic, seed-fixed)
```
=== Policy BEFORE training ===
  uniform prior            -> best=template_rich      | rich=0.33  terse=0.33  offtopic=0.33

Training for 30 rounds x 30 episodes (stdlib judges) ...
  round   0: mean_reward=-0.020  P(template_rich)=0.40
  round  10: mean_reward=+0.644  P(template_rich)=0.87
  round  20: mean_reward=+0.909  P(template_rich)=0.94
  round  29: mean_reward=+0.975  P(template_rich)=0.95

=== Policy AFTER training ===
  learned                  -> best=template_rich      | rich=0.95  terse=0.03  offtopic=0.02
```

Per-template judge decomposition (single ticket):

| Template | Reward | Intent | Adherence | Completion |
|---|---|---|---|---|
| `template_rich` | **+0.97** | 0.87 | 1.00 | 1.00 |
| `template_terse` | −0.36 | 0.20 | 0.50 | 0.00 |
| `template_offtopic` | −1.00 | 0.06 | 0.33 | 0.00 |

The policy shifts from a uniform prior to **0.95** on `template_rich` in 30 batches, driven entirely by the judge scores.

## Key Learnings
1. The tiered judges are usable as the reward source today — you bridge `JudgeScore → MetricResult` and the rest of the pipeline is unchanged.
2. Tier 1 needs **no external dependencies and no Azure**: adherence and completion are rule engines; the intent head fits in-process.
3. The `routing_correct` / `hallucinated_class` shaping terms are real levers — pass them into `RewardShaper.shape` and they push a misrouted, hallucinating answer to the reward floor.

## How It Differs From the Other Examples
| | [quickstart.py](quickstart.py) | [next_best_action.py](next_best_action.py) | judged_optimization.py |
|---|---|---|---|
| Reward source | Stubbed constant | Simulated outcome | **Real stdlib judges** |
| Policy | `SoftmaxPolicy` | `ContextualSoftmaxPolicy` | `SoftmaxPolicy` |
| Learner | Built-in `ReinforceLearner` | Custom contextual learner | Built-in `ReinforceLearner` |
| Shaping penalties | latency only | none | **routing + hallucination** |

## Swapping the judge tier
Change one line to move up the tier stack (extras required):

```python
build_judges(JudgeRuntimeConfig(tier="nlp"))   # TF-IDF + scikit-learn  ([nlp] extra)
build_judges(JudgeRuntimeConfig(tier="slm"))   # Phi-4-mini ONNX        ([slm] extra)
build_judges(JudgeRuntimeConfig(tier="llm"))   # azure-ai-evaluation    ([llm] extra)
```

The `JudgeScore → MetricResult` adapter and everything downstream stay the same. See [../docs/DESIGN.md](../docs/DESIGN.md) for the full four-tier design.

## Run It
```powershell
python examples/judged_optimization.py
```
