<p align="center">
  <img src="images/362d5160ecde885f.png" alt="Agent Learning — Native reinforcement learning for AI agents" width="640" style="max-width:100%; height:auto;" />
</p>

# azure-agents-learning-sdk

Native reinforcement learning SDK for AI agents. An in-process
learner optimizes a small, interpretable policy over discrete agent
configuration choices (prompt variants, retrieval-k, tool selection
strategies, …) using Azure AI Evaluation judge metrics as the reward
signal.

## How it works

The SDK improves agents without LLM weight fine-tuning. There are no
GPU fine-tune jobs and no opaque update cycles — just three pieces
that run in your existing Python process:

1. The **policy** is a softmax distribution over `N` discrete
   actions (e.g., "use prompt template A", "use template B"). It
   lives in Python and updates in milliseconds.

   <img src="images/0f85e08d0c47cd01.png" alt="Policy selects one of N discrete actions" width="360" style="max-width:100%; height:auto;" />

2. Each episode is **judged** by three Azure AI Evaluation
   evaluators — `IntentResolutionEvaluator`, `TaskAdherenceEvaluator`,
   and `TaskCompletionEvaluator` — whose scores are combined into a
   single scalar reward.

   <img src="images/246d112f995b785a.png" alt="Three judge evaluators feed a single scalar reward" width="360" style="max-width:100%; height:auto;" />

3. A **REINFORCE-with-baseline** learner updates the policy logits
   directly from logged episodes. Updates are tiny gradient steps
   that run on CPU and persist through a pluggable store — in-memory
   or local files by default, with Cosmos DB optional.

   <img src="images/cc970c453583c982.png" alt="Policy quality improves with every batch of episodes" width="360" style="max-width:100%; height:auto;" />

Every episode, reward, run, and deployment is captured by the
configured store — in-memory or local files by default, or Cosmos DB —
giving you a complete lineage and audit trail of how the policy
evolved over time.

## Architecture

<p align="center">
  <img src="images/86caebedfa1e2ab5.png" alt="Architecture: Orchestrator turn → Cosmos DB → LearningRunner" width="520" style="max-width:100%; height:auto;" />
</p>

<details>
<summary>Text diagram (same flow, plain ASCII)</summary>

```
┌──────────────────────────────────────────────────────────┐
│  Orchestrator turn                                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ policy.choose() → Action                            │ │
│  │ EpisodeCapture.start(action_id=…, logprob=…)        │ │
│  │ … run agent, record tool calls …                    │ │
│  │ EpisodeCapture.end(assistant_output=…)              │ │
│  └─────────────────────────────────────────────────────┘ │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Cosmos DB: episodes, metrics, rewards, policies     │ │
│  └─────────────────────────────────────────────────────┘ │
│                       │                                  │
│                       ▼                                  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ LearningRunner.run_offline_batch(agent_id)          │ │
│  │   ┌─ evaluate (3 judges)                            │ │
│  │   ├─ shape (weighted sum + penalties → reward)      │ │
│  │   ├─ persist per-metric + aggregate rewards         │ │
│  │   └─ ReinforceLearner.update(policy, episodes)      │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

</details>

## Install

Released versions are published to PyPI:
<https://pypi.org/project/azure-agents-learning-sdk/>.

```bash
pip install azure-agents-learning-sdk
```

For local development against a checkout of this repository:

```bash
pip install -e .
```

## Configure

The SDK reads its configuration from environment variables. Every
variable is optional — with no configuration the SDK runs against an
in-memory store. The most important ones are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `AGENT_LEARNING_STORE_BACKEND` | Storage backend: `memory`, `cosmos`, or `local` | `memory` |
| `AGENT_LEARNING_COSMOS_ENDPOINT` | Cosmos DB account URL (only used when backend is `cosmos`) | unset |
| `AGENT_LEARNING_COSMOS_DATABASE` | Cosmos DB database name (only used when backend is `cosmos`) | `dq_rl` |
| `AGENT_LEARNING_LOCAL_STORE_DIR` | Directory for the `local` file backend | `./data/agent-learning/store` |
| `AGENT_LEARNING_JUDGE_ENDPOINT` | Azure OpenAI endpoint used by the judge | unset |
| `AGENT_LEARNING_JUDGE_DEPLOYMENT` | Judge deployment name | unset |
| `AGENT_LEARNING_W_INTENT` | Weight for intent-resolution reward | `0.4` |
| `AGENT_LEARNING_W_ADHERENCE` | Weight for task-adherence reward | `0.3` |
| `AGENT_LEARNING_W_COMPLETION` | Weight for task-completion reward | `0.3` |
| `AGENT_LEARNING_LR` | REINFORCE learning rate | `0.05` |
| `AGENT_LEARNING_BASELINE_DECAY` | EMA decay on the value baseline | `0.9` |

By default the SDK uses a volatile in-memory store. Set
`AGENT_LEARNING_STORE_BACKEND=cosmos` (together with the Cosmos
variables above) for durable Cosmos DB persistence, or `=local` to
persist to JSON files on disk. When the judge configuration is
missing, the SDK skips evaluations so unit tests still pass.

## Use it

```python
from agent_learning import (
    Action, EpisodeCapture, LearningRunner, SoftmaxPolicy,
)

actions = [
    Action(id="concise"),
    Action(id="detailed"),
]
policy = SoftmaxPolicy.from_actions(actions, agent_id="nba")

# At inference time
decision = policy.choose()
capture = EpisodeCapture()
ctx = capture.start(
    user_input="Summarise Q3 sales",
    policy_id=policy.snapshot().id,
    policy_version=policy.snapshot().version,
    action_id=decision.action.id,
    action_logprob=decision.logprob,
)
# … run your agent, then call:
capture.end(ctx, assistant_output="…")

# Periodically (cron, manual, event-driven)
runner = LearningRunner(policy=policy)
run = runner.run_offline_batch("nba", episode_limit=500)
```

The included CLI exposes the same flow:

```bash
agent-learn init-policy --agent-id dq --actions ./actions.json
agent-learn train --agent-id dq --limit 500
agent-learn policy --agent-id dq
```

## Examples

Three runnable examples in [examples/](examples/) build on each other.
All run in-process against the in-memory store with **no Azure
credentials** required.

| Example | Reward source | Objects it showcases |
| --- | --- | --- |
| [quickstart.py](examples/quickstart.py) | Stubbed constant | `SoftmaxPolicy`, built-in `ReinforceLearner`, `RewardShaper`, `LearningRunner` |
| [next_best_action.py](examples/next_best_action.py) | Simulated outcome | `ContextualSoftmaxPolicy` (contextual bandit), a contextual policy-gradient learner |
| [judged_optimization.py](examples/judged_optimization.py) | **Real Tier 1 judges** | `build_judges` (tiered judges), `JudgeScore`→`MetricResult`, routing + hallucination **shaping** penalties, rich `Episode` records |

Start with [judged_optimization.py](examples/judged_optimization.py) to
see the SDK's judge layer, reward shaping, metrics, policy, learner, and
episode capture working together end to end:

```bash
python examples/judged_optimization.py
```

## Layout

```
src/agent_learning/
├── types.py            # Durable record types
├── config.py           # Env-driven configuration
├── capture.py          # Episode capture hook
├── storage/            # LearningStore (Cosmos + local file + in-memory)
├── metrics/            # IntentResolution/TaskAdherence/TaskCompletion
├── rewards/            # Shaping + writer
├── policy/             # SoftmaxPolicy
├── learners/           # REINFORCE
├── training/           # End-to-end runner
└── cli.py              # `agent-learn` command-line
```

## Unit Testing

```bash
pytest -q
```

The test suite covers types, the in-memory store, the policy,
reward shaping, the REINFORCE learner, and an end-to-end training
loop with a stubbed metric evaluator.

## Use

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

```
