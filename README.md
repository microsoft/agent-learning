# agent-learning

Native reinforcement learning SDK for AI agents. An in-process
Learner optimizes a small, interpretable TaskPolicy over discrete agent choices (e.g., "take action A", "take action B", "take action C") using on-device evaluation scores as the reward
signal by default.

<p align="center">
   <img src="images/agent-learning-loop.svg" alt="Animated TaskPolicy Score Learner loop: TaskPolicy chooses a task action, Score evaluates the episode, and Learner updates TaskPolicy" width="960" style="max-width:100%; height:auto;" />
</p>

## How it works

The SDK improves agents without LLM weight fine-tuning. There are no GPU fine-tune jobs and no opaque update cycles — just three pieces that run in your existing Python process:

1. **TaskPolicy** is a softmax distribution over `N` discrete
   actions (e.g., "take action A", "take action B", "take action C"). It lives in Python and updates in milliseconds.

   <img src="images/0f85e08d0c47cd01.png" alt="TaskPolicy selects one of N discrete actions" width="360" style="max-width:100%; height:auto;" />

2. **Score** evaluates each episode locally with three stdlib scorers for intent
   resolution, task adherence, and task completion. Their scores are combined
   into one scalar reward. No scoring endpoint or environment variable is
   required. Configured Azure AI evaluators remain available as an opt-in.

   <img src="images/246d112f995b785a.png" alt="Three evaluator scores feed a single scalar reward" width="360" style="max-width:100%; height:auto;" />

3. **Learner** applies REINFORCE-with-baseline to update TaskPolicy logits
   directly from stored episodes. Updates are tiny gradient steps
   that run on local compute and persist through a pluggable store — in-memory
   or local files by default, with Azure Cosmos DB optional.

   <img src="images/cc970c453583c982.png" alt="Policy quality improves with every batch of episodes" width="360" style="max-width:100%; height:auto;" />

Every episode, reward, run, and deployment is captured by the
configured store — in-memory or local files by default, or Azure Cosmos DB —
giving you a complete lineage and audit trail of how the policy
evolved over time.

## Install

### Windows CLI

For a Python-independent installation, download `agent-learn.exe` or the
standalone installer from the
[latest GitHub release](https://github.com/microsoft/agent-learning/releases/latest).
The installer can add its installation directory to your user `PATH`, so
`agent-learn` works from PowerShell or Command Prompt without Python or `pip`.

```powershell
agent-learn.exe --help
```

### Python SDK

Released versions are published to PyPI:
<https://pypi.org/project/agent-learning/>.

## Functional Testing

A good way to see the SDK in action is to run the
interactive capture scenario first, followed by the offline batch update:

```powershell
python tests/functional_cli_interactive.py
python tests/functional_cli_batch.py
```

## Usage

The `agent-learn` CLI provides the current task-learning-loop operations:

```text
agent-learn list
agent-learn tasks-list <agent_id>
agent-learn task-episodes-count <agent_id> [--task-id <task_id>]
agent-learn task-episodes-list <agent_id> [--task-id <task_id>] [--limit <1-500>] [--include-incomplete]
agent-learn task-policy-init --agent-id <agent_id> --task-id <task_id> --actions ./actions.json
agent-learn task-episode-register --agent-id <agent_id> --task-id <task_id> --episode ./episode.json
agent-learn score --agent-id <agent_id> [--task-id <task_id>] [--limit <1-500>]
agent-learn train --agent-id <agent_id> [--task-id <task_id>] [--limit <1-500>] [--start-date <date>] [--end-date <date>] [--skip-scoring]
agent-learn task-policy --agent-id <agent_id> --task-id <task_id>
```
