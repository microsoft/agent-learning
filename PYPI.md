# agent-learning

Native reinforcement learning SDK for AI agents. An in-process Learner optimizes a small, interpretable TaskPolicy over discrete agent choices (understand intent and complete task by choosing the right outcome).

TaskPolicies model reusable decisions among executable alternatives such as
models, skills, tools, workflows, or workloads. Factual questions, ordinary
chat, reporting, and learning automation are not policy tasks.

## How it works

The SDK improves agents without LLM weight fine-tuning. There are no GPU fine-tune jobs and no opaque update cycles — just three pieces that run in your existing Python process:

1. **TaskPolicy** is a softmax distribution over `N` discrete actions (e.g., "take action A", "take action B", "take action C"). It lives in Python and updates in milliseconds.

2. **Score** evaluates each episode on-device with three stdlib scorers for intent resolution, task adherence, and task completion. Their scores are combined into a single scalar reward with no scoring endpoint or environment variables required. Azure AI evaluators remain available as an opt-in.

3. **Learner** applies REINFORCE-with-baseline to update TaskPolicy logits directly from logged episodes. Updates are tiny gradient steps that run on local compute and persist through a pluggable store — in-memory or local files by default, with Azure Cosmos DB optional.

`task-policy-decide` closes the loop at execution time by returning the selected
action plus historical correctness, reward, result summaries, and per-metric
quality feedback for the agent to use on its next delegated decision.

Every episode, reward, run, and deployment is captured by the configured store — in-memory or local files by default, or Azure Cosmos DB — giving you a complete lineage and audit trail of how the policy evolved over time.
