# agents-learning-sdk

Native reinforcement learning SDK for AI agents. An in-process learner optimizes a small, interpretable policy over discrete agent choices (understand intent and complete task by choosing the right outcome).

## How it works

The SDK improves agents without LLM weight fine-tuning. There are no GPU fine-tune jobs and no opaque update cycles — just three pieces that run in your existing Python process:

1. The **policy** is a softmax distribution over `N` discrete actions (e.g., "take action A", "take action B", "take action C"). It lives in Python and updates in milliseconds.

2. Each episode is **evaluated** by three AI Evaluation evaluators — `IntentResolutionEvaluator`, `TaskAdherenceEvaluator`, and `TaskCompletionEvaluator` — whose scores are combined into a single scalar reward.

3. A **REINFORCE-with-baseline** learner updates the policy logits directly from logged episodes. Updates are tiny gradient steps that run on local compute and persist through a pluggable store — in-memory or local files by default, with Azure Cosmos DB optional.

Every episode, reward, run, and deployment is captured by the configured store — in-memory or local files by default, or Azure Cosmos DB — giving you a complete lineage and audit trail of how the policy evolved over time.
