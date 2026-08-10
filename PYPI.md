# agent-learning

Evidence-driven decision SDK for AI agents. Each recurring agent-task decision
has one small, interpretable TaskPolicy over explicit executable alternatives.

TaskPolicies model reusable decisions among executable alternatives such as
models, skills, tools, workflows, or workloads. Factual questions, ordinary
chat, reporting, and learning automation are not policy tasks.

## How it works

The SDK improves decisions without LLM weight fine-tuning. There are no GPU
fine-tune jobs and no opaque update cycles. Four pieces run in the existing
Python process:

1. **TaskPolicy** owns `N` discrete actions and one persisted decision authority.
	`low` selects from learned softmax evidence; `full` evaluates a structured
	DecisionFrame against the same action set.

2. **DecisionResolver** applies hard constraints, confidence-weighted Bayesian
	evidence aggregation, Pareto elimination, robust utility, and information
	needs. A close result requires an explicit accept/reject tie-break.

3. **Score** evaluates each episode on-device for intent resolution, task
	adherence, and task completion. Azure AI evaluators remain opt-in.

4. **Learner** applies REINFORCE-with-baseline to low-authority TaskPolicy
	logits from observed outcomes and accept/reject feedback. Full-authority
	episodes are scored and audited but are not treated as softmax samples.

`task-policy-decide` closes the loop at execution time. It returns learned
feedback for low authority or an auditable decision certificate, information
needs, and any required tie-break for full authority. Both routes preserve the
same policy ID, version lineage, and action taxonomy.

It also returns a complexity-proportional autonomy assessment. A persisted
profile covers intent ambiguity, context variability, outcome observability,
decision impact, reversibility, and mandatory approval; action-space size is
derived. The resulting low, standard, high, or critical tier scales required
outcomes, Wilson confidence, reward, probability, margin, stable snapshots, and
drift-audit rate. Autonomous executions continue learning from observable
outcomes, while tier-scaled samples request user feedback to detect drift.
An explicit accepted-feedback episode is a separate durable authorization path:
it pins that action for the task policy and suppresses future feedback prompts
until the user explicitly rejects it.

Every episode, reward, run, and deployment is captured by the configured store — in-memory or local files by default, or Azure Cosmos DB — giving you a complete lineage and audit trail of how the policy evolved over time.
