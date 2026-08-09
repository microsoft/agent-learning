# agent-learning

Helping agents make better decisions with measurable feedback.

`agent-learning` is a lightweight decision layer for an existing agent. It
learns which option works best for a recurring task from a small, explicit set
of executable alternatives, such as prompt variants, retrieval depths, tools,
models, workflows, or escalation paths.

The foundation model stays frozen. There are no GPU training jobs and no
hidden prompt rewrites. The agent makes an explicit choice, records the
user-visible outcome, scores the evidence, and applies a small CPU update to
the next decision.

<p align="center">
   <img src="images/agent-decision-making.svg" alt="Animated agentic decision loop: choose a bounded action, execute it, score the observed outcome, and improve the next decision" width="960" style="max-width:100%; height:auto;" />
</p>

## The decision loop

1. **Frame a reusable decision.** Define a stable decision context and at
   least two actions the agent can actually execute. A TaskPolicy owns the
   probability distribution for that `(agent_id, task_id)` pair.

2. **Choose and execute.** `task-policy-decide` samples an action from the
   active softmax policy and returns learned feedback from earlier attempts.
   The agent uses that feedback and executes the selected action.

3. **Observe and score.** A completed episode preserves the decision context,
   selected action, output, result summary, latency, and independently
   supported correctness evidence. Local scorers measure intent resolution,
   task adherence, and task completion and shape them into one reward.

4. **Improve the next decision.** REINFORCE-with-baseline nudges a few policy
   logits, then persists a new policy snapshot. Better-than-usual choices gain
   probability, worse-than-usual choices lose probability, and exploration
   remains available.

Everything runs in the existing Python process. Scoring is local by default;
configured Azure AI evaluators remain available as an opt-in. Stores can be
in-memory, local JSON files, or Azure Cosmos DB.

## What counts as a decision

A TaskPolicy is appropriate only when all of these are true:

- The agent must choose among at least two explicit executable alternatives.
- The alternatives are stable enough to reuse on future requests.
- The choice can affect quality, correctness, latency, cost, safety, or
  completion.
- An observable outcome can later score whether the choice was useful.

Good examples include choosing a retrieval strategy, model, tool, workflow,
Azure workload, or escalation path for a concrete use case. Factual questions,
ordinary chat, summaries, reporting, and the learning automation itself are not
decision policies. Repetition without an executed outcome is not feedback.

Before every eligible execution, the decision response includes the selected
and currently recommended actions, policy version and probability, correctness
rate, mean reward, metric scores, and recent result summaries. This makes the
learned evidence useful during execution instead of producing a policy that is
never consumed.

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

```shell
pip install agent-learning
```

## Quickstart: improve one recurring decision

Use one durable store across CLI processes:

```powershell
$env:AGENT_LEARNING_STORE_BACKEND = "local"
$env:AGENT_LEARNING_LOCAL_STORE_DIR = Join-Path $env:LOCALAPPDATA "agent-learning\store"
```

Define actions the agent can really execute in `actions.json`:

```json
[
   {
      "id": "use_text_search",
      "description": "Search for exact symbols and terms",
      "parameters": {"strategy": "text"}
   },
   {
      "id": "use_semantic_search",
      "description": "Search the repository by meaning",
      "parameters": {"strategy": "semantic"}
   }
]
```

Initialize the decision once, then ask the active policy what to execute:

```powershell
agent-learn task-policy-init `
   --agent-id code-reviewer `
   --task-id choose-repository-search `
   --decision-context "Choose a repository search strategy for a coding task" `
   --actions .\actions.json

agent-learn task-policy-decide `
   --agent-id code-reviewer `
   --task-id choose-repository-search
```

Execute the returned `selected_action`, preserve its policy fields, and record
the independently observed outcome in `episode.json`. Then close the loop:

```powershell
agent-learn task-episode-register `
   --agent-id code-reviewer `
   --task-id choose-repository-search `
   --episode .\episode.json `
   --require-decision-policy

agent-learn score --agent-id code-reviewer --task-id choose-repository-search
agent-learn train --agent-id code-reviewer --task-id choose-repository-search --decision-only
agent-learn task-policy-decide --agent-id code-reviewer --task-id choose-repository-search
```

The final decision consumes the updated probabilities and feedback from prior
executions. Training defaults to a minimum of five completed episodes. See the
[decision-making guide](docs/decision-making.md) for the episode schema,
evidence rules, math, and deployment patterns.

## Functional Testing

A good way to see the SDK in action is to run the
interactive capture scenario first, followed by the offline batch update:

```powershell
python tests/functional_cli_interactive.py
python tests/functional_cli_batch.py
```

## Usage

The `agent-learn` CLI provides the decision lifecycle and inspection
operations:

```text
agent-learn list
agent-learn --version
agent-learn tasks-list <agent_id> [--decision-only]
agent-learn task-episodes-count <agent_id> [--task-id <task_id>] [--include-incomplete] [--start-date <date>] [--end-date <date>]
agent-learn task-episodes-list <agent_id> [--task-id <task_id>] [--limit <1-500>] [--include-incomplete] [--start-date <date>] [--end-date <date>]
agent-learn task-policy-init --agent-id <agent_id> --task-id <task_id> --decision-context <context> --actions ./actions.json
agent-learn task-policy-decide --agent-id <agent_id> --task-id <task_id> [--history-limit <1-500>] [--greedy] [--seed <integer>]
agent-learn task-episode-register --agent-id <agent_id> --task-id <task_id> --episode ./episode.json [--require-decision-policy]
agent-learn score --agent-id <agent_id> [--task-id <task_id>] [--limit <1-500>]
agent-learn train --agent-id <agent_id> [--task-id <task_id>] [--decision-only] [--limit <1-500>] [--min-episodes <1-500>] [--start-date <date>] [--end-date <date>] [--skip-scoring]
agent-learn task-policy --agent-id <agent_id> --task-id <task_id>
```

## Documentation

- [Agentic decision making](docs/decision-making.md): concepts, evidence,
   workflow, math, and deployment.
- [Scout decision integration](docs/scout-agent-learn-skill.md): apply learned
   delegated decisions during execution.
- [Scout decision training](docs/scout-automation-agent-learning.md): train
   eligible policies without turning automation into a policy task.
- [The math, explained simply](docs/math-explained-simply.md) and
   [mathematical reference](docs/math.md): softmax, rewards, baselines, and
   REINFORCE.
- [Tiered scoring design](docs/design.md): local and Azure-backed outcome
   scoring.
