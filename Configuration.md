## Configuration

The SDK reads its configuration from environment variables. Every
variable is optional — with no configuration the SDK runs against an
in-memory store. The most important ones are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `AGENT_LEARNING_STORE_BACKEND` | Storage backend: `memory`, `cosmos`, or `local` | `memory` |
| `AGENT_LEARNING_COSMOS_ENDPOINT` | Cosmos DB account URL (only used when backend is `cosmos`) | unset |
| `AGENT_LEARNING_COSMOS_DATABASE` | Cosmos DB database name (only used when backend is `cosmos`) | `dq_rl` |
| `AGENT_LEARNING_LOCAL_STORE_DIR` | Directory for the `local` file backend | `./data/agent-learning/store` |
| `AGENT_LEARNING_SCORE_MODE` | Compatibility scorer mode: `nlp` or `llm` | `llm` |
| `AGENT_LEARNING_SCORE_TIER` | Preferred scorer tier: `stdlib`, `nlp`, `slm`, or `llm` | `stdlib` for CLI/runner scoring |
| `AGENT_LEARNING_SCORE_ENDPOINT` | Azure OpenAI endpoint used by the scorer | unset |
| `AGENT_LEARNING_SCORE_DEPLOYMENT` | Scorer deployment name | unset |
| `AGENT_LEARNING_SCORE_API_KEY` | API key for the LLM scorer | unset |
| `AGENT_LEARNING_SCORE_API_VERSION` | Azure OpenAI API version used by the LLM scorer | `2024-10-21` |
| `AGENT_LEARNING_SCORE_CREDENTIAL_MODE` | Azure credential resolution mode for the LLM scorer | unset |
| `AGENT_LEARNING_SCORE_USER_ASSIGNED_CLIENT_ID` | User-assigned managed identity client ID | unset |
| `AGENT_LEARNING_SCORE_CREDENTIAL_SCOPE` | Azure token scope for the LLM scorer | `https://cognitiveservices.azure.com/.default` |
| `AGENT_LEARNING_NLP_SCORE_DIR` | Snapshot directory for feature-based NLP scorers | `./data/agent-learning/nlp-scores` |
| `AGENT_LEARNING_STDLIB_SCORE_DIR` | Snapshot directory for Tier 1 stdlib scorers | `./data/agent-learning/stdlib-scores` |
| `AGENT_LEARNING_NLP_TEXT_SCORE_DIR` | Snapshot directory for Tier 2 NLP text scorers | `./data/agent-learning/nlp-text-scores` |
| `AGENT_LEARNING_W_INTENT` | Weight for intent-resolution reward | `0.4` |
| `AGENT_LEARNING_W_ADHERENCE` | Weight for task-adherence reward | `0.3` |
| `AGENT_LEARNING_W_COMPLETION` | Weight for task-completion reward | `0.3` |
| `AGENT_LEARNING_LR` | REINFORCE learning rate | `0.05` |
| `AGENT_LEARNING_BASELINE_DECAY` | EMA decay on the value baseline | `0.9` |

By default the SDK uses a volatile in-memory store and scores episodes locally
with the dependency-free stdlib tier. Set
`AGENT_LEARNING_STORE_BACKEND=cosmos` (together with the Cosmos
variables above) for durable Cosmos DB persistence, or `=local` to
persist to JSON files on disk. To opt into Azure AI evaluation, configure
`AGENT_LEARNING_SCORE_ENDPOINT` and `AGENT_LEARNING_SCORE_DEPLOYMENT`, or set
the score tier to `llm` and pass an explicit `ScoreConfig`.