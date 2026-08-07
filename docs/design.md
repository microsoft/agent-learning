---
title: Tiered judge design for the agent-learning SDK
description: Four-tier judge architecture (Python stdlib, NLP library, small language model, large language model) for intent resolution, task adherence, and task completion, with managed-identity support for enterprise deployments
author: Microsoft
ms.date: 2026-05-17
ms.topic: concept
keywords:
  - reinforcement learning
  - reward shaping
  - intent classification
  - task adherence
  - task completion
  - non-LLM
  - small language model
  - managed identity
  - sdk
estimated_reading_time: 18
---

## Overview

The agent-learning SDK is a reinforcement-learning toolkit for agents. Today the reward shaper consumes three metric scores (intent resolution, task adherence, task completion) that the caller must produce.

This document proposes a first-class judge layer inside the SDK, organized as four capability tiers. Each tier is a self-contained scoring stack with a different cost, latency, and dependency profile. Callers pick the tier that matches their environment; the public `JudgeScore` shape, the reward shaper, the learner, and the storage layers stay unchanged across tiers.

The two operational modes (`nlp` and `llm`) sit on top of the tier model. The `nlp` mode draws from Tiers 1 through 3 depending on which extras are installed. The `llm` mode is Tier 4. The default mode is `llm` for backwards compatibility.

## Goals and non-goals

### Goals

* Ship a default judge implementation that runs without an external service when the caller chooses one of the local tiers.
* Preserve the existing LLM judge path as a first-class tier.
* Keep the public API stable for callers already using the SDK.
* Provide a reproducible, deterministic scorer that the offline classifier-tuner CronJob can fit.
* Support enterprise managed-identity authentication for the LLM tier, so customers can stand up the SDK without provisioning a static API key.

### Non-goals

* We are not removing the LLM backend. Callers who prefer the LLM judge keep it.
* We are not bundling a large language model in the wheel.
* We are not redesigning the reward shaper. Only the scorer that feeds it changes.
* We are not training the three judges jointly. Each judge is independent and snapshot-versioned on its own.

## The mode switch

The SDK gains a `judge_mode` axis that ripples through the configuration, factory, and snapshot metadata.

The runtime configuration carries two top-level pieces:

* `judge_mode`, one of `"nlp"` or `"llm"`. The environment variable `AGENT_LEARNING_JUDGE_MODE` overrides the in-code default. When unset, the mode is `"llm"` to preserve existing behavior.
* A backend-specific block: `NlpJudgeConfig` for the local stack, `JudgeConfig` for the LLM stack. Both blocks always exist; only the active one is consulted.

A single factory entry point hides the backend choice. The factory inspects the mode, the installed extras, and the configuration, then returns the three judges (intent, adherence, completion) bound to the appropriate tier. Callers never instantiate a judge directly.

Snapshot metadata carries the resolved tier and mode alongside each scored episode, so historical episodes know which scorer produced them. Re-shaping a past episode under a new tier becomes a deterministic, reproducible operation.

## The four-tier capability stack

> [!NOTE]
> Sizes are wheel-on-disk plus transitive runtime dependencies, rounded. Latencies are p99 per judgement on a 4-vCPU CPU-only worker with the response held under 4 KB of text.

### Tier 1, Python standard library

* Install: `pip install agents-learning-sdk`.
* Footprint on disk: 0 MB additional.
* Latency: sub-millisecond.
* License of new code: MIT (the SDK itself).

Tier 1 is a pure-standard-library scorer. Every dependency it needs already exists in `agent_learning.classifiers.base` (lightweight tokenizer, bag-of-words feature hash, pure-Python multinomial logistic regression). It is the floor everyone gets, and it is what the SDK falls back to when no optional extras are installed.

Tier 1 handles the easy bands of all three judges: closed-set intent classification with small label inventories, hard-rule contract checks (length, required substrings, forbidden substrings, JSON-shape via stdlib `json`), and required-token coverage measured by exact and case-insensitive matching.

### Tier 2, NLP library

* Install: `pip install agents-learning-sdk[nlp]`.
* Footprint on disk: about 75 MB total.
* Latency: 1 to 5 milliseconds per judgement.
* License posture: every package below is permissive (BSD, MIT, or Apache 2.0).

Tier 2 adds proven, narrow NLP libraries that fit comfortably in a CPU container:

* `scikit-learn`, BSD-3-Clause, about 13 MB. TF-IDF vectorizers, calibrated logistic regression, isotonic calibration.
* `numpy`, BSD-3-Clause, about 18 MB. Numerical backbone for scikit-learn.
* `scipy`, BSD-3-Clause, about 40 MB. Sparse linear algebra used by scikit-learn's calibrators.
* `joblib`, BSD-3-Clause, about 0.3 MB. Snapshot serialization for fitted vectorizers and classifiers.
* `threadpoolctl`, BSD-3-Clause, about 0.05 MB. Required by scikit-learn.
* `rapidfuzz`, MIT, about 1.7 MB. Fast fuzzy matching for required-token coverage.
* `jsonschema`, MIT, about 0.1 MB. Strict JSON-contract validation for adherence.
* `regex`, Apache-2.0, about 0.8 MB. Unicode-aware regex with full Posix character classes.
* `textstat`, MIT, about 0.1 MB. Readability metrics for contracts that mandate a grade level.
* `rouge-score`, Apache-2.0 (Google), about 0.05 MB. ROUGE-L F1 for completion scoring against reference responses.

Tier 2 is the recommended default for production deployments that cannot tolerate network calls in the reward path. It matches fine-tuned BERT on small in-domain corpora for intent, gives strong rule-plus-soft-head adherence checks, and supplies overlap-based completion scoring.

### Tier 3, small language model

* Install: `pip install agents-learning-sdk[nlp-semantic]` (pulls Tier 2 transitively).
* Footprint on disk: about 380 MB total, dominated by PyTorch.
* Latency: 5 to 30 milliseconds per judgement on CPU.
* License posture: every package below is permissive (Apache-2.0 or BSD-3).

Tier 3 adds a single small language model for semantic similarity:

* `sentence-transformers`, Apache-2.0, about 2 MB. The sentence-embedding wrapper.
* `transformers`, Apache-2.0 (Hugging Face), about 8 MB.
* `tokenizers`, Apache-2.0 (Hugging Face), about 4 MB.
* `torch` (CPU build), BSD-3-Clause (Meta), about 200 MB. The neural-network runtime.
* Model weights, `sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0, 22 million parameters, about 90 MB on disk.

Tier 3 makes the judges robust to paraphrase. Intent gains per-class centroids and cosine similarity; adherence gains semantic detection of "did the response say what the contract required even if it used different words"; completion gains semantic coverage of required artifacts.

> [!NOTE]
> We considered `paraphrase-MiniLM-L3-v2` (smaller, faster) and `bge-small-en-v1.5` (slightly better quality, similar size). MiniLM-L6 is the right balance for CPU containers. Rasa DIET was rejected because it pulls TensorFlow and is too heavy for an SDK. fastText was rejected because the upstream is unmaintained and the on-disk model size advantage disappears once Tier 2 is already installed.

### Tier 4, large language model

* Install: `pip install agents-learning-sdk[llm]`.
* Footprint on disk: about 10 MB of Python code plus network egress to the model endpoint.
* Latency: 200 to 1000 milliseconds per judgement, network-bound.
* License posture: permissive Python clients; the model itself sits behind Azure OpenAI or Azure AI Foundry and is governed by the customer's deployment contract.

Tier 4 wraps the existing Azure AI evaluation surface:

* `azure-ai-evaluation`, MIT, about 1 MB. Hosts `IntentResolutionEvaluator`, `TaskAdherenceEvaluator`, and `TaskCompletionEvaluator`.
* `azure-identity`, MIT, about 0.5 MB. Provides credential resolution (`DefaultAzureCredential`, `ManagedIdentityCredential`, `WorkloadIdentityCredential`, and friends).
* Transitive Azure Core libraries, MIT.

Tier 4 is the highest-fidelity tier and the one that the offline classifier-tuner uses to mint labels for fitting Tiers 2 and 3.

### Side-by-side

| Tier | Mode | Install | Size | Latency p99 | License posture |
|---|---|---|---|---|---|
| 1 | nlp | base wheel | 0 MB | under 1 ms | MIT (SDK only) |
| 2 | nlp | `[nlp]` extra | ~75 MB | 1 to 5 ms | BSD / MIT / Apache-2.0 |
| 3 | nlp | `[nlp-semantic]` extra | ~380 MB | 5 to 30 ms | Apache-2.0 / BSD |
| 4 | llm | `[llm]` extra | ~10 MB plus network | 200 to 1000 ms | MIT clients; model governed by customer |

## Per-judge tier mapping

Each judge has a recipe at every tier. The factory picks the highest tier whose extras are installed.

| Judge | Tier 1 (stdlib) | Tier 2 (NLP library) | Tier 3 (small LM) | Tier 4 (LLM) |
|---|---|---|---|---|
| Intent | Bag-of-words logistic regression over a closed label set. | TF-IDF plus calibrated logistic regression. Strong on small in-domain corpora. | Tier 2 plus cosine similarity to per-class MiniLM centroids. | `IntentResolutionEvaluator` over chat-history. |
| Adherence | Exact and case-insensitive substring rules, length bounds, stdlib JSON parse. | Adds `jsonschema`, `regex`, `rapidfuzz` near-match rules, and a calibrated soft head on TF-IDF features. | Adds semantic-equivalence checks for "the contract required X, did the response say something equivalent to X". | `TaskAdherenceEvaluator` Likert scored against the contract. |
| Completion | Required-token coverage by exact and case-insensitive match; length ratio. | Adds `rapidfuzz` fuzzy coverage, ROUGE-L F1 against a reference response (when present), and structural-element coverage. | Adds MiniLM cosine similarity between response and reference, and semantic coverage of required artifacts. | `TaskCompletionEvaluator` Likert against the artifact set. |

## Authentication and identity

Enterprise customers cannot ship static API keys with their workloads. The SDK supports three credential paths for Tier 4:

* **Explicit credential.** The caller passes any `azure.core.credentials.TokenCredential` (or `AsyncTokenCredential`) on the runtime config. The SDK uses it verbatim. This is the path for callers that already manage credentials in their own bootstrap.
* **Credential mode hint.** The caller sets a string hint and the SDK builds the credential itself. The accepted values are `"default"` (chained `DefaultAzureCredential`), `"managed-identity"` (system-assigned `ManagedIdentityCredential` or user-assigned when a client id is given), `"workload-identity"` (`WorkloadIdentityCredential`, the AKS-native path), `"environment"` (env-var driven `EnvironmentCredential`), `"azure-cli"` (developer laptops), and `"none"` (no credential; falls back to API key if present).
* **User-assigned managed identity.** When the workload runs on AKS with a user-assigned identity, the caller sets the client id alongside the mode hint. The SDK passes it through to the credential constructor.

Two environment variables mirror these fields so the same SDK build can switch between developer-laptop, CI, and production AKS without code changes: `AGENT_LEARNING_JUDGE_CREDENTIAL_MODE` and `AGENT_LEARNING_JUDGE_USER_ASSIGNED_CLIENT_ID`.

When both a credential and an API key are present, the credential wins. The API key path remains for testing and for environments where managed identity is not available.

## Snapshot lineage

Every reward record persisted to Cosmos already carries an immutable lineage block. The judge layer adds:

* `judge_mode`, the active mode at scoring time.
* `judge_tier`, the resolved tier (1, 2, 3, or 4).
* The model identifier (for Tier 4) or the snapshot hash of the fitted classifier (for Tiers 1 through 3).

This lets the validator answer "show me every episode that was scored by Tier 2 intent" and lets the rollback path re-shape episodes under a new tier without losing the original signal.

## Validation strategy

A scorer is acceptable when it matches Tier 4 closely enough on a held-out set of episodes:

* Agreement: Cohen's kappa of at least 0.85 for intent, 0.80 for adherence, and 0.75 for completion against Tier 4 labels on a 10,000-episode validation set.
* Calibration: per-judge F1 within 5 percentage points of Tier 4 across the validation set, with the operating threshold tuned per tier.
* Determinism: identical inputs produce identical outputs across runs (Tiers 1 through 3 are fully deterministic; Tier 4 is sampled at temperature zero with a fixed seed where the evaluator exposes one).
* Latency: p99 under 10 milliseconds for Tier 1 and 2, under 50 milliseconds for Tier 3, on a 4-vCPU CPU-only worker.

Cohen's kappa is the right statistic here because the judge label is binary and the base rate is heavily skewed toward "pass" (around 94 percent in the current corpus). Raw accuracy and even macro F1 are misleading at that skew, because a scorer that just says "pass" looks excellent on accuracy and still respectable on F1. Cohen's kappa subtracts the agreement that two raters would reach by random chance at the observed marginal distribution, so it isolates the agreement that comes from the scorer actually understanding the response. A kappa of 1 is perfect agreement, 0 is chance-level, and negative values are worse than chance. Reporting kappa next to F1 prevents the headline "F1 = 0.97" from masking a scorer that is mostly riding the base rate.

## Training data and the offline tuner

Tiers 2 and 3 are trained, not handwritten. The corpus is the same set of captured episodes the LLM judge already scored: chat history, action contract, response, and the LLM judge's pass-fail label per judge. The CronJob fits one logistic-regression head per judge per tier, calibrates it on a held-out fold, snapshots the fitted vectorizers and classifiers to blob storage, and registers the snapshot hash with the SDK. Tier 1 needs no fitting; Tier 4 needs no training because the model is hosted.

> [!NOTE]
> The synthetic corpus used by `classifier_f1.py` today carries the action embedding and the metric labels but does not yet store the response text. The offline tuner will fit on real captured episodes that do carry response text. Until that pipeline is wired, the F1 numbers in the dq repo's `AGENTS_LEARNING_DESIGN.md` §11.10 use the embedding-only shortcut and are bounded by what the embedding alone can predict.

## Phasing

| Phase | Tier delivered | What ships |
|---|---|---|
| P1 | Tier 1 (stdlib) | `judge_mode` plumbing, `JudgeRuntimeConfig`, `build_judges` factory, stdlib intent and adherence and completion judges, full test coverage. Default mode stays `"llm"`. |
| P2 | Tier 2 (NLP library) | `[nlp]` extra, TF-IDF and calibrated soft heads for all three judges, snapshot serialization, offline tuner integration. |
| P3 | Tier 3 (small LM) | `[nlp-semantic]` extra, MiniLM centroid intent boost, semantic adherence equivalence, semantic completion coverage. |
| P4 | Tier 4 (LLM) plus managed identity | Tier 4 already exists; this phase adds `credential`, `credential_mode`, and `user_assigned_client_id` to `JudgeConfig`, wires `DefaultAzureCredential` and `ManagedIdentityCredential` and `WorkloadIdentityCredential`, and documents the AKS workload-identity recipe. |

## Decisions

* **Default mode is `"llm"`.** Existing callers must not see a behavior change when they upgrade. They pick local tiers only by opting in.
* **Tier 1 stays pure standard library.** Callers who refuse the `[nlp]` extra still get a working scorer, just a weaker one. No transitive dependency is allowed to creep into the base wheel.
* **Judges are independent.** Each of intent, adherence, and completion is its own snapshot. We do not train them jointly; that locks unrelated bugs together and complicates rollback.

## References

* Snips NLU intent engine algorithm: <https://github.com/snipsco/snips-nlu>
* Rasa intent classifiers, including the logistic-regression classifier: <https://rasa.com/docs/rasa/components/>
* `sentence-transformers` model card for `all-MiniLM-L6-v2`: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
* ROUGE scoring: <https://github.com/google-research/google-research/tree/master/rouge>
* `azure-ai-evaluation` reference: <https://learn.microsoft.com/python/api/azure-ai-evaluation/>
* `azure-identity` credential reference: <https://learn.microsoft.com/python/api/azure-identity/>
* AKS workload identity: <https://learn.microsoft.com/azure/aks/workload-identity-overview>
* Cohen's kappa, Jacob Cohen, 1960, "A Coefficient of Agreement for Nominal Scales": <https://doi.org/10.1177/001316446002000104>
