# Cost model and break-even analysis

Avoiding a foundation-model training job changes the cost structure; it does
not establish that policy learning is always cheaper. Evaluating every episode
with hosted judges can outweigh savings in training compute. Local scoring can
reduce API spending but still consumes compute and may measure a different
notion of success.

This guide and the [standalone calculator](../scripts/estimate_learning_cost.py)
address [issue #4](https://github.com/microsoft/agent-learning/issues/4) with an
explicit, configurable cost model. The calculator makes no network calls and
needs only Python 3.10 or later. It does not import the SDK, train models, fetch
prices, or require Azure credentials.

## Run the examples

From the repository root:

```shell
python scripts/estimate_learning_cost.py --config examples/cost_model_local.json
python scripts/estimate_learning_cost.py --config examples/cost_model_hosted.json --episodes 0 10000 20000 100000
```

The first command uses the configured episode count. `--episodes` overrides it
with one or more non-negative integer volumes. Results are JSON on stdout;
invalid input produces an error on stderr and exit code 2. Redirect stdout to a
file to retain a report. Decimal amounts are JSON **strings**, preserving
decimal arithmetic without rounding each episode to cents. Episode counts and
period length are JSON integers. The guide rounds displayed costs to cents only
after calculating totals; repeating decimal crossover volumes are approximate.

For direct Python calls, `Scenario.compare(None)` uses the configured volume.
An explicit empty list raises `ValueError`; `[0]` is a valid zero-volume sweep.

The [local scenario](../examples/cost_model_local.json) and
[hosted scenario](../examples/cost_model_hosted.json) use **invented prices and
workload assumptions**, not current provider pricing or measured performance.
Do not use their totals as a quote, a benchmark, or evidence of equal quality.

## Define the comparison first

Use the same workload volume, currency, and observation period for both
approaches. Count all costs needed to deliver and evaluate that workload, not
just a CPU update versus a GPU training job. Common inference or platform costs
belong in `shared` and appear once in each alternative's total; they cancel in
the difference. They are not two expenses incurred together.

Before making a real comparison, establish comparable task quality, safety,
latency, and evaluation coverage. This tool cannot establish those conditions.
If model quality or workload coverage differs, report the cost scenarios
separately rather than interpreting the cheaper result as a recommendation.

## Configuration

All top-level fields are required. Unknown fields, duplicate JSON keys,
negative values, non-finite numbers, and booleans in numeric fields are rejected.
Use JSON numbers rather than quoted numeric strings for input.

| Top-level field | Meaning |
|---|---|
| `currency` | Non-empty label shared by all costs; no currency conversion is performed. |
| `period_days` | Positive integer length of the comparison period. |
| `episodes` | Non-negative integer workload volume in that period. |
| `assumptions` | Non-empty list documenting sources, exclusions, scope, and quality caveats. |
| `shared` | Explicit `fixed` cost per period and `per_episode` cost, including zeros. |
| `policy_learning`, `fine_tuning` | Separate objects using the identical fields below. |

Approach fields default to zero except `evaluation_fraction`, which defaults
to 1. An omitted cost is **excluded**, not proven free. Empty approach objects
are permitted for sensitivity tests, not evidence of zero real-world cost.

| Approach field | Unit and interpretation |
|---|---|
| `evaluation_fraction` | Fraction of episodes evaluated, from 0 through 1. |
| `judge_calls_per_evaluated_episode` | Mean billable calls for one evaluated episode, including retries/rescoring if applicable. |
| `judge_cost_per_call` | Mean currency cost per call, including input and output tokens. Use zero API cost for local scoring. |
| `cpu_seconds_per_episode` | Mean metered CPU/instance time per workload episode, including local scoring if applicable. |
| `cpu_cost_per_hour` | Cost per hour for that same measured resource; seconds divided by 3,600. |
| `writes_per_episode` | Mean billable storage writes per episode. Include episodes, metric/reward records, and other attributable writes. |
| `cost_per_million_writes` | Effective cost per million writes, not a provider-specific storage tariff. |
| `average_retained_gb` | Average existing storage footprint during the period, independent of new volume. |
| `average_retained_gb_per_episode` | Additional average footprint attributable to each episode over the period, accounting for retention. |
| `storage_cost_per_gb_month` | Storage capacity rate; this model defines a month as 30 days and uses decimal GB. |
| `cycles_per_period` | Integer number of scheduled update/training and rollout cycles. |
| `compute_cost_per_cycle` | Additional compute budget per cycle; exclude CPU time already counted per episode. |
| `rollout_cost_per_cycle` | Additional validation, canary, or promotion cost per cycle; do not duplicate evaluation/serving costs. |
| `fixed_cost_per_period` | Other approach-specific costs, such as reserved compute, operations, or amortized setup labor. |
| `serving_cost_per_episode` | Approach-specific serving/inference cost beyond any common cost in `shared`. |
| `other_cost_per_episode` | Other variable costs, such as human review, reads, or data transfer. |

To price token-based judges, calculate cost per call from representative input
and output token counts and dated rates, then multiply by measured calls. Three
metric names do not necessarily mean three billable calls in every integration.
For local SLM/GPU scoring, use an appropriate instance-hour rate and measured
runtime, or include a reserved deployment in fixed costs; do not count both for
the same resource.

For RU-based storage, estimate RU per write times price per million RU as an
effective write rate; model reads, provisioned capacity, minimum charges, and
replicas separately. Do not confuse operations with bytes or assume the sample
write count matches every SDK workflow. For uniformly arriving 40 KB records
retained until the end of the period, average new storage is approximately
20 KB per episode, or `0.00002` decimal GB. Longer history belongs in the
existing footprint. Compression and expiration require different inputs.

Record price source/date/region, model and token budget, retention, update
cadence, and any omissions in `assumptions`. Network, taxes, retries, monitoring,
human review, and initial integration costs are not automatically discovered.
Amortize setup costs over a stated horizon or include them in the first period.
This is only a total-cost-of-ownership estimate when the supplied inputs cover
the relevant ownership costs.

## Model and break-even formula

For each approach, period cost is affine in episode volume $N$:

$$
C(N) = F + vN + C_{\mathrm{shared}}(N).
$$

Fixed cost $F$ includes scheduled cycle compute and rollouts, existing average
storage capacity, and other fixed costs. Unit cost $v$ includes judge calls,
per-episode compute, writes, new average storage, serving, and other variable
costs. In particular:

$$
v_{\mathrm{judge}} = f_{\mathrm{evaluated}}\,j_{\mathrm{calls}}\,p_{\mathrm{call}},
\qquad
C_{\mathrm{cycles}} = k\,(c_{\mathrm{compute}} + c_{\mathrm{rollout}}).
$$

Storage cost is the average footprint times the GB-month rate times
`period_days / 30`. Cycle counts and fixed-cost inputs already cover the chosen
period: changing `period_days` prorates storage only. Update all other budgets
and cadence yourself when changing the period.

Let $P$ denote policy learning and $T$ fine-tuning. Shared costs cancel. If
$v_P \ne v_T$, the continuous-volume equality point is:

$$
N^* = \frac{F_T - F_P}{v_P - v_T}.
$$

The output distinguishes a non-negative crossover, one approach being cheaper
at all non-negative volumes, and equal cost at all volumes. A crossover at zero
means equal fixed costs; there is no non-negative region below it. A fractional
crossover may have no integer volume where costs are exactly equal.
`first_integer_volume_above` identifies the first whole episode count strictly
above the equality point; `cheaper_above` tells you which approach wins there.
`fine_tuning_minus_policy_learning` is positive when policy learning costs less
and negative when it costs more. It is a cost difference, not ROI.

## Worked comparison

Both sample configurations cover 30 days and share 100 in fixed USD costs plus
0.002 per episode. Both allocate 0.02 CPU seconds per episode at 0.36/hour,
eight writes at 1/million writes, and capacity at 0.25/GB-month with 2 GB of
existing data and 0.00002 average GB per new episode.

Policy learning budgets four cycles, each with 0.50 compute and 2 rollout
cost, plus 10 other fixed costs. Fine-tuning budgets one cycle with 200 compute
and 20 rollout cost, plus 10 other fixed costs. The effective fixed costs before
shared costs are therefore 20.50 and 230.50 respectively.

In both examples, fine-tuning evaluates a 10% subset with three hosted calls at
0.004 each, yielding 0.0012 judge cost per workload episode. Policy learning
evaluates every episode: locally in the first scenario, or with three hosted
calls at the same price in the second. These are deliberately different
evaluation workloads. **No equal-quality or equal-coverage claim follows.**

| Episodes | Policy: local scoring (USD) | Policy: hosted judges (USD) | Fine-tuning baseline (USD) |
|---|---:|---:|---:|
| 0 | 120.50 | 120.50 | 330.50 |
| 10,000 | 140.65 | 260.65 | 362.65 |
| 20,000 | 160.80 | 400.80 | 394.80 |
| 100,000 | 322.00 | 1,522.00 | 652.00 |

Local scoring has a unit cost of 0.000015 before shared costs, versus 0.001215
for fine-tuning. Policy learning has both lower fixed and lower variable costs,
so there is **no non-negative crossover** under these assumptions.

Hosted judging raises policy learning's unit cost to 0.012015. Its fixed-cost
advantage is 210, but each additional episode costs 0.0108 more than the
fine-tuning scenario. Equality is at approximately **19,444.44 episodes per
30-day period**. At 19,445 whole episodes and above, fine-tuning is cheaper in
this model. At 10,000 episodes, policy learning is 102 cheaper; at 100,000 it
is 870 more expensive. More volume does not always favor policy learning.

For a like-for-like hosted evaluation sensitivity check, set the fine-tuning
evaluation fraction to 1 as well. Marginal costs then match and the 210 fixed
cost difference remains at every volume. Conversely, a lower fine-tuned
serving cost, higher local scoring compute, or different rollout cadence can
change the result again. Match those assumptions to a measured workload.

For a separate serving-cost sensitivity check, start with the original hosted
configuration and set only `fine_tuning.serving_cost_per_episode` to `0.0024`.
This is an illustrative premium above the serving costs already in `shared`,
not a provider price. Policy learning's variable-cost disadvantage falls from
0.0108 to 0.0084 per episode, moving equality from approximately 19,444.44 to
**25,000 episodes**. Both approaches cost 470.875 USD at that volume; fine-tuning
becomes cheaper at 25,001 episodes. This changes only the cost assumption and
does not establish equivalent quality or evaluation coverage.

## Limits and validation

The sweep holds unit rates, retained footprint per episode, cycle counts, and
per-cycle budgets fixed. At zero volume it still charges scheduled cycles and
existing storage: set cycle counts to zero to model no training. Real fine-tune
jobs may grow with dataset size, and policy batches may grow with volume. Put
linear growth in per-episode compute, or run separate configurations when a
threshold changes the cadence or budget. This model does not solve tiered
pricing, minimum charges, autoscaling steps, or multiple crossover points.

The tool calculates expected counts, so sampled evaluation and mean writes
can be fractional. It does not simulate randomness, retry distributions,
quality gains, lost revenue, or statistical confidence in the input estimates.
Do not infer equivalent agent capabilities from equivalent costs.

Run the offline regression tests from a development environment:

```shell
python -m pytest tests/test_cost_model.py -q
```

Tests cover the worked examples, cost breakdowns, both crossover directions,
shared costs, zero and fractional volumes, equal-cost/no-crossover cases,
invalid inputs, and operation without installed SDK dependencies.