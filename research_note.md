# Cost-Aware Create-or-Reuse Decisions for LLM Response Systems

*A short research note accompanying the GenAction prototype. It explains the
problem, the policies, and the intuition behind the Doubly Optimistic rule.
It is **inspired by** the create-or-reuse view of online decision making with
generative action sets; it does not reproduce a specific theorem and proves no
regret bound.*

---

## 1. Motivation

LLM-powered response systems — customer-support assistants, sales copilots,
agentic tools — do not answer every request from scratch. They keep a library of
reusable assets (canned answers, FAQ macros, retrieval snippets, cached
generations, learned tools) and continually face a micro-decision: **is an
existing asset good enough for this request, or is it worth paying to generate a
new, reusable one?**

Two properties make this more than a similarity threshold. First, the costs are
heterogeneous: creating consumes tokens, latency, and human curation, whereas a
poorly-fitting reuse silently degrades answer quality. Second, the action set is
**endogenous and generative**: a good creation now becomes an asset that serves
many similar future requests. The optimal behaviour therefore amortises creation
cost over future reuse, and must be learned online as the stream unfolds.

## 2. Online decision formulation

At round $t = 1, \dots, T$ the agent observes a context $x_t$ and holds an action
set $\mathcal{A}_t$. It chooses one of:

- **reuse** an existing action $a_i \in \mathcal{A}_t$ → loss $\ell(x_t, a_i)$,
  $\mathcal{A}_{t+1} = \mathcal{A}_t$;
- **create** a new action $a_\text{new}$ → cost $c$,
  $\mathcal{A}_{t+1} = \mathcal{A}_t \cup \{a_\text{new}\}$.

$$
\text{cost}_t = \begin{cases} \ell(x_t, a_i) & \text{(reuse } a_i)\\ c & \text{(create)} \end{cases}
\qquad\qquad \min \; \sum_{t=1}^{T}\text{cost}_t .
$$

The mismatch loss used here is

$$
\ell(x, a) = \mathrm{clip}\big(\, d(x,a)^{\gamma} + \lambda\,\mathbb{1}[\mathrm{cat}(x) \neq \mathrm{cat}(a)] + \varepsilon,\; 0,\, 1\,\big),
$$

with $d(x,a) = \mathrm{clip}(1 - \cos(\phi(x), \phi(a)),0,1)$ the embedding
distance to the action's canonical query, $\gamma$ the convexity of the
distance–loss curve, $\lambda$ a wrong-intent penalty, and $\varepsilon$ small
zero-mean noise. The agent can compute $d(x,a)$ at decision time but learns the
$\lambda$ and $\varepsilon$ terms only from realised feedback after a reuse —
i.e. it faces **bandit-style partial feedback** over a **growing** arm set.

## 3. The create-or-reuse trade-off

The two trivial policies are the two extremes of a single axis.

- **Always create** drives mismatch loss to zero but pays $c$ every round —
  total cost $Tc$. Wasteful whenever good reuse exists.
- **Always reuse (nearest)** never pays $c$ but is capped by how well the current
  library covers the stream — total cost $\sum_t \ell(x_t, a_{i^\star(t)})$.

The interesting region is in between, and it is *state-dependent*: a query that
is well covered by the library should be reused; a genuinely novel query should
trigger a creation **because the new action will be reused by the similar queries
that follow**. Formally, creating is worthwhile for a cluster of $m$ upcoming
similar queries when the one-off cost $c$ undercuts the avoided mismatch,
$c \lesssim \sum_{k=1}^{m} \ell(x_k, a_\text{best})$ — a small online
amortisation argument. A good policy must therefore estimate both *how poor*
reuse currently is and *how uncertain* that estimate is, relative to $c$.

## 4. Policies compared

| Policy | Decision rule | What it tests |
| --- | --- | --- |
| `AlwaysCreate` | create every round | cost ceiling / quality floor |
| `NearestReuse` | reuse $\arg\min_i d(x,a_i)$; never create | value of any creation |
| `FixedProbability(p)` | create w.p. $p$ else reuse nearest | value of *targeted* creation vs. random creation at the same rate |
| `StaticThreshold` | create iff $d(x,a_\text{nearest})^{\gamma} > c$ | cost-aware ablation: is the LCB/UCB machinery needed at all? (§7.1) |
| **`DoublyOptimistic`** | LCB selection + UCB creation test | the adaptive method |

`FixedProbability` is the most informative *cost-unaware* baseline: it spends a
comparable creation budget but allocates it *blindly*, so any gap to the adaptive
policy isolates the value of deciding **where** to create, not just how often.
`StaticThreshold` is the *cost-aware* ablation that removes only the confidence
bounds and the learning, isolating what those two ingredients add (§7.1).

## 5. Doubly Optimistic policy — intuition

For each action $a_i$ and the current query $x$, maintain a kernel-weighted,
prior-anchored estimate of the reuse loss:

$$
\mu_i = \frac{k_0\,\pi_i + \sum_j w_{ij}\,\ell_{ij}}{k_0 + \sum_j w_{ij}},
\qquad
\sigma_i = \frac{\sigma_0}{\sqrt{k_0 + \sum_j w_{ij}}},
$$

- $\pi_i = d(x,a_i)^{\text{prior\_power}}$ — the belief *before any feedback*,
  from the always-available distance;
- $w_{ij} = \exp\!\big(-(d(x, x_{ij})/h)^2\big)$ — weights past reuses of $a_i$
  by similarity of their query $x_{ij}$ to $x$ (bandwidth $h$), so *relevant*
  experience counts most;
- $k_0 = \texttt{min\_observations}$ — a prior pseudo-count.

Then form two confidence bounds and act:

$$
\text{LCB}_i = \mu_i - \alpha\,\sigma_i,
\qquad
\text{UCB}_i = \mu_i + \beta\,\sigma_i,
$$

1. **Optimistic selection (LCB).** choose
   $i^\star = \arg\min_i \text{LCB}_i$. For loss minimisation, the *lower*
   confidence bound is the optimistic value: we act as if the chosen action could
   be as good as plausibly possible. This gives under-observed actions a chance
   to be tried and learned about (the explore half).

2. **Confidence-aware creation (UCB).** **create** iff
   $\text{UCB}_{i^\star} > c$; otherwise **reuse** $i^\star$. We commit to reuse
   only when we are confident — even in the pessimistic case — that the best
   candidate beats paying for a guaranteed-good new action.

> **Why "doubly optimistic".** The rule is optimistic on *both* sides of the
> comparison: optimistic about the *reuse* option through LCB selection, while
> the *create* option's value is itself optimistic (a freshly created action is a
> near-perfect fit, so its expected mismatch is ~0 at known price $c$). The two
> bounds answer two different questions — LCB: *which* action looks most
> promising? UCB: is even that good enough to trust over creating? Compactly:
> **create when reuse is expected to be poor (high $\mu$) or too uncertain to
> trust (high $\sigma$) relative to $c$.**

Two limiting checks fall out immediately and are unit-tested:
$c \to 0$ ⇒ $\text{UCB} > c$ always ⇒ create always;
$c$ large ⇒ $\text{UCB} \le 1 < c$ ⇒ reuse always. As $\sum_j w_{ij}$ grows,
$\sigma_i \to 0$, so the agent stops creating out of uncertainty and reuses a
proven action — uncertainty-driven creation is a transient, learning cost.

## 6. Experimental setup

- **Data.** A synthetic but realistic support stream (`data/`): 12 seed FAQ
  actions and 66 incoming queries across 10 intents. Six categories are
  *well-covered* by the FAQ (reuse should win); four (`subscription_cancel`,
  `sales_followup`, `crm_update`, `meeting_summary`) have **no** seed action, and
  arrive in clusters so that a single creation can be amortised over later,
  similar queries.
- **Embeddings.** Deterministic TF-IDF by default (offline, no keys);
  sentence-transformers auto-used if installed.
- **Loss.** $\gamma=2$, $\lambda=0.4$, noise std $0.03$ (`LossModel`).
- **Sweep.** Every policy at $c \in \{0.05, 0.10, 0.20, 0.35, 0.50\}$, seed $0$.
- **Metrics.** cumulative total / mismatch / creation cost, number created, final
  library size, average loss, create rate.

## 7. Results summary

Average total cost per query (lower is better):

| policy | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NearestReuse | 0.614 | 0.614 | 0.614 | 0.614 | 0.614 |
| FixedProb(0.25) | 0.414 | 0.424 | 0.444 | 0.473 | 0.503 |
| FixedProb(0.5) | 0.241 | 0.263 | 0.307 | 0.373 | 0.439 |
| AlwaysCreate | 0.050 | 0.100 | 0.200 | 0.350 | 0.500 |
| **DoublyOptimistic** | **0.050** | **0.077** | **0.123** | **0.172** | **0.221** |

- The adaptive policy is best at **every** cost; its margin over the best
  baseline grows from ~23% ($c{=}0.10$) to ~50% ($c{\ge}0.35$).
- Its **create rate decreases monotonically** with $c$ ($1.00\to0.21$) while
  every baseline's rate is flat — it genuinely responds to price.
- On the cost–quality plane (creation effort vs. mismatch loss) it traces a
  frontier that **dominates** `FixedProbability`: at a comparable creation
  budget it achieves a small fraction of the mismatch, confirming that deciding
  *where* to create — not just how often — is the source of the gain.
- Robustness: it still beats every cost-*unaware* baseline under a *mis-specified*
  prior (`prior_power=1`), and avg-loss varies by ≈0.001 across seeds.

### 7.1 Ablation — does the "doubly optimistic" machinery pay off?

`StaticThreshold` is the myopic ablation of the main policy: the same cost-aware
comparison, but **no confidence bounds and no learning** — create iff
$d(x, a_\text{nearest})^{\text{prior\_power}} > c$. Comparing the two isolates
the value of optimism + online learning.

| prior | policy | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| well-specified (p=2) | StaticThreshold | 0.038 | 0.068 | 0.113 | 0.172 | 0.220 |
| well-specified (p=2) | DoublyOptimistic | 0.050 | 0.077 | 0.123 | 0.172 | 0.221 |
| mis-specified (p=1) | StaticThreshold | 0.039 | 0.077 | 0.140 | 0.194 | 0.222 |
| mis-specified (p=1) | DoublyOptimistic | 0.050 | 0.082 | 0.153 | 0.215 | 0.230 |

**Honest finding.** On this **stationary, low-noise, distance-informative**
benchmark the myopic threshold is a **strong baseline that slightly beats** the
confidence-bound policy — the "cost of optimism" is about **+0.007** average loss
with a well-specified prior and **+0.012** when the prior is mis-specified. Both
cost-aware policies still beat the cost-*unaware* baselines by a wide margin (§7).

**Why.** When the prior is already informative and feedback is nearly noiseless,
the LCB/UCB bonus mostly buys *exploratory creations* that do not earn back their
cost, and there is little hidden structure for online learning to recover. The
lever that matters most here is plain **cost-awareness**, which both policies
share.

**When should optimism help?** Precisely where this benchmark is easy: noisy or
ambiguous reuse feedback, weakly-informative priors (embedding distance a poor
proxy for the true loss), and non-stationary streams where past observations must
be down-weighted. Demonstrating that regime is the natural next step (§9); we
report this negative result rather than tune the benchmark to favour the method.
See `scripts/run_ablation.py` and `results/ablation_prior_sensitivity.png`.

## 8. Limitations

- **No formal guarantee.** The policy is inspired by optimism-based methods;
  regret is not analysed.
- **Synthetic, calibrated loss** rather than human-judged answer quality; the
  convex distance assumption is the policy's modelling choice (shown non-fragile).
- **Semi-online embeddings** (vocabulary fixed up front for TF-IDF).
- **Greedy, idealised creation**: gold response stored verbatim; no generation
  variance, deduplication, or library eviction; small dataset.

## 9. Possible extensions

1. **Finite/budgeted library** with eviction → a knapsack-flavoured
   create-reuse-replace problem.
2. **Regret analysis** against a best-fixed-library (or hindsight clustering)
   comparator; principled scheduling of $\alpha, \beta$ toward a target create
   rate or creation budget.
3. **Non-stationarity**: time-decayed observation weights for drifting intents.
4. **Real corpora & embeddings**: drop in a public support dataset and
   sentence-transformers; the data contract is just
   `category, query_text, gold_response`.
5. **Richer creation model**: variable creation cost/quality, partial reuse
   (retrieve-then-edit), and tool/skill synthesis as the "action".

---

*References (directional, not exhaustive):* online learning and the
explore/exploit trade-off (multi-armed and contextual bandits; optimism / UCB);
and the create-or-reuse view of online decision making with generative action
sets that motivates this prototype.
