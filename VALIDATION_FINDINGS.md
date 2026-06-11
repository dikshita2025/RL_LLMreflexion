# LLM4Teach — Validation Findings

Validation pass after the 15-requirement implementation. Focus: confirm the
architecture is correct and decide whether it actually helps learning. Hardware:
**Windows, CPU-only** (the env asserts CPU). Ollama present (`qwen2.5:3b` planner,
`qwen2.5:7b` reflector).

---

## 1. Functional checks — PASS

**Teacher-only success (offline planner, 10 episodes/env):**

| env | success | avg_len | key pickup | door open |
|---|---|---|---|---|
| SimpleDoorKey | 1.00 | 25.8 | 1.00 | 1.00 |
| ColoredDoorKey | 0.50 | 89.9 | 1.00 | 0.50 |
| LavaDoorKey | 1.00 | 24.8 | 1.00 | 1.00 |
| TwoDoor | 0.00 | 150.0 | 1.00 | 0.00 |

- **Key pickup = 100% in every env** — the stateless Pickup recovery (req 14) works everywhere.
- ColoredDoorKey 0.50: agent always grabs *a* key but sometimes the wrong color → drop/regrab (avg 90 steps).
- TwoDoor 0.00: picks the key every time, but with view-size 3 and a 150-step budget it explores only ~15–20% of the 20×20 room, so it never reaches the wall-mounted doors. **Pure exploration/step-budget limit**, not a skill bug.

**Reflection memory (req 9, 10, 15):**
- Hash dedup: the same reflection added ×10 → **1 entry, frequency=10** (success/failure counts tracked, 5/5).
- Coordinate / location / path rejection — all of these are rejected on write:
  `Key at (7,3)`, `upper-right corner`, `row 3 / col 7`, `go north then east`, `move south`.
  Accepted: `Pick up the key before opening the door`, `Search frontier regions…`, `Avoid oscillation…`.
- Retrieval ordering verified: cluster relevance → success rate → frequency.
- Added retrieval introspection: `ReflectionMemory.verbose_retrieval=True` prints each
  retrieval; `memory._last_retrieval` exposes (query_cluster, retrieved_cluster, match,
  success_rate, frequency) for notebook inspection.

---

## 2. Phase 1 vs Phase 2 ablation (offline, early-training)

Ran the real research pipeline (`run_research_pipeline`, the notebook's Cell-9 path) for
`ppo_only` vs `planner`, SimpleDoorKey, 50 iterations, offline planner, seed 1.
**These are early-training numbers (50 iters), NOT convergence — PPO configs target 1000–2000 iters.**
The purpose is to (a) validate the ablation harness end-to-end and (b) show the kickstarting signal.

| metric (50 iters, SimpleDoorKey) | Phase 1 `ppo_only` | Phase 2 `planner` |
|---|---|---|
| student success | 0.00 | 0.00 |
| student avg reward | 0.000 | 0.000 |
| student avg ep len | 150.0 | 150.0 |
| **teacher upper bound** | **0.00** | **1.00** |
| **teacher agreement** | **0.16** (≈ 1/6 chance) | **0.34** (> chance) |
| **interventions** | **0** | **202** |
| student–teacher gap | 0.00 | 1.00 |

**This cleanly validates the phase-gating and the harness:**
- `ppo_only` is genuinely teacher-less — teacher upper bound 0.00, agreement ≈ random
  (1/6 ≈ 0.167), 0 interventions. The `_NoTeacher` stub behaves exactly as intended.
- `planner` has a working teacher — upper bound 1.00 (offline teacher solves SimpleDoorKey),
  agreement 0.34 (PPO's action distribution is measurably pulled toward the teacher), 202
  interventions (failure detector active).
- **Student success = 0 for both is the 50-iteration scale limit** (configs target 1000–2000;
  both students still time out at 150 steps), NOT a result about the method. The real
  early-training signal is the agreement gap **0.16 → 0.34**: kickstarting is working.

Full table: `results/summary.csv`.

---

## 3. Why the "does reflection improve convergence?" experiment is blocked here

Three measured facts, not opinions:

1. **Offline reflection is inert.** Instrumented `query_codex` calls in offline mode = **0**.
   Offline plans are all pre-cached in `plans_dict`, so `_build_prompt_with_reflection()`
   (the only place memory is injected) is never reached. Offline Phase 3 ≡ Phase 2.

2. **Online is ~18.5 s per LLM call** (`qwen2.5:3b`, CPU). Online planning calls the LLM on
   every *uncached* state; a 50-iteration online run is multi-hour. Infeasible interactively.

3. **The cache key excludes reflection.** `planner.plan(text)` keys on the symbolic
   observation string and returns cached plans without re-consulting memory. So reflection
   influences a plan only on a **cache-miss** (first encounter of each of ~6–14 symbolic
   states) or after a **cache invalidation** (confidence-based, ~15 failures). Across a long
   run it touches a handful of decisions — its leverage is small *by design*.

Net: proving "Reflection > Planner" requires (a) the online teacher and (b) enough
compute for convergence — i.e. a GPU/API run, not this CPU box. See §5 for the recipe.

---

## 4. PROPOSAL — make reflection actually influence planning (no code changed yet)

Reflection currently helps only at cache-miss/invalidation. To give it real leverage while
respecting **req 13 (plan stability — no aggressive replanning)**, here are three options,
least → most invasive. Recommendation: **Option B**.

### Option A — Reflection-gated cache invalidation (smallest change)
When a new reflection lands in a cluster relevant to a cached state that has been failing,
nudge that state's confidence down by a small amount so it re-plans sooner (re-injecting the
new strategy). Bounded: only failing states, only on genuinely new (frequency==1) reflections.
- *Pro:* tiny, localized, stability-preserving (only touches already-failing states).
- *Con:* still no effect on states that never fail.

### Option B — Reflection epoch tag in the cache key *(recommended)*
Add a coarse "memory epoch" counter that increments only when a *new* cluster first appears
(not on every duplicate/frequency bump). Fold the current epoch into the cache key:
`key = (symbolic_obs, memory_epoch)`. Plans are reused within an epoch (stable), but when the
strategy repertoire genuinely grows, the next encounter re-plans with the new context.
- *Pro:* reflection provably changes plans as strategies accumulate; re-planning frequency is
  controlled by *new-cluster* events (rare), so stability is largely preserved.
- *Con:* cache hit-rate drops slightly after each new cluster; a few extra LLM calls online.
- *Stability budget:* with ≤7 clusters, at most ~7 epoch bumps per run → bounded re-planning.

### Option C — Always inject, soft-blend (most invasive)
Re-run the LLM with reflection context every N encounters and blend with the cached plan via
`_diversify_plan`. Maximum reflection influence, but the most LLM calls and the largest risk
to plan stability — least aligned with req 13. Not recommended unless on fast hosted inference.

**Evaluation hook (any option):** with retrieval logging already added, a meaningful online
A/B is: Phase 2 (planner) vs Phase 3 (planner + reflection + chosen rewire), same seed, track
success-rate and time-to-first-success. Expect separation only if reflection changes plans.

---

## 5. Recipe to run the full ablation where it's feasible (GPU / hosted LLM)

The notebook `Research_Ablation_Notebook.ipynb` already supports this (now that `Game` accepts
`exp_config`). On Kaggle/Colab GPU or with a DashScope key:

1. Cell 3: `TASK='SimpleDoorKey'`, `RUN_RESEARCH_PIPELINE=True`.
2. LLM backend: `QWEN_BACKEND='dashscope'` + `QWEN_API_KEY=...` (hosted inference avoids the
   18 s/call CPU latency), or `'ollama'` on a GPU host.
3. `PHASE_CONFIGS`: keep `ppo_only`/`planner`/`reflection` at 1000–1500 iters.
4. Run Cell 9 (pipeline) → Cell 11 (ablation table) → Cell 12 (`results/summary.csv`).

For a fair reflection test, run `reflection` **online** and apply a rewire option from §4 —
otherwise reflection cannot move the plans and Phase 3 will track Phase 2.
