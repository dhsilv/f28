# F28 Codebase Audit

**Date:** 2026-04-19
**Last updated:** 2026-04-19 (post-fix pass)
**Scope:** `f28/` — full Python codebase (alpha, engine, execution, risk, signals, strategy, ops, main.py, backtester)
**Purpose:** Pre-C++ port correctness review. Identify logic bugs, unit/scale mismatches, spec-vs-code drift, and questionable design choices that must be resolved *before* porting to `f28-cpp/` for Strategy Studio.

**Legend:** ~~strikethrough~~ = fixed. Each fixed item carries a `[FIXED: ...]` note pointing at the commit/file. Unmarked items remain open.

---

## Executive Summary

**Post-fix status (2026-04-19):** all 4 critical items and most questionable-decision / inconsistency items are resolved. New infra: `constants.py`, `config.json`, `config.py`, `tick_contract.py`. Remaining open work is logging cleanup (Q5/M6), threshold tuning (I2), and minor defensive cleanups — none are correctness blockers for the C++ port.

The codebase is in reasonable shape for a research-stage strategy, but there are a handful of issues that would cause silent correctness problems if carried into the C++ port verbatim. Two categories dominate:

1. **Time-basis inconsistencies** between the EKF's default `dt` and the master's computed `dt`. Same variable, two different unit conventions — will bias every process-noise term in live operation.
2. **Scattered hyperparameters** (VPIN / KL thresholds, kappa maps, burn-ins, liquidity penalties) with no single source of truth. Multiple files must be edited in lockstep to recalibrate, which is fragile and will be much worse in two-language form.

No showstopper correctness bugs were found — the math of the EKF, HMM, Almgren–Chriss schedule, and Hurst / VPIN / KL estimators all check out on careful read. But there are several *defensive* gaps (NaN guards happening too late, missing shape validation on the tick contract, silent CSV column omissions) that should be closed before porting.

Findings are ordered by severity. A to-do list prioritized for the port is at the end.

---

## Critical — Correctness / port-blocking

### ~~C1. `dt` unit mismatch between EKF default and master's computation~~ [FIXED]
**[FIXED: new `constants.py` with `SECONDS_PER_YEAR = 365.25*24*3600`; imported by both `ekf_overlay.py` and `f28_master.py`. One constant, one import.]**

- **Files:** [alpha/ekf_overlay.py:48](f28/alpha/ekf_overlay.py:48), [strategy/f28_master.py:204](f28/strategy/f28_master.py:204)
- **What:** The EKF's `default_dt = 1.0 / (252.0 * 6.5 * 3600.0)` assumes a *trading-time* year (equities convention: 252 days × 6.5 hrs). `f28_master._compute_dt()` divides elapsed seconds by `365.25 * 24.0 * 3600.0` — a calendar-time year. The two are ~1.7× apart.
- **Why it matters:** The process-noise term `_process_noise(dt)` is linear in `dt`. A systematic 1.7× scale error on process noise permanently biases the Kalman gain — the filter will either under- or over-trust new observations depending on which branch feeds it. On any cold-start or post-gap tick that falls back to `default_dt`, the mismatch also produces a discontinuity in state uncertainty.
- **Fix:** Pick one convention (calendar is correct for CL per the comment at `f28_master.py:197-200`) and enforce it everywhere. Define `SECONDS_PER_YEAR = 365.25 * 24 * 3600` in one place and import it into both modules.

### ~~C2. EKF NaN guard fires after NaN has already propagated~~ [FIXED]
**[FIXED: `ekf_overlay.py:_update` now pre-validates `S_t`/`F_market`/`tau`/`r` with `math.isfinite`, early-returns at `tau<=0` or `S_t<=0`, and checks `F_theo`/`H` before building `S_cov`. Original `S_cov` guard kept as belt-and-suspenders.]**

- **File:** [alpha/ekf_overlay.py:107-126](f28/alpha/ekf_overlay.py:107)
- **What:** `S_cov = H * self.P * H + self.R` is computed unconditionally. The guard `if S_cov <= 0 or not np.isfinite(S_cov): return` runs after the arithmetic. If `H` or `F_theo` overflows (e.g. very large `tau` in a misrouted tick, or degenerate inputs), `S_cov` can be `Inf`/`NaN` and the `return` path still leaves `self.y` untouched — which is fine here — but the same defensive pattern elsewhere in the file doesn't guard the inputs.
- **Why it matters:** Currently benign, but any change that moves arithmetic before the guard (common refactoring mistake in a port) will leak NaN into `self.y` and silently poison every downstream signal.
- **Fix:** Validate `tau`, `F_theo`, and `H` *before* using them: `if not (math.isfinite(F_theo) and math.isfinite(H)): return`. Keep the `S_cov` guard as a belt-and-suspenders check.

### ~~C3. No validation of the tick contract on entry~~ [FIXED]
**[FIXED: new `tick_contract.py` with `TickData` TypedDict + `validate_tick()`. `f28_master.on_tick` does full validation on the first tick, cheap invariants (l3 length) on every subsequent tick. Also resolves I3.]**

- **Files:** [strategy/f28_master.py](f28/strategy/f28_master.py) (`on_tick`), [engine/backtester.py:56](f28/engine/backtester.py:56)
- **What:** `on_tick` trusts the input dict to contain every field listed in F28_STRATEGY.md Appendix B, with the right shapes. `l3_features` must be length-3 (HMM emission is hardcoded 3-D at [signals/hmm_regime.py:133](f28/signals/hmm_regime.py:133)). `curve_prices` / `curve_symbols` must be in `[F1, F2, F3, ...]` order (PCA at [alpha/curve_geometry.py:162](f28/alpha/curve_geometry.py:162) assumes slot 0 = F1). None of this is checked.
- **Why it matters:** A CSV with columns in the wrong order will produce silently wrong edges ("roll into F3" when you meant F2) rather than a loud failure. In a research setting this is a footgun; in live it's a loss event.
- **Fix:** Add a `_validate_tick(tick_data)` called once on boot (cheap field/shape check) plus cheap invariants on every tick (`curve_symbols[0] == "F1"`, `len(l3_features) == 3`). The backtester should also assert required CSV columns before iterating rows.

### ~~C4. EKF state has no explicit reset / reseed hook~~ [FIXED]
**[FIXED: `ConvenienceYieldEKF.reset(keep_y=True)` added; `f28_master` detects F1 symbol change and calls `ekf.reset(keep_y=True)` at roll boundary. Rationale documented in the reset() docstring: convenience yield is a market state, not a per-contract state, so we keep `y` and only reseed `P`.]**

- **File:** [alpha/ekf_overlay.py:40-71](f28/alpha/ekf_overlay.py:40)
- **What:** `self.y = theta` at construction is the only initial condition. There's no public method to reset the filter on a new contract roll or after Totem halts. The master keeps the same `ConvenienceYieldEKF` instance across rolls ([strategy/f28_master.py](f28/strategy/f28_master.py)).
- **Why it matters:** After a roll, the tau discontinuity and the new F1/F2 pair represent a different physical situation. Carrying forward `self.y` and `self.P` is defensible (convenience yield is a market state, not a per-contract state) but it's never explicitly *argued*. The spec doesn't address this either.
- **Fix:** Either (a) document that the filter state is explicitly preserved across rolls and why, or (b) add a `reset(keep_y: bool = True)` method and decide policy per-event (roll vs. Totem halt vs. day-boundary).

---

## Questionable Decisions

### ~~Q1. Hyperparameters defined in multiple places~~ [FIXED]
**[FIXED: new `config.json` is the single source of truth for every hyperparameter. `config.py` loads it (strips `_comment` keys, normalizes stringified int-keyed kappa_map). `main.py` `**cfg[...]`-splats into every constructor. Override the path via `F28_CONFIG` env var. The C++ port will parse the same file.]**

- **Files:** [signals/frank_module.py:42](f28/signals/frank_module.py:42), [main.py:55](f28/main.py:55), [execution/almgren_chriss.py:50](f28/execution/almgren_chriss.py:50)
- **What:** VPIN threshold (0.75), KL entropy limit (2.0), VPIN bucket volume (1000), and the kappa map `{0:0.1, 1:1.5, 2:5.0}` all appear in at least two places as defaults.
- **Why it matters:** Recalibration is a cross-file grep. In the port, two languages and two build systems make this worse. Also invites accidental divergence between offline training (`ops/train_f28_models.py`) and live strategy.
- **Fix:** One `config.py` (or `config.json`) loaded at boot. Defaults live there; every module receives its parameters via `__init__`, no module-level magic numbers.

### Q2. Kappa map printed as defaults without a calibration warning
- **File:** [execution/almgren_chriss.py:50-54](f28/execution/almgren_chriss.py:50)
- **What:** The spec (Part IX) explicitly says kappa values are "initial heuristics; final calibration by minimizing realized slippage." The code silently uses the heuristics if `kappa_map=None`.
- **Why it matters:** Easy to forget during a backtest that you're on uncalibrated kappas.
- **Fix:** Log a loud warning when the default map is used. Or require the caller to pass one explicitly.

### ~~Q3. F3 liquidity-penalty multiplier hardcoded inline~~ [FIXED]
**[FIXED: `PCAModel.__init__` now takes `f3_penalty_multiplier: float = 1.5`, used in the `cost_f3` expression. Exposed in `config.json` under `pca`.]**

- **File:** [alpha/curve_geometry.py](f28/alpha/curve_geometry.py) (search for `* 1.5`)
- **What:** The `1.5×` F3 penalty multiplier from the spec is embedded in the expression rather than a named parameter. Same for the `0.5 *` spread coefficient.
- **Fix:** Promote both to `__init__` parameters with the spec-derived defaults.

### ~~Q4. CL-specific `price_floor=5.0` hardcoded in `main.py`~~ [FIXED]
**[FIXED: Totem params live under `totem` in `config.json`; swapping to a non-negative-price commodity means setting `price_floor: null`. `commodity` field at top-level of the config documents intent.]**

- **File:** [main.py](f28/main.py) (Totem instantiation)
- **What:** The spec claims generalizability to other commodities, but the Totem floor is a CL-specific value hardwired into `build_strategy`.
- **Fix:** Move commodity-specific parameters into a per-commodity config block.

### Q5. `print()` used for operational events
- **Files:** [risk/totem_protocol.py:159](f28/risk/totem_protocol.py:159), [strategy/f28_master.py:228](f28/strategy/f28_master.py:228)
- **What:** Halt notifications and order emissions are plain `print`. No severity, no redirection, no machine-parseable format.
- **Why it matters:** Fine for research, unacceptable in the C++ port where Strategy Studio expects structured logging.
- **Fix:** `logging` module now. Port the pattern to spdlog (or Strategy Studio's logger) in C++.

### Q6. EKF default `dt` trading-time basis (see also C1)
- **File:** [alpha/ekf_overlay.py:48](f28/alpha/ekf_overlay.py:48)
- **What:** Even if C1 is fixed, the `default_dt` fallback used at cold start assumes ~1-second cadence. Any tick gap larger than that (post-overnight, post-halt) silently uses a wrong `dt`.
- **Fix:** On the first tick after a gap longer than some threshold, either predict through the actual elapsed gap or reset `P` to `initial_uncertainty` and skip the predict step.

### Q7. Final-slice flush in Almgren–Chriss not asserted non-negative
- **File:** [execution/almgren_chriss.py:81-100](f28/execution/almgren_chriss.py:81)
- **What:** The final-slice branch uses `self.remaining_qty` directly. With the fractional accumulator and `min()` guard, this should never be negative, but there's no explicit `assert remaining_qty >= 0`. A subtle bug anywhere else (e.g. a duplicated step call) could push it negative and ship a short order on what is supposed to be a roll close.
- **Fix:** `assert self.remaining_qty >= 0` before the final flush, and clamp with `max(0, ...)`.

### Q8. Burn-in of 30 ticks is parameterized but its meaning is undocumented
- **File:** [alpha/curve_geometry.py:40](f28/alpha/curve_geometry.py:40)
- **What:** Constructor takes `burn_in: int = 30`, but the value is not justified anywhere. Why 30? Is it "enough ticks to get a reasonable EWMA seed"?
- **Fix:** One-line comment stating the rationale so the port doesn't arbitrarily pick a different number.

### Q9. HMM regularization `1e-10` hardcoded
- **File:** [signals/hmm_regime.py:119](f28/signals/hmm_regime.py:119)
- **What:** `cov + np.eye(D) * 1e-10`. Value is fine for full-rank covariance; near-singular covariance from short training data could need more.
- **Fix:** Promote to a parameter.

---

## Inconsistencies

### ~~I1. HMM internal-state-to-logical-state map not serialized~~ [FIXED]
**[FIXED: `train_f28_models.train_hmm` now computes the sort-by-trace state map and emits it in the params JSON under `state_map`. `hmm_regime.load_from_params` trusts it verbatim if present, falls back to re-deriving with a `RuntimeWarning` if loading an older JSON. C++ port now has an authoritative mapping.]**

- **File:** [signals/hmm_regime.py:83-110](f28/signals/hmm_regime.py:83) (`_map_hidden_states`)
- **What:** The HMM trains with arbitrary internal state indices, then remaps at load time by sorting covariance traces (Quiet/Trending/Distressed). Loading a trained HMM from JSON recomputes this mapping. The JSON itself does not encode "state 2 is Distressed" — it's derived.
- **Why it matters:** The C++ port must replicate the *exact* same sort-by-trace heuristic, otherwise `kappa_map[regime]` will index a different state than the Python version did. Worse, if the covariances are close, floating-point noise across languages could flip the mapping.
- **Fix:** Serialize the resolved state map in the JSON payload alongside the HMM matrices. Port reads the map directly; no re-derivation required.

### I2. Spec says VPIN/KL thresholds should be "tuned on historical rolls" — they aren't
- **Files:** F28_STRATEGY.md vs. [main.py:55-61](f28/main.py:55)
- **What:** Defaults live as hardcoded values. `ops/train_f28_models.py` trains the KDE baseline and HMM but does not emit tuned thresholds.
- **Fix:** Either (a) add a tuning step to `train_f28_models.py` that fits thresholds against historical roll windows and emits them into the params JSON, or (b) explicitly document that these are untuned heuristics.

### ~~I3. Tick-schema contract exists only in the spec, not in code~~ [FIXED]
**[FIXED by C3: `tick_contract.py` now defines `TickData` TypedDict + `validate_tick()`. This file is the source of truth for the tick schema and the shape the C++ port's tick struct will mirror.]**

- **Files:** F28_STRATEGY.md Appendix B vs. code
- **What:** No `TypedDict` / dataclass / JSON schema for the tick contract. `on_tick` receives `dict`.
- **Fix:** Define a `TickData` TypedDict or dataclass — this also becomes the source of truth for the C++ port's tick struct. Makes I3 and C3 enforceable at the type level.

### ~~I4. Two layers of "if None then use defaults" obscure calibration state~~ [FIXED]
**[FIXED: `main.build_strategy` now emits `logger.warning` when Frank baseline or HMM matrices are missing. `config.py` raises `FileNotFoundError` if `config.json` is absent — no silent-default fallback for hyperparameters.]**

- **Files:** [main.py:62-65](f28/main.py:62) (frank_kde param loading), [main.py `build_strategy`](f28/main.py) (hmm params loading)
- **What:** `build_strategy` silently accepts `params=None` and uses module defaults. No log line says "running with uncalibrated baseline."
- **Fix:** Loud warning on any `None` fallback.

---

## Minor / Cleanup

- **M1.** [main.py:62-65](f28/main.py:62) — `import numpy as np` inside a conditional branch; move to module top.
- **M2.** [ops/train_f28_models.py:178](f28/ops/train_f28_models.py:178) — data directory hardcoded; take `argv[1]` or an env var.
- **M3.** [engine/backtester.py:67-94](f28/engine/backtester.py:67) — `_row_to_tick` lacks a docstring listing required CSV columns and the `;`-separated array convention.
- **M4.** [signals/frank_module.py:96](f28/signals/frank_module.py:96) — `timestamp` parameter received but unused; either log it or drop it from the signature.
- **M5.** KDE singular-covariance exceptions at `curve_geometry.py` are caught silently; add a counter / periodic warning so a persistently-failing KL estimator doesn't go unnoticed.
- **M6.** Most modules would benefit from module-level `logger = logging.getLogger(__name__)` wiring now, so the port can use the same structured pattern from day one.

---

## Verified Correct (No Action)

The following were carefully checked and are correct — listing them to save time on the port:

- **Hurst aggregation** ([signals/totem_protocol.py](f28/signals/totem_protocol.py)): non-overlapping 2-aggregation, `ddof=1` bias cancels in the ratio, flat-price edge case returns `H=0.0` which correctly trips pinning.
- **HMM Gaussian log-density** ([signals/hmm_regime.py:127-135](f28/signals/hmm_regime.py:127)): sign and constant terms are right.
- **OU exact transition** ([alpha/ekf_overlay.py:100-105](f28/alpha/ekf_overlay.py:100)): mean and variance match the closed-form OU solution; `_process_noise` handles the `kappa*dt → 0` limit with a Taylor fallback.
- **VPIN cold-start bucket logic** ([signals/frank_module.py:121-135](f28/signals/frank_module.py:121)): zero-sign ticks are correctly dropped, not misclassified; buckets always represent exactly `bucket_volume`.
- **Curve-geometry EWMA seeding** ([alpha/curve_geometry.py:100-110](f28/alpha/curve_geometry.py:100)): burn-in builds true running mean, EWMA takes over cleanly — no first-observation bias.
- **Almgren–Chriss schedule math** ([execution/almgren_chriss.py](f28/execution/almgren_chriss.py)): `sinh` trajectory, fractional accumulator, and final-slice flush all match Appendix D.

---

## TO-DO (prioritized for pre-port)

**Must fix before port (correctness / will silently corrupt C++ behavior):**
1. ~~**C1** — Unify `dt` time basis.~~ **DONE.**
2. ~~**C3 / I3** — Define a `TickData` TypedDict; validate on entry.~~ **DONE.**
3. ~~**I1** — Serialize HMM state map into the params JSON.~~ **DONE.**
4. ~~**C4** — EKF reset across rolls.~~ **DONE.**

**Should fix before port (design will be painful to untangle in two languages):**
5. ~~**Q1** — Single config file for all hyperparameters.~~ **DONE** (`config.json`).
6. **Q5 / M6** — Replace remaining `print()` calls (Totem halt notifications, f28_master order emissions, ETL progress) with `logging`. Partial progress: `main.py`, `ekf_overlay.py`, `f28_master.py` new callsites use `logging`; old `print()` lines in Totem and the order sender remain.
7. ~~**Q4** — Commodity-specific constants into config.~~ **DONE** (`totem.price_floor` in config; `commodity` field documents intent).
8. **I2** — Either add threshold-tuning to `train_f28_models.py` or document that thresholds are untuned. (Currently flagged in `config.json` via `_comment`.)

**Nice-to-have cleanup (can happen after port):**
9. ~~**C2** — Pre-validate EKF inputs before arithmetic.~~ **DONE.**
10. ~~**Q2 / I4** — Loud warnings on None-params fallbacks.~~ **DONE** (I4). **Q2 (default-kappa-map warning) NOT needed** — the kappa_map now always comes from `config.json`, never falls back to hardcoded defaults in the live path.
11. ~~**Q3** — F3 penalty multiplier as named parameter.~~ **DONE.** Spread coefficient (`0.5`) remains inline.
12. **Q7** — Explicit `assert remaining_qty >= 0` in Almgren–Chriss final flush.
13. **Q6** — Post-gap dt handling in EKF.
14. **Q8, Q9, M1–M5** — Minor cleanups.

**Remaining pre-port work:** items 6, 8, 12, 13, and the minor cleanups in 14. None are correctness blockers; 6 and 8 are the most impactful. Safe to begin port scaffolding in parallel.
