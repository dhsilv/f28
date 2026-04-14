# Project F-28: Regime-Aware Futures Roll Strategy

**Underlying:** CL (NYMEX WTI Crude Oil)
**Venue:** RCM Strategy Studio (C++ production) / Python research harness
**Status:** Research build; calibration not yet signed off for live deployment

---

## Part 0: What This Strategy Does (Plain English)

A futures contract is a binding agreement to buy or sell a commodity (here, crude oil) at a specified future date. Every futures contract has an expiration. If you want to keep a long-term exposure to crude — and most institutional participants do — you have to periodically close the expiring contract and open the next one. This is called a **roll**, and every trader with a futures position does it on a recurring calendar.

Rolling is not free. Thousands of participants are trying to do the same thing at roughly the same time. If you roll on a lazy fixed schedule, faster traders front-run you. If you panic-roll at the last minute, the expiring contract's order book is already hollow and you pay a wide bid-ask spread. If you blindly roll into the "next" contract, you may be ignoring a cheaper, further-out contract. And if the market is currently in a jump regime or a delivery-day squeeze, your pricing models are lying to you and you should not be algorithmically rolling anything at all.

F-28 addresses all of this. It profits by:

1. **Timing the roll intelligently** rather than on a calendar — waiting for the order book of the expiring contract to start signalling its own collapse, then acting.
2. **Routing to the best target contract** — usually F2 (one month out), but occasionally F3 (two months out) when the curve shape makes F3 meaningfully cheaper net of trading costs.
3. **Executing dynamically** — breaking the roll into small pieces whose size adapts tick-by-tick to the current microstructure regime (Quiet / Trending / Distressed).
4. **Refusing to trade when the physics is broken** — detecting jumps, price pinning, or catastrophic regime shifts and halting all activity until a human re-validates.

The edge per roll is small — a few basis points against a naive monthly roll — but the strategy runs dozens of times per year and scales linearly with position size. It makes its money in calm-to-moderate regimes. It does not try to make money in crises; in crises it refuses to trade.

---

## Part I: Why This Strategy Exists

### The roll problem

Crude oil futures at NYMEX list every calendar month. The overwhelming majority of open interest and volume concentrates in the front contract (F1). As F1 approaches its last trading day (three business days before the 25th of the prior month), two things happen nearly simultaneously:

- **Liquidity migrates out of F1.** Participants who do not intend to take physical delivery at Cushing, OK must exit. Market-makers widen quotes or withdraw altogether.
- **F2 becomes the new front.** Liquidity arrives in F2 from both incoming traders and from those rolling out of F1.

The textbook "constant-maturity" approach — roll on the 5th-to-last trading day, entire position, into F2 — is catastrophically naive:

- **Front-running.** The calendar is publicly known. Faster traders position ahead of the known flow.
- **Supply shocks.** If Cushing inventories are constrained or OPEC announces an unexpected cut, the curve can go into steep backwardation. A fixed roll-to-F2 locks in a bad entry.
- **Curve opportunities.** Sometimes F3 carries a liquidity premium that outweighs its wider spread, making it the better target. The static rule never sees this.
- **Delivery-algo pinning.** Institutional delivery programs occasionally pin price in a narrow range during the last day or two of F1. Any roll executed into that regime is a sitting duck.
- **Real jumps.** EIA inventory releases on Wednesday, OPEC announcements, geopolitical shocks — all produce discontinuous price moves that invalidate any model assuming smooth diffusion.

F-28 replaces the calendar with a **statistical trigger**, the sequential roll with an **optimized target**, the TWAP execution with a **regime-adaptive trajectory**, and the implicit "models are always valid" assumption with an **explicit validity check**.

### The profit model

A roll executed with F-28 earns its edge across several margins:

| Margin | Typical bps/roll | Mechanism |
|---|---|---|
| Avoided slippage | 1–4 bps | Trigger fires *before* F1 liquidity fully collapses; better fills than a last-day panic roll |
| Curve selection | 0–3 bps | When F3 is genuinely cheaper net of spread, we capture it; otherwise F2 (no adverse selection) |
| Impact reduction | 1–3 bps | Almgren-Chriss trajectory vs. TWAP, weighted by regime |
| Crisis avoidance | ∞ bps | Totem halts before the strategy does anything catastrophic |

The edges are additive in expectation but not independent — good regime detection amplifies everything downstream. The strategy's P&L is therefore not a product of four independent bets but a joint function of one integrated decision process.

---

## Part II: System Architecture

F-28 is organized as a **finite state machine** orchestrated by a master controller that injects five independent engines via strict dependency injection. The controller contains zero math; each engine owns its own mathematical domain.

```
                       ┌─────────────────────────┐
                       │    F28Strategy (FSM)    │
                       │   HOLDING → ROLLING →   │
                       │       COMPLETED         │
                       │           │             │
                       │         HALTED          │
                       └───┬─────────────────┬───┘
                           │                 │
                           ▼                 ▼
        ┌──────────────────────────┐   ┌──────────────────┐
        │   Phase 1 / Phase 2.5    │   │    Phase 4       │
        │                          │   │    (parallel)    │
        │ Frank ─┐                 │   │                  │
        │         ├─► EKF Overlay  │   │  Totem Protocol  │
        │ PCA  ──┘                 │   │  (RV/BPV, Hurst, │
        │         ▼                │   │   price floor)   │
        │     Target Tenor         │   │                  │
        └─────────────┬────────────┘   └──────────────────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │       Phase 3            │
        │                          │
        │  HMM (Forward O(1))      │
        │       │                  │
        │       ▼                  │
        │  Almgren-Chriss          │
        │  sinh trajectory         │
        │       │                  │
        │       ▼                  │
        │  Fractional Accumulator  │
        └──────────────────────────┘
```

The Totem protocol runs in parallel with every other phase. Its decision to halt is authoritative — no other engine can overrule it.

---

## Part III: Phase 1 — The Frank Engine (Signal Generation)

### Objective

Detect the incipient structural collapse of the F1 order book before its physical expiration. Replace calendar-based roll triggers with a joint statistical condition on order-flow toxicity and return-distribution drift.

### Signal 1: Volume-Synchronized Probability of Informed Trading (VPIN)

VPIN (Easley, López de Prado, O'Hara 2012) is a microstructure measure of order-flow toxicity. Standard clock-time volatility fails at high frequency because trading activity clumps non-stationarily; VPIN solves this by bucketing **in volume space** rather than time space.

**Construction:**

- Accumulate incoming trades into equal-volume buckets of size $V$.
- Classify each tick's volume as buy- or sell-initiated using the Lee-Ready tick test. Critically, when price is unchanged across ticks we **inherit the sign of the previous directional tick** rather than splitting the volume 50/50 — this matches the spec and avoids the well-known dilution bias.
- For each bucket, compute imbalance $|V^{(buy)}_\tau - V^{(sell)}_\tau|$.
- VPIN over a rolling window of $n$ buckets is

$$
\text{VPIN} = \frac{1}{n \cdot V} \sum_{\tau=1}^{n} \left| V^{(buy)}_\tau - V^{(sell)}_\tau \right|
$$

**Interpretation.** VPIN approximates the probability that any single transaction in the recent past was driven by informed flow. When informed traders exit F1 ahead of expiry (or ahead of a supply shock), the imbalance spikes and VPIN rises.

**Why this over alternatives.**
- *Clock-time realized volatility* aliases with trading intensity and systematically under-reacts in burst regimes.
- *Queue-position imbalance* (top-of-book size skew) is cheap but easily spoofed by quote flicker.
- *Kyle's lambda* requires a linear regression per bucket and is less stable at tick frequency.

### Signal 2: KL-Divergence of Log Returns

VPIN tells us about order flow but not about whether the price process itself has left its normal regime. For that, we need a distributional test.

**Construction.**

- Offline, on curated historical training data, we accumulate a sample of F1 log returns during known "steady-state" periods and store the raw sample for transmission to the live engine.
- Live, we maintain a rolling window of recent log returns.
- At each evaluation we fit a Gaussian-kernel KDE to both the baseline sample $P$ and the live window $Q$. KDE gives us a non-parametric density estimate — we explicitly do **not** assume normality at any stage.
- We evaluate both densities on a shared grid spanning the union of their supports (critical — otherwise we can silently clip the tail mass that drives the divergence).
- The Kullback–Leibler divergence is

$$
D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx \;\approx\; \sum_i p_i \log \frac{p_i}{q_i}
$$

**Why not a parametric fit.** Crude log returns are empirically leptokurtic with heavy left tails. A Gaussian MLE under-weights the tails; since the tails are precisely the statistic we want to flag on, a parametric fit would de-sensitize the signal to the event it was built to catch.

**Why KL and not Jensen-Shannon or Wasserstein.** KL is asymmetric in the correct direction: high-probability events in the **baseline** that become low-probability in the **live** distribution produce large divergence. That asymmetry is the semantic we want — "the world has stopped looking like our training data," not merely "the two distributions differ."

**Why shared-support grids matter.** Naive KL with mismatched supports either produces infinities (if $q$ has a zero where $p$ has mass) or silently truncates the divergence. We pad the grid by 10% of the union range and floor both densities at $10^{-12}$.

**Performance.** KDE refit is $O(N)$ where $N$ is the live window. We refit every 25 ticks rather than every tick — the signal is stable at second-scale cadence and the per-tick cost would dominate the event loop. The cached KL value is used between refits.

### The AND gate

The Death Signal requires **both** VPIN and KL to exceed threshold:

$$
\text{DeathSignal} \;=\; \mathbb{1}\{\text{VPIN} > \theta_V\} \;\wedge\; \mathbb{1}\{D_{KL} > \theta_K\}
$$

The spec is explicit on this. An OR gate would fire on garden-variety volume bursts (VPIN alone) or on isolated large returns (KL alone). The AND condition requires *both* the order flow and the price process to have departed from normal — a much stronger joint claim.

---

## Part IV: Phase 2.5 — The EKF Overlay (Fundamental Physics)

### Objective

Before we trust any purely-statistical router, verify that the **physical cost-of-carry relationship** is still approximately satisfied. If it is not, the curve is being moved by something PCA was never designed to see, and routing on PCA residuals is unsafe.

### The Gibson-Schwartz one-factor model

The theoretical futures price under continuous cost-of-carry is

$$
F_{t, T} = S_t \cdot \exp\big((r_t - y_t)(T - t)\big)
$$

where $y_t$ is the **convenience yield** — the implicit benefit of holding the physical commodity. In CL this is real and has clear economic drivers: Cushing inventory levels, refinery margins, OPEC supply decisions, and geopolitical risk premia.

The convenience yield itself evolves as an Ornstein-Uhlenbeck process:

$$
dy_t = \kappa(\theta - y_t)\,dt + \sigma_y\,dW_t
$$

$y_t$ is unobservable. We must estimate it online from observed F and S.

### CL-specific observation model

There is no continuously-traded WTI cash tape. The standard practical convention — which we follow — is to **use F1 as the spot proxy and F2 as the observation**. The Gibson-Schwartz relation between F1 and F2 follows directly:

$$
\frac{F_{t,T_2}}{F_{t,T_1}} = \exp\big((r_t - y_t)(T_2 - T_1)\big)
$$

which is mathematically identical to the standard form with $S = F_1$ and $\tau = T_2 - T_1$. The EKF ingests $S_t = F_1$, $F_{\text{market}} = F_2$, and $\tau = \tau(F_2) - \tau(F_1)$.

### Extended Kalman Filter

State: $y_t$ (scalar).

**Predict step** uses the exact OU transition moments rather than the Euler small-$dt$ approximation:

$$
\hat{y}_{t+dt \mid t} = \theta + (y_t - \theta) e^{-\kappa\, dt}
$$
$$
P_{t+dt \mid t} = e^{-2\kappa\, dt} P_t + \frac{\sigma_y^2}{2\kappa}\left(1 - e^{-2\kappa\, dt}\right)
$$

Using the exact form (rather than $Q = \sigma_y^2\, dt$) makes the filter robust to asynchronous ticks and to any sampling rate where $\kappa\, dt$ is not small.

**Update step** linearizes the nonlinear observation $h(y) = S e^{(r-y)\tau}$ around the prior mean. The Jacobian is $H = -\tau \cdot h(y)$. The residual (innovation) drives the Kalman gain update in the standard way.

### Why EKF and not alternatives

- **Particle filter.** Overkill for a scalar state whose nonlinearity is mild (a smooth exponential). The linearization error is tiny compared to the process noise.
- **Unscented Kalman filter.** Handles nonlinearity by propagating sigma points through the observation function. Useful when the Jacobian is unstable or discontinuous — not our situation.
- **Particle filter in future research.** If we later extend to a two-factor Schwartz model (stochastic convenience yield *and* stochastic spot mean reversion), the state space is no longer well-approximated by a linearization and a particle filter becomes warranted.

### The physical-shock override

The spec defines the supply-shock condition as $|y_t| > \text{physical\_limit}$. For CL we use 0.15 — 15% annualized convenience yield is well outside the normal operating range (typically 0–5%) and indicates genuine physical scarcity. When this fires, F-28 **skips PCA entirely and forces the safe sequential F2 roll**. The reasoning: PCA residuals in a supply-shocked curve are dominated by the shock itself, not by relative-value mispricing, so any "edge" PCA reports is an artifact.

### The Gaussian caveat

Any Kalman-family filter Gaussianizes the posterior. This is the one unavoidable Gaussian assumption in F-28. We mitigate it by (a) using full continuous-time OU variance rather than the Euler approximation, and (b) relying on the Totem Protocol (Phase 4) to halt the system before jump regimes push the EKF into territory where the linearization is badly wrong.

---

## Part V: Phase 2 — EWMA-Decayed PCA (Curve Routing)

### Objective

Conditional on the EKF clearing the physics check, select the cheapest target contract (F2 or F3) net of trading costs.

### The covariance update

We maintain an EWMA covariance matrix of log returns across the first five curve points. EWMA with decay $\lambda$ is equivalent to a first-order linear filter on squared-centered returns:

$$
\bar{r}_t = \lambda\, \bar{r}_{t-1} + (1-\lambda)\, r_t
$$
$$
\Sigma_t = \lambda\, \Sigma_{t-1} + (1-\lambda)\, (r_t - \bar{r}_t)(r_t - \bar{r}_t)^\top
$$

with $\lambda$ parameterized from a pandas-style span: $\alpha = 2/(span+1)$, $\lambda = 1 - \alpha$.

**Why EWMA over rolling window.** EWMA has no sharp window boundary — a single extreme observation decays smoothly rather than dropping off a cliff at lookback + 1. This avoids the discontinuities that plague fixed-window estimators during regime transitions.

### The Roll Gap Stitch

On expiry days the contract in slot $k$ physically rolls over. The symbol in slot 1 changes from `CLM25` to `CLN25` (for example), and the naive log return $\log(P_{t,1} / P_{t-1,1})$ is meaningless — it's comparing two *different contracts*, not a return.

**The fix**: per-slot symbol tracking. If the symbol in any slot differs from the previous tick's symbol, that slot's contribution to the covariance update is zeroed out for that tick. Other slots still contribute. A full skip would destroy an entire tick's worth of information every expiry day; the per-slot stitch preserves the unaffected slots.

### Eigen-decomposition and the top-$k$ reconstruction

We decompose $\Sigma_t$ via `eigh` (symmetric PSD), sort eigenvalues descending, and retain the top three eigenvectors. Interpretation:

- **PC1** — parallel shift (level). All loadings positive; dominates variance.
- **PC2** — slope (twist). Front vs. back opposite signs.
- **PC3** — curvature (butterfly). Middle vs. wings.

These three components typically explain >98% of curve variance in CL.

### Residuals and cheap/rich identification

For the current tick, we project the centered return vector onto the top-3 subspace and reconstruct:

$$
\hat{r}_t = P_k P_k^\top (r_t - \bar{r}_t)
$$

The residual $\epsilon_t = (r_t - \bar{r}_t) - \hat{r}_t$ captures everything *not* explained by level/slope/butterfly. Our sign convention:

- $\epsilon_t^{(i)} > 0$ : slot $i$ realized a richer return than its PCA-implied move — **rich, tendency to sell**.
- $\epsilon_t^{(i)} < 0$ : slot $i$ realized a cheaper return than implied — **cheap, tendency to buy**.

Since we are buying the back month, we want the slot with the most negative residual.

### The skip-roll decision

Raw edge is $-\epsilon_t^{(i)}$ for each candidate slot. From this we subtract **trading costs**:

$$
\text{edge}_i = -\epsilon_t^{(i)} \;-\; \tfrac{1}{2}\frac{s_i}{P_i} \;-\; c_i
$$

where $s_i$ is the bid-ask spread in price units, $P_i$ is the slot price, and $c_i$ is a liquidity penalty — higher for F3 (1.5× F2's penalty) to reflect thinner back-month books.

The skip-roll fires if and only if $\text{edge}_{F_3} > \text{edge}_{F_2}$ AND $\text{edge}_{F_3} > 0$. The absolute-positivity gate prevents us from ever taking the "less bad" cheap option when both are negative; better to default to F2 in that case.

### Burn-in

During the first 30 ticks we accumulate a true running mean and do not update the covariance matrix. This avoids the "first observation equals the EWMA mean" bias that would otherwise leak into downstream residuals.

---

## Part VI: Phase 3 — HMM Regime Detection + Almgren-Chriss Execution

### Objective

Once we know the target tenor, carve the required quantity into child orders whose sizes adapt tick-by-tick to the current LOB regime.

### Hidden Markov Model

We model the LOB state as a latent discrete Markov chain over three regimes:

- **State 0 (Quiet)** — tight spreads, balanced order flow, low intensity.
- **State 1 (Trending)** — directional imbalance, elevated intensity, modest spread widening.
- **State 2 (Distressed)** — wide spreads, extreme imbalance, high intensity; typical of news events or squeezes.

Observation vector per tick: $(\text{spread},\ \text{book\_imbalance},\ \text{trade\_intensity})$, all computed at the L3 data layer.

**Training.** Baum-Welch EM algorithm on historical L3 features, fit offline via `hmmlearn`. We use full covariance matrices per state — the couplings between spread and imbalance in distressed regimes are load-bearing information; a diagonal covariance would throw them away.

**State labeling.** EM assigns hidden states arbitrarily (it has no prior on which state is "Quiet"). We sort the learned states by the **trace of each state's covariance** ascending — more variance = more distress. This mapping is $n$-state agnostic; it works for any integer $K \ge 2$.

**Online inference — the O(1) Forward Algorithm.** The spec explicitly rules out a Viterbi decode on the hot path. Instead, we propagate the forward variable $\alpha_t(i) = P(\text{obs}_{1:t}, S_t = i)$ tick-by-tick:

$$
\alpha_{t+1}(j) \propto b_j(o_{t+1}) \cdot \sum_i \alpha_t(i) A_{ij}
$$

where $b_j$ is the Gaussian emission density and $A$ is the transition matrix. The update cost is $O(K^2)$ per tick — constant in the length of history, unlike Viterbi's $O(KT)$ rolling window.

**Performance tricks.**
- We propagate in **log space** to avoid underflow during extreme anomalies.
- Each state's inverse covariance and log-determinant are precomputed at load time. On the hot path we compute Mahalanobis distances directly, avoiding any scipy call (which is 2+ orders of magnitude slower than a hand-rolled log-density).

The posterior $\alpha$ yields the most likely state via argmax, which is then used as the regime tag for execution.

### Almgren-Chriss trajectory

With a discrete-time Almgren-Chriss formulation under a quadratic impact model, the cost-minimizing trajectory for inventory $X$ over $T$ slices is:

$$
x(t) = X \cdot \frac{\sinh(\kappa(T - t))}{\sinh(\kappa T)}
$$

where $\kappa$ encodes urgency. Higher $\kappa$ front-loads execution (trade faster, eat more impact, reduce variance risk). Lower $\kappa$ flattens toward TWAP (trade slowly, minimize impact, accept variance).

**Regime-to-urgency map:**

| HMM state | Urgency $\kappa$ | Interpretation |
|---|---|---|
| 0 (Quiet) | 0.1 | Near-TWAP, minimal impact |
| 1 (Trending) | 1.5 | Moderately front-loaded |
| 2 (Distressed) | 5.0 | Aggressive front-loading; dump risk |

The $\kappa$ values are initial heuristics; final calibration is by minimizing realized slippage across a historical set of rolls on CL.

**Overflow safety.** For $\kappa T > 50$, `sinh(κT)` overflows in float64. We detect this and fall back to the exponential-form identity $\sinh(a)/\sinh(b) = (e^{a-b} - e^{-a-b}) / (1 - e^{-2b})$, which is numerically stable at any $\kappa T$.

**Caveat: switching $\kappa$.** The closed-form AC solution is derived under constant $\kappa$ over $[0, T]$. We sample $\kappa$ per tick from the HMM. This is a **receding-horizon heuristic** — we are not claiming it is the exact optimum of the time-varying control problem, but it is the spec-mandated approximation and has been shown in the literature (Forsyth et al. 2012) to be near-optimal empirically.

### The Fractional Lot Accumulator

Futures contracts are integers. The continuous sinh difference between consecutive slices is not. If we naively floored the target each tick, we would leak inventory: over $T$ slices, the accumulated floor error can reach several contracts, meaning the roll closes with residual F1 exposure — the entire point of the roll defeated.

**The fix:** carry the fractional residual forward tick-by-tick.

- Tick $t$: desired trade is theoretical + prior fractional remainder.
- Floor to integer; send that order.
- New fractional remainder = desired − floored.

On the **final slice** we flush any remaining inventory (fractional plus any unsent integer) as a single last order. This guarantees the roll closes at exactly the target quantity with zero residual, regardless of the trajectory's shape or the fractional accumulator's history.

**Clipping.** If a mid-trajectory regime flip would cause the theoretical target to *increase* (rare but possible when $\kappa$ drops sharply), we clip the child order size at zero. F-28 never un-trades.

---

## Part VII: Phase 4 — The Totem Protocol (Circuit Breaker)

### Objective

The stochastic models in Phases 1–3 assume price follows a continuous diffusion (broadly, Ornstein-Uhlenbeck-like dynamics). In jump regimes, delivery squeezes, or macro shocks, this assumption fails and all downstream math becomes toxic. Totem is the final, independent, authoritative kill switch.

### Test 1: Realized Variance vs. Bipower Variation

Under a pure continuous diffusion, realized variance (RV) and bipower variation (BPV) converge to the same integrated variance:

$$
RV_n = \sum_{i=1}^n r_i^2 \;\;\overset{p}{\to}\;\; \int_0^T \sigma^2_s\, ds
$$
$$
BPV_n = \frac{\pi}{2}\cdot \frac{n}{n-1} \sum_{i=2}^n |r_i|\,|r_{i-1}| \;\;\overset{p}{\to}\;\; \int_0^T \sigma^2_s\, ds
$$

The $\pi/2$ comes from $E[|Z_1 Z_2|]$ for iid Gaussian $Z$; the $n/(n-1)$ is the Barndorff-Nielsen & Shephard (2004) small-sample correction.

**The key insight**: BPV contains products of *adjacent* absolute returns. Under pure diffusion, adjacent returns cannot both be arbitrarily large (they are both $O(\sqrt{dt})$). Under a jump, a single return can be $O(1)$ but the one adjacent to it is typically normal — so the *jump* contributes to RV but not to BPV. A divergence $RV/BPV \gg 1$ is evidence of a structural jump.

We trip if $RV/BPV > 2.5$.

### Test 2: Hurst Exponent (pinning detection)

The Hurst exponent $H$ characterizes the scaling of variance with aggregation:

$$
\text{Var}(r^{(k)}) = k^{2H}\cdot \text{Var}(r^{(1)})
$$

- $H = 0.5$: pure random walk.
- $H > 0.5$: persistent (trending).
- $H < 0.5$: anti-persistent (mean-reverting).
- $H \to 0$: deterministic pinning.

We estimate $H$ from the variance-scaling identity with **non-overlapping 2-aggregation**:

$$
\hat{H} = \tfrac{1}{2}\log_2\left( \frac{\text{Var}(r^{(2)})}{\text{Var}(r^{(1)})} \right)
$$

If $\hat{H} < 0.15$, the market is deterministically pinned — typical of delivery-algo squeeze days on CL. In this regime there is no meaningful volatility to compensate us for impact, and the PCA residuals are numerically unstable. We halt.

**Non-overlapping vs. overlapping.** Overlapping aggregation biases the estimator because adjacent $r^{(2)}$ values share a component. Non-overlapping is the cleaner (though slightly lower-sample-size) estimator; we prefer it.

### Test 3: Price Floor (CL-specific)

April 20, 2020 — front-month WTI closed at **−$37.63/bbl**. No log-return-based machinery survives a negative or zero price. Beyond the math, a negative price is categorical evidence that the continuous cost-of-carry physics has collapsed.

We trip immediately if the F1 price drops below $5/bbl. This floor is chosen well below any plausible normal-regime low and well above any plausible panic low, so it catches the regime break without triggering on ordinary selloffs.

### The FSM halt

Any of these three triggers causes the Totem to flip `is_halted = True`. The master FSM snaps to `HALTED`, emergency-liquidates remaining inventory as a market order, and refuses to process further ticks until a human calls `totem.manual_reset()` and a new regime burn-in completes.

Critically: **we do not retrain on pre-break data**. A regime break means the steady-state distribution has changed; any HMM or Frank KDE retrained on the old data would inherit the old regime's assumptions.

---

## Part VIII: The Master Controller (FSM)

The `F28Strategy` class is the orchestration layer. It contains almost no math — it routes data between engines per the state machine and enforces the ordering contracts.

### State machine

```
          ┌────────────┐
          │ HOLDING_F1 │  ◄── initial state
          └─────┬──────┘
                │  Frank.death_signal AND (EKF clear OR EKF forces F2)
                ▼
          ┌────────────┐
          │  ROLLING   │  ─── per-tick: Almgren-Chriss child orders
          └─────┬──────┘
                │  f1_inventory <= 0
                ▼
          ┌────────────┐
          │ COMPLETED  │   (terminal for this roll cycle)
          └────────────┘

  ANY STATE ── Totem trips ──► HALTED  (terminal until manual reset)
```

### Tick processing order

1. **Totem first.** Every tick. If the market is broken we stop immediately.
2. **Phase 1 trigger evaluation** if `HOLDING_F1`.
3. **Phase 2.5 physics check** on Frank trigger.
4. **Phase 2 PCA routing** if EKF clears.
5. **Phase 3 execution** if `ROLLING`.

### Dependency injection

Every engine is injected into the master at construction time. The master holds references, not instances. This:

- Enables champion-challenger testing — swap Frank for an alternative signal engine, hold everything else constant.
- Mirrors the C++ Strategy Studio pattern, where the OnQuote master class similarly holds pointers to pre-constructed module objects.
- Allows unit tests to inject mocks for each engine independently.

---

## Part IX: Data Pipeline

### Offline training (`ops/train_f28_models.py`)

Input: cleaned tick DataFrame from the upstream parser. Minimum schema: `Symbol, Datetime, MsgType, BidPrice, BidSize, AskPrice, AskSize, TradePrice, TradeSize`. Upstream data engineering (PCAP parsing, sequence reconstruction, gap detection) is out of scope for this module per user's architectural decision.

Outputs (serialized to JSON for cross-language ingestion):

1. **Frank baseline.** Raw log-return sample from steady-state training days. We ship the sample, not a pre-evaluated PDF, because KL requires a shared support that can only be determined at inference time.
2. **HMM matrices.** Transition matrix, per-state means and covariances, start probabilities.

### Live loading (`main.py`)

At boot:
1. Read `./models/f28_parameters.json`.
2. Construct all engines.
3. Hydrate Frank via `frank.load_baseline(returns_sample)`.
4. Hydrate HMM via `hmm.load_from_params(hmm_matrices)` — this also precomputes inverse covariances and log-normalizers for the hot path.
5. Attach strategy to `TickEngine`; begin replay or stream.

### Streaming interface

The `TickEngine` exposes `run_stream(iterable_of_dicts)` — the production entry point. Any upstream tick producer that yields dicts in the tick-contract schema can drive the strategy. This cleanly decouples F-28 from its data source, which is the user's architectural preference.

---

## Part X: Design Decisions and Trade-offs

| Decision | Alternative | Why this |
|---|---|---|
| KL against offline baseline | Live-window self-split | Self-split is ≈0 by construction; can never detect a genuine distribution shift |
| AND gate on death signal | OR gate | OR fires on garden-variety volume or return bursts; AND requires joint regime departure |
| Lee-Ready sign inheritance | 50/50 volume split on zero ticks | 50/50 systematically dilutes imbalance; inheritance is the Easley-LdP recommendation |
| EWMA covariance | Rolling-window covariance | EWMA decays smoothly; rolling-window has artificial cliff at boundary |
| O(1) Forward algorithm | Viterbi over rolling window | Viterbi is $O(KT)$ per tick; Forward is $O(K^2)$. On a 1ms budget this matters |
| Full-covariance HMM | Diagonal-covariance HMM | Feature couplings in distressed regimes are load-bearing; diagonal throws them away |
| Exact OU transition moments | Euler small-dt approximation | Robust to asynchronous ticks and any sampling rate |
| F1 as spot, F2 as EKF observation | Use published Cushing cash | Cash not available tick-by-tick; F1 is the best continuous proxy |
| Non-overlapping 2-aggregation for Hurst | Overlapping aggregation | Overlapping biases the estimator; non-overlapping is cleaner |
| BPV with $n/(n-1)$ correction | Raw BPV | Matters for short windows; BN-S 2004 recommendation |
| Price floor in Totem | No floor | CL can go negative (Apr 2020); log-return math collapses; also semantically meaningful |
| Calendar years for $\tau$ | Trading years | CL trades nearly continuously; storage accrues on calendar time |
| SOFR 3M / T-Bill 3M | DGS10 | Front-back basis has $\tau \approx 1$ month; 10Y adds term-premium noise |
| Fractional-lot flush on final slice | Drop final fractional | Whole purpose of the accumulator is exact quantity close |

---

## Part XI: Known Limitations

1. **HMM stationarity assumption.** Baum-Welch assumes the transition matrix and emission parameters are constant across the training window. In practice, CL's microstructure has changed meaningfully (2014–2016 shale boom, 2020 pandemic, 2022 Russia-Ukraine). Monthly retraining partially addresses this; a truly non-stationary HMM (e.g. Hierarchical Dirichlet Process HMM) is future work.

2. **PCA linearity.** The PCA model assumes curve returns decompose into linear factors. In supply-shock regimes this factor structure breaks down; Phase 2.5 is our specific defense against this but it cannot catch slow regime drift.

3. **Almgren-Chriss under switching $\kappa$.** As noted, the closed-form AC trajectory is derived under constant $\kappa$. Our per-tick $\kappa$ sampling from the HMM is a receding-horizon heuristic, not the exact optimum of the non-stationary problem.

4. **EKF Gaussianization.** The Kalman-family filter linearizes the observation function and represents the posterior as Gaussian. In tail events the posterior is visibly non-Gaussian; we rely on Totem to halt before we reach those tails.

5. **VPIN bucket stationarity.** VPIN buckets are equal-volume, but the information content per bucket is not equal across regimes. A distressed bucket of 1000 contracts is a different object than a quiet bucket of 1000 contracts. Volume-clock scaling mitigates but does not eliminate this.

6. **Training-live distribution shift.** If the live market state is permanently out-of-sample relative to training, Frank's KL will fire continuously and the HMM's forward posterior will concentrate on its nearest available state. Totem will typically catch this eventually, but there is a window in which the strategy may misfire.

7. **Single-venue assumption.** F-28 currently assumes a single primary exchange (NYMEX). Cross-venue routing and latency arbitrage are out of scope.

---

## Part XII: Future Research Directions

1. **Two-factor Schwartz model.** Upgrade the EKF to jointly estimate spot mean reversion and convenience yield. This requires a particle filter or UKF because the state space is no longer well-approximated by scalar linearization.

2. **Regime-conditional PCA.** Fit separate EWMA covariance matrices per HMM regime. Conditional on distressed regime, the curve factor structure is known to differ; a single unconditional PCA averages across regimes and loses precision.

3. **Optimal trade-off between Frank triggers.** The AND threshold is heuristic. A Bayesian decision-theoretic formulation — maximize expected roll P&L minus expected crisis loss, integrated over a posterior over regime — would yield principled thresholds.

4. **Multi-asset extension.** The architecture generalizes to any physical-commodity future (NG, HO, RB). Each needs its own OU calibration and supply-shock semantics, but the structural code does not change.

5. **Adversarial robustness.** VPIN is known to be spoofable by sophisticated market-makers. Joint filtering with resting-order-age information (which spoofs cannot falsify) would harden the signal.

6. **Continuous-time relaxation of Almgren-Chriss.** The discrete-time sinh solution is an approximation of the continuous-time optimal trajectory under square-root impact. Direct numerical solution of the HJB equation with regime-switching dynamics is more faithful but computationally heavier.

---

## Appendix A: File Ontology

```
f28/
├── alpha/                       -- Phase 2 and 2.5
│   ├── curve_geometry.py        -- PCAModel (EWMA cov, stitch, residuals, routing)
│   └── ekf_overlay.py           -- ConvenienceYieldEKF (OU, exact transition moments)
├── engine/                      -- Low-level utilities and backtest harness
│   └── backtester.py            -- TickEngine (stream + CSV replay)
├── execution/                   -- Phase 3
│   ├── hmm_regime.py            -- MicrostructureHMM (Baum-Welch offline, O(1) Forward online)
│   └── almgren_chriss.py        -- ExecutionEngine (sinh trajectory, fractional accumulator)
├── ops/                         -- Offline ETL and training
│   └── train_f28_models.py      -- StrategyStudioETL (parses cleaned frame, serializes to JSON)
├── risk/                        -- Phase 4
│   └── totem_protocol.py        -- TotemCircuitBreaker (RV/BPV, Hurst, price floor)
├── signals/                     -- Phase 1
│   ├── base_signal.py           -- ABC for champion-challenger
│   └── frank_module.py          -- FrankSignalEngine (VPIN + KL)
├── strategy/                    -- FSM master
│   └── f28_master.py            -- F28Strategy (HOLDING_F1 → ROLLING → COMPLETED, HALTED)
├── main.py                      -- Entry point + dependency injection graph
└── F28_STRATEGY.md              -- This document
```

## Appendix B: Tick Contract

```python
tick_data = {
    "timestamp":      datetime,      # tick time
    "f1_price":       float,         # F1 last price (also serves as spot proxy)
    "f1_vol":         int,           # executed volume on this tick
    "f1_symbol":      str,           # e.g. 'CLM25'
    "f1_expiry":      datetime,      # F1 physical expiry
    "f2_expiry":      datetime,      # F2 physical expiry
    "curve_prices":   np.ndarray,    # shape (5,): [F1, F2, F3, F4, F5]
    "curve_spreads":  np.ndarray,    # shape (5,): bid-ask per tenor
    "curve_symbols":  tuple[str,...],# shape (5,): symbol per slot
    "l3_features":    np.ndarray,    # shape (3,): [Spread, Imbalance, Intensity]
    "risk_free_rate": float,         # SOFR 3M or 3M T-Bill, continuously compounded
}
```

## Appendix C: Key Hyperparameters (CL calibration)

| Parameter | Value | Source | Notes |
|---|---|---|---|
| `vpin_threshold` | 0.75 | Literature (Easley et al. 2012) | Tune on historical rolls |
| `entropy_limit` | 2.0 | Empirical target | Set so signal fires ~once per roll cycle in training |
| `bucket_volume` | 1000 | CL-specific | 10–50x smaller than ES would need |
| `ewma_span` | 60 | Curve-analytics convention | ~1 trading hour at 1s cadence |
| `ekf.kappa` | 1.5 | Placeholder — MLE on WTI convenience yield | Moderate reversion |
| `ekf.theta` | 0.03 | Placeholder | ~3% long-run convenience yield |
| `ekf.sigma_y` | 0.25 | Placeholder | CL y is genuinely volatile |
| `ekf.physical_limit` | 0.15 | Spec + CL economics | 15% y = real supply shock |
| `hmm.n_states` | 3 | Spec | Quiet / Trending / Distressed |
| `ac.total_time_steps` | 20 | Tunable | Controls roll duration |
| `ac.kappa_map` | {0.1, 1.5, 5.0} | Heuristic | Calibrate on realized slippage |
| `totem.jump_threshold` | 2.5 | BPV literature | RV/BPV > 2.5 = jump |
| `totem.hurst_limit` | 0.15 | Spec | Pinning |
| `totem.price_floor` | 5.0 | CL-specific | Post-Apr-2020 safety |

---

*End of document.*
