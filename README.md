# qkd-sdn-sim — phases 2-8

**To reproduce the paper, read [REPRODUCE.md](REPRODUCE.md).** This file is the
development log for the eight build phases and is kept for provenance.

Related work and the citations behind every parameter: [`docs/related_work.md`](docs/related_work.md).

A discrete, fixed-step simulator of a QKD network with trusted relays: key
generation per link, Poisson session demand, SDN-style k-shortest-path routing,
admission control and key consumption in both KM-OTP and KM-AES modes.

- **Phase 2** — the core simulator: baseline **B0**, no attack, no detection, no policy.
- **Phase 3** — compromised relays (profiles P/G/L/E), the observation noise model, and damage accounting (`D_eff`, `D_raw`).
- **Phase 5-6**

| # | criterion | status |
|---|---|---|
| 1 | five competing policies B0-B4 implemented | done |
| 2 | B0 numerically identical to the phase 4 path | `test_b0_is_numerically_identical_to_no_policy` |
| 3 | the oracle isolates the whole compromised set | DRR_relay 0.988, `test_oracle_removes_almost_all_relay_damage`. NOT a bound on damage prevented: it never throttles, so a throttling policy can pass it by admitting less traffic. See REPRODUCE.md. |
| 4 | B2 relays XOR shares over node-disjoint legs | exhaustive truth table, `tests/test_multipath.py` |
| 5 | the policy never sees ground truth except in B4 | `test_policy_isolation` |
| 6 | DRR / PSI / C joined to matched baselines | `scripts/analyze.py`, hash join on `scenario_id` |
| 7 | Pareto frontier over the operating points | `--pareto`, non-dominated set marked |
| 8 | the graded policy's benefit is targeting, not volume | +0.9 over the blind control at matched cost on L/E; ~+0.08 on P |
| 9 | a volume control exists so the two can be told apart | `policy.type: BT` |
| 10 | the comparison survives an adversary who picks the intensity | minimax table; B1 worst case 0.00, B3 0.23-0.41 |
| 11 | connectivity is an axis, not three anecdotes | 12 calibrated members, degree 2.2-4.4 |
| 12 | the drain/evict modelling choice is measured, not assumed | worth +0.003 at 82 s and +0.125 at 5400 s |
| 13 | compared against published methods, not only our own ablations | B5/B6/B7, `config/sweep_baselines.yaml` |
| 14 | each baseline reported at its OWN best operating point | `beta_trust`, `m_paths`, `alpha_key` swept |
| 15 | every physical constant derived or cited | parameter table; `L_0` derived from alpha = 0.2 dB/km |

**Phase 4** — the three detection features, the posterior risk score `S`, and an evaluation harness independent of any policy.
- **Phase 5** — the five response policies B0-B4, including the graded proposal and the oracle upper bound.
- **Phase 6** — the experiment matrix, the DRR/PSI/C metrics and the security-vs-cost Pareto frontier.

Phases 7 (analytical model validation) and 8 (weight sensitivity) are planned
and not yet built.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows
python scripts/build_topologies.py                 # writes data/topologies/*.json
python -m sim.runner --config config/base_t1_otp.yaml
python -m pytest -q
```

---

## Layout

```
sim/
  config.py      dataclasses + YAML loading + deterministic run_id (sha1 of sorted JSON)
  topology.py    graph loading, stable edge indexing, R_e = R_max*exp(-L_e/L_0)
  state.py       NetworkState, Session, runtime invariants
  keygen.py      vectorised buffer update, O(|E|) per step
  demand.py      Poisson arrivals, admission, consumption, AES pulses, starvation
  routing.py     lazy k-shortest-path table with staleness-gated rebuild
  interfaces.py  Attack / Detector / Policy ABCs + Null implementations + factories
  metrics.py     accumulators, warm-up gating, node telemetry
  logging_io.py  parquet writing (summary + events)
  runner.py      the main loop
  sweep.py       cartesian expansion, parallel + resumable execution
  calibrate.py   lambda calibration and Phi_i (baseline key flow)
  noise.py       observation noise N1-N3 (phase 3)
  attack.py      compromised relays, profiles P/G/L/E (phase 3)
  telemetry.py   raw features x1, x2, x3 - reads ONLY controller-visible fields
  detector.py    robust normalisation, prior, score, EWMA, 7-way ablation
  evaluate.py    ROC/AUC, detection delay, FPR; in-process and offline CLI
  policy.py      B0-B4, hysteresis + dwell + partition guard (phase 5)
config/          base_t1_otp, base_t2_otp, base_t3_otp, base_t3_aes, sweep_calibration
data/topologies/ nsfnet.json (T1), usnet.json (T2), net50.json (T3)
scripts/         build_topologies.py, aggregate.py, report_phase34.py, analyze.py
tests/           invariants, determinism, routing, sessions
results/raw/     {run_id}.parquet and {run_id}_events.parquet
```

---

## Design decisions worth knowing

### Aggregate consumption rate (mandatory, section 3 of the build spec)
Each edge carries `edge_rate`, updated **only** when a session opens or closes.
A step is then one vector operation, `buffers -= edge_rate*dt`, i.e. **O(|E|)**
instead of O(sessions × hops). `state.recompute_edge_rate_from_sessions` is the
reference implementation used by the deep invariant (every 1000 steps) and by
`test_edge_rate_consistency`.

### AES re-key pulses
In AES `edge_rate` stays zero; consumption is a min-heap of 256-bit pulses at
`t_start + k*T_rk`, exactly `ceil(T_s/T_rk)` of them. A pulse is all-or-nothing:
if any hop cannot pay it, that session could not re-key and is interrupted.

### Starvation, LIFO
When a buffer goes negative it is clamped to zero, the overdraw is clawed back
out of the consumption accounting, and sessions on that link are torn down
**newest first** while `edge_rate[e] > R[e]`. Older sessions, which have more
invested in them, survive. If the aggregate rate is already sustainable, the
buffer recovers by itself on the next step and nothing is cut — so a link can
never stay starved forever. *Mention the LIFO choice in the paper.*

### Session timing
A session admitted at `t0` with duration `T_s` consumes on the steps
`t0+dt … t0+T_s`, i.e. exactly `T_s/dt` steps, and retires at the end of the
step at `t0+T_s`.

### Seed streams
Four independent `np.random.default_rng` streams: topology, demand, attack,
policy. This is a methodological requirement, not a nicety — if they shared a
stream, changing `λ` would move the compromised-node selection and contaminate
every cross-scenario comparison. `test_seed_isolation` guards it with a probe
attack that records everything it draws.

### Reproducibility
`run_id` is a sha1 over the canonically sorted JSON of the whole config, minus
`output_dir` — never the salted builtin `hash()`. Wall-clock timings are the only
non-deterministic numbers a run produces, and they are kept **out** of the
parquet (they stay in the returned summary dict), so re-running with the same
seed reproduces the output files byte for byte.

---

## Parameters, and where each number comes from

Every constant in the model is either derived from physics, taken from a
deployed network, or declared free and swept. Nothing is a round number chosen
because it looked reasonable.

| symbol | value | where it comes from |
|---|---|---|
| `α` fibre attenuation | 0.2 dB/km | standard 1550 nm SMF; the figure the QKD networking literature assumes throughout |
| `L_0` key-rate decay | **21.71 km** | **derived, not chosen**: `L_0 = 10/(α ln10)` — see below |
| link lengths | 8.7 – 52 km | the deployed metro regime: Tokyo QKD 1–45 km, Hefei 46-node metro, MadQCI Madrid |
| `R_max` | 20 kbit/s | free; see the invariance argument below |
| `B_max` buffer | 10 Mbit | free, with `R_max`; a session reserves ~1.7 % of it |
| session rate | 10 – 1000 bit/s | free, with `R_max` |
| topology scale | 14 / 24 / 50 nodes | NSFNET-14 and a 24-node US mesh for connectivity comparability; 50 nodes because Hefei is 46 |
| `f` compromised | 0.05 – 0.30 | swept; Luo & Li sweep 0.10–0.30 on 50 nodes |
| QBER base | 1.5 – 3.0 % | per-link draw, typical decoy-BB84 operating range |
| `T_rk` AES re-key | 60 s | free; swept implicitly through the OTP/AES contrast |
| detector `α`, weights, `λ_s`, `θ₀` | — | `α` swept; weights swept (phase 8); `(λ_s, θ₀)` **solved** per topology by `calibrate.calibrate_score` |
| `β_trust` (B5), `α_key` (B7) | swept | the source papers give no numeric values |

### `L_0` is not a free parameter

Secret key rate scales with channel transmittance `η = 10^(−αL/10)`, so

```
exp(−L/L_0) = 10^(−αL/10)   ⟹   L_0 = 10 / (α · ln 10) = 21.71 km  at α = 0.2 dB/km
```

The phase 1 spec's `L_0 = 50 km` would require **α = 0.087 dB/km**, below the
Rayleigh scattering limit of silica — a fibre that does not exist. This was
found while grounding the parameters and is corrected;
`test_L0_is_the_value_the_fibre_attenuation_implies` pins it.

The correction was **free**, and that is worth stating because it is unusual.
Link length enters the model *only* through `key_rate()` — `qber_base` is an
independent per-edge draw — so scaling every length and `L_0` by the same factor
leaves every `R_e` exactly unchanged. Taking `k = 21.71/50` moved the links from
[20, 120] km to [8.7, 52] km without moving a single result, and it moved them
*into* the deployed metro regime rather than out of it. Verified to 2.3 × 10⁻⁸
relative at regeneration, and pinned by
`test_length_rescaling_is_exactly_rate_preserving`.

### Why the absolute key rate does not need defending

`R_max = 20 kbit/s` is well below a current commercial system — Toshiba's LD
product is ~300 kb/s at 10 dB, and the Tokyo field trial sustained 300 kbit/s
over 45 km — and comfortably above others in the same field trials, such as
Tokyo's entanglement link at 0.25 kbit/s. The deployed range spans three orders
of magnitude at comparable distance, so no single value is "realistic".

It does not matter, and this is checkable rather than arguable. **Every metric
this paper reports is dimensionless (rejection rate, AUC, hop count, buffer
utilisation) or a ratio of two damages (DRR).** Scaling the key supply and the
key demand together by 25× leaves all of them *bit-identical* and scales only
the extensive quantities, exactly:

| | `RR` | `n_admitted` | `mean_leg_len` | `buffer_util` | `AUC` | `D_eff_relay` | `KPD` |
|---|---|---|---|---|---|---|---|
| ×25 | identical | identical | identical | identical | identical | ×25.000 | ×25.000 |

So `R_max = 20 kbit/s` with 10–1000 bit/s sessions and `R_max = 500 kbit/s` with
250 bit/s – 25 kbit/s sessions are the same experiment. The only modelling
choice is the **ratio** of key supply to key demand, which sets how binding the
key constraint is, and that is stated and swept through the load axis.
`test_only_the_ratio_of_key_supply_to_demand_matters` pins it.

## Two places where this build departs from the written spec

Both are numerical inconsistencies between `phase1-model-spec` and
`phase2-build-spec`; both are isolated in data/config files, so reverting them is
a one-line edit. **Both must be stated in the method section of the paper.**

### 1. Link lengths are rescaled into the deployed metro range

`R_e = R_max·exp(-L_e/L_0)` with `R_max = 20000 bit/s`, `L_0 = 50 km` and the
*geographic* span of NSFNET/USNET (600–2800 km) gives `exp(-22) ≈ 1e-10 bit/s`:
no usable key anywhere. Trusted-relay QKD networks exist precisely because
individual QKD links are short.

The reference topologies are therefore used for their **connectivity** (literature
comparability), while link lengths are linearly rescaled from the geographic
distances into **[20, 120] km**, order preserving. That yields
`R_e ∈ [1814, 13406] bit/s` and `B_max/R_e ∈ [746, 5512] s` — a range in which
buffers, admission and starvation are all meaningful. See
`scripts/build_topologies.py`.

The adjacency lists there follow commonly used variants of NSFNET-14 (14/21) and
a 24-node US mesh (24/43). They are plain JSON, so substituting the exact edge
list of whichever paper you cite is a one-file edit, no code change. **Check
them against your intended citation before publication.** `net50` (50/90) is
generated deterministically from `seed_topology`: Euclidean MST plus shortest
fill, which guarantees connectivity and a roughly planar mesh.

### 2. OTP session data rate

`phase2-build-spec` sets `rate_min=1e3, rate_max=1e5 bit/s`. With
`R_e ≤ 13.4 kbit/s` and OTP's one-key-bit-per-data-bit-per-hop, a single average
session (≈50 kbit/s over ≈5 hops) consumes more than the **entire** network
generates, so no session ever completes and the results are degenerate.

The shipped configs use `rate_min=10, rate_max=1000 bit/s`, which keeps a typical
session at ≈1.7 % of `B_max` and gives a load range where λ can actually be
calibrated to 1 % / 10 % / 30 % rejection. The dataclass defaults in
`sim/config.py` are still the literal spec values; the override lives in
`config/base_*_otp.yaml` with a comment. This is the physically correct fix —
OTP over QKD only supports low-rate traffic — but it is a change, so it is
flagged rather than hidden.

---

## Calibration results (6 h horizon, 30 min warm-up, seed_demand=2)

Recalibrated with the N4 background stream active (from phase 3 on it is part of
the model and carries ~10 % of the offered load) and on the 2-connected `net50`,
whose shorter paths raised T3's capacity by roughly 4x.

| config | low (RR≈1 %) | medium (RR≈10 %) | saturated (RR≈30 %) |
|---|---|---|---|
| T1 / OTP | λ = 0.1750 (1.1 %) | λ = 0.3999 (10.2 %) | λ = 0.900 (32.0 %) |
| T2 / OTP | λ = 0.1380 (1.0 %) | λ = 0.3045 (10.2 %) | λ = 0.914 (29.4 %) |
| T3 / OTP | λ = 0.1384 (0.9 %) | λ = 0.2284 (9.3 %) | λ = 0.554 (30.8 %) |
| T3 / AES | λ = 16.03 (1.0 %) | λ = 21.71 (10.7 %) | λ = 43.42 (30.8 %) |

`calibrate.py` also solves the score operating point `(λ_s, θ₀)` in closed form
and records it in the same file; see "Score operating point" below.

The `lam` in each `config/base_*.yaml` is set to that row's **medium** load.

Regenerate with:

```bash
python -m sim.calibrate --config config/base_t3_otp.yaml --horizon 21600
```

which writes `config/calibration_{topology}_{km_mode}.yaml` containing the three
λ values, the achieved rejection rates and `Φ_i` (min-max normalised key flow per
node — the prior component of phase 1 section 7, deliberately *not* degree
centrality).

In **AES** the buffer is a far weaker bottleneck by construction: one 256-bit
pulse per hop every 60 s is ≈4.3 bit/s against `R_e ≥ 1814 bit/s`, so rejection
only appears at arrival rates one to two orders of magnitude higher than in OTP.
`calibrate.py` searches up to `LAM_CAP = 200` demands/s and reports explicitly if
a target is unreachable rather than silently returning the cap.

---

## Sweeps

```bash
python -m sim.sweep --sweep config/sweep_calibration.yaml --no-assert --dry-run
python -m sim.sweep --sweep config/sweep_calibration.yaml --no-assert
python scripts/aggregate.py --out results/summary.parquet --csv results/summary.csv
```

Workers are fully independent, each writing its own parquet; a run whose output
already exists is skipped, so a sweep is interruptible and resumable. A failing
run is logged and never takes the sweep down. `max_workers` defaults to
`cpu_count()-1`.

---

## Phases 3 and 4

```bash
python -m sim.runner --config config/attack_t1_otp.yaml --set attack.profile=G --ablation
python -m sim.sweep  --sweep config/sweep_ablation.yaml --no-assert
python scripts/report_phase34.py --ablation --damage --operating
```

12 h horizon, `t_c = 10800 s`. The detector samples from the end of the metric
warm-up so its robust baseline is built on steady-state behaviour, not on the
initial transient; the 3600 s base window fills at 5370 s, i.e. 1.5 h before the
attack starts, leaving 9 h of attack for the ROC.

### The noise model is load-bearing

> In a noiseless simulator every detector is perfect.

Four independent sources, all scaled together by `noise.scale` (the sweep axis):

| | source | default | why it has to exist |
|---|---|---|---|
| N1 | relative QBER measurement noise | 0.10 | |
| N2 | key rate jitter | 0.05 | |
| N3 | buffer report noise, **per link side** | 0.02 | without it `x2` is identically zero for every healthy node |
| N4 | consumption invisible to the ledger | 0.10 | **the only source of `x1` residual on a healthy node** — remove it and F1 scores AUC 1.0 against profile G |

N4 is a real traffic stream (`managed=False`): routed like any other session,
consuming real key, absent from the ledger and from every efficiency metric.
A fifth, free source — sessions straddling a window boundary — needs no code but
should be named in the analysis.

The observation noise draws from its **own** stream (`seed_noise`). Sharing the
attack stream would make the noise realisation move when the profile changes and
would contaminate every profile-to-profile comparison.

### Profile P is undetectable by construction

Not empirically, structurally: with no policy feedback a passive compromise
cannot touch the dynamics, so a profile-P run is bit-identical to a no-attack
run. `test_passive_profile_changes_nothing` asserts that observable by
observable, which is stronger than the KS test the spec asks for. An AUC near
0.5 on P is the correct result.

The four profiles are also **paired**: `V_c` is drawn before any profile-specific
random draw, so the same `seed_attack` compromises the same nodes in P, G, L and E.

### Damage accounting

Three running aggregates updated on three events only — session open, session
close, and one reclassification pass at `t_c` — so the per-step cost stays O(1)
and the phase 2 architecture survives. Counted **once per session** however many
compromised nodes its path crosses: it is the same key material and it is not
disclosed twice. In AES the charge lands at the re-key pulse, so key relayed
before `t_c` is not exposed even when the traffic it protects continues past it —
a real OTP/AES difference worth stating in the paper.

`D_raw_hops` is also recorded, summing key over hops incident to a compromised
node, because the spec's AES snippet reads that way while its "count once" rule
reads the other. Both are available; the paper should say which it uses.

### Detector isolation

`telemetry.py` may read only `qber_obs`, `R_obs`, `buffer_reported`,
`ledger_consumed` and the topology. `test_detector_isolation` overwrites every
privileged field with NaN and asserts the features come out unchanged.

---

## Three corrections the features needed to work at all

Each was found by the ablation gate failing, and each is a substantive change to
the written formula. **All three belong in the method section.**

### 1. `x1` must account for buffer saturation

A link sitting at `B_max` discards the key it generates, so the naive
`R·W − ΔB` reads the whole discarded amount as consumption and every idle-but-full
link becomes a permanent false positive. The controller knows `B_max` and knows
what it authorised, so it can bound the key the link could actually have stored:

```
G_hat = min(R_obs·W, headroom_at_window_start + K_exp)
K_obs = G_hat − (B_now − B_past)
```

On a saturated link this collapses to a residual of zero — the honest answer:
**key stolen from a link that refills to the brim leaves no trace in the buffer
level.** F1 is observable only where links run near key capacity. That is a real
property of the mechanism, and it is why `x1` is strong on T1 (AUC 0.73) and weak
on T3 at the RR-calibrated load (0.55): T3 runs at ~9 % key utilisation with
`mean_buffer_util` 0.97. Raising the offered load recovers it monotonically —
measured AUC 0.56 → 0.64 → 0.74 → 0.81 as λ goes 0.056 → 0.2 → 0.6 → 1.5 — and so
does targeting the high-flow nodes (`selection=top_keyflow`: 0.80).

### 2. `x1` is a standardised residual, not a relative one

The spec writes `x1 = |K_obs − K_exp| / (K_exp + ε)`. That diverges on an idle
node (pure report noise over ε) and shrinks on a busy one, which **inverts** the
ranking — measured at AUC 0.22 for profile G on T3. Dividing by the residual the
controller should *expect* instead makes `x1` a signal-to-noise ratio: ≈1 wherever
nothing is wrong regardless of traffic, and >1 only on genuine excess.

```
sigma = sqrt( sum_adj (report_sigma·B_reported·sqrt2)^2 + (unmanaged_assumed·K_exp)^2 )
x1 = |K_obs − K_exp| / sigma
```

`report_sigma` and `unmanaged_assumed` are **operator-calibrated constants in the
detector config**, deliberately separate from the simulator's true noise settings,
so a later sweep can ask what happens when the operator's assumption is wrong.

### 3. `x3` needs the dual baseline at link level too

Section 4.2 defines the quantum-layer deviation against a per-link moving average
alone. With a persistent tap that average absorbs the attack within `W_ema` and
the evidence disappears for the rest of the run — measured AUC 0.55 for profile E,
the signal surviving about a tenth of the post-compromise window. Applying the
same `max(temporal, peer)` construction section 6 already prescribes one level up,
but at the link level, keeps a permanent tap permanently visible: **AUC 0.95**.

Both branches are standardised (z against the link's own measurement spread, or
against the cohort), so `b1` and `b2` combine like quantities. The SKR peer
comparison runs on `R_obs / R_nominal`, since link rates differ by an order of
magnitude across lengths and the nominal rate is commissioning data the operator
has anyway.

---

## Ablation: the phase 4 gate

`use_prior=False`, 5 attack seeds, `selection=random`. **Bold = the feature that
profile is designed to trigger.**

**T1 / NSFNET**

| features | P | G | L | E |
|---|---|---|---|---|
| {x1} | 0.472 | **0.727** | 0.585 | 0.474 |
| {x2} | 0.560 | 0.482 | **0.972** | 0.500 |
| {x3} | 0.574 | 0.574 | 0.565 | **0.943** |
| {x1,x2,x3} | 0.547 | 0.790 | 0.963 | 0.938 |

**T3 / net50**

| features | P | G | L | E |
|---|---|---|---|---|
| {x1} | 0.480 | 0.553 | 0.422 | 0.480 |
| {x2} | 0.450 | 0.438 | **0.974** | 0.443 |
| {x3} | 0.472 | 0.472 | 0.470 | **0.954** |
| {x1,x2,x3} | 0.439 | 0.497 | 0.965 | 0.954 |

The diagonal is clean, P sits at chance everywhere, and nothing reaches 1.0 —
the three health criteria of section 0. `x1` on T3 is the load-limited case
described above; under `selection=top_keyflow` it recovers to 0.796.

### The prior is a confound in both directions, and must never be averaged

| block | profile P | what it measures |
|---|---|---|
| random, prior off | ≈0.47 | feature quality — **read the ablation here** |
| random, prior on | ≈0.46 | honest end-to-end operating figure; the prior is uncorrelated with `V_c` here, so it dilutes a weak feature rather than helping |
| top_keyflow, prior off | ≈0.45 | feature quality on high-flow nodes |
| top_keyflow, prior on | **≈0.92** | the prior alone. The compromised nodes *are* the high-Φ nodes, so it identifies them with no evidence whatsoever |

That last row is the bias the spec warns about in section 4.4, now measured: on
the undetectable control profile the prior still scores 0.92.
`scripts/report_phase34.py` prints all four blocks separately and never averages
them.

---

## Phases 5 and 6 — response policies and the experiment matrix

```bash
python -m sim.runner --config config/attack_t1_otp.yaml --set policy.type=B3
python -m sim.sweep  --sweep config/sweep_policy_slice.yaml --no-assert
python scripts/analyze.py --drr --pareto --cost
```

Five policies, one config section, one swept axis — so every run in a scenario
hashes identically outside `policy.*`, which is what makes the DRR baseline join
exact.

| | policy | levers |
|---|---|---|
| B0 | no defence | — |
| B1 | binary | isolate at fixed `τ` on the same `S̄` |
| B2 | static multipath | XOR key shares over `m` **node-disjoint** legs, no detection |
| B3 | graded (the proposal) | `w = exp(κS̄)`, `ρ` ramp, isolation at `S_iso` |
| B4 | oracle | perfect knowledge of `V_c` |
| BT | blind throttle | uniform `ρ`, no detection — the volume control, not one of the five |

B1 gets the **same** hysteresis band and dwell timer as B3 on purpose. The
interesting claim is that a graded response beats a well-implemented binary one,
not that it beats a flapping one.

### Headline result — T3 / net50, f = 0.10, 20 seeds, paired DRR with bootstrap CI

`DRR_relay = 1 - D_eff_relay(policy) / D_eff_relay(B0)`, each seed contributing
its own ratio.

| profile | detector | B1 (tau=0.5) | B3 (rho0=0.26) | B2 (m=2) | B4 oracle |
|---|---|---|---|---|---|
| **P** passive | blind by design | 0.001 | ~0 | 0.852 | 0.981 |
| **G** greedy | AUC ~0.68 | 0.040 | ~0 | 0.841 | 0.977 |
| **L** liar | AUC ~0.97 | 0.915 | **0.980** | 0.852 | 0.980 |
| **E** tap | AUC ~0.96 | 0.755 | **0.965** | 0.848 | 0.978 |

And the cost, same cell, as extra rejection / extra key per demand:

| policy | ΔRR | ΔKPD | mean hops | mean legs |
|---|---|---|---|---|
| B0 | — | — | 5.87 | 1.00 |
| B1 | +0.12 | +0.02 | 6.16 | 1.00 |
| B3 (rho0=0.26) | +0.29 | +0.02 | 4.80 | 1.00 |
| B2 (m=2) | +0.36 | **+0.59** | 15.70 | 2.50 |
| B4 oracle | +0.12 | −0.01 | 6.25 | 1.00 |

**Where the detector sees the attack, the graded policy dominates blind
redundancy on every axis at once** - more protection (0.965 vs 0.848 on E), less
rejection (+0.29 vs +0.36) and far less key (+0.02 vs +0.59). Where it is blind
(P, G) the ordering reverses and B2 is the only thing that helps, because it
never needed to detect anything. That is a clean division of labour and it is the
result the paper is for.

**B4 validates the whole chain.** The oracle removes 97.7-98.1 % of relay
exposure; the residual is the pre-`t_c` window plus whatever the partition guard
vetoes. Anything much below that would have meant a broken accounting path.

Behaviour against the compromise ratio `f` (0.05 → 0.30) is monotone and
interpretable: detection-driven policies degrade as more nodes are compromised
(B3 on E: 0.976 → 0.979 → 0.953 → 0.435) because the partition guard runs out of
room, while B2 degrades for a different reason - with more compromised nodes the
chance that *all* m legs are compromised rises (0.907 → 0.848 → 0.721 → 0.713).

### The volume control — how much of DRR is targeting at all?

Any throttling policy reduces damage two ways at once, and they must not be
reported as one number. Admitting fewer sessions cuts exposure **mechanically**,
whether or not the throttle is aimed at anything: a policy that rejects half the
traffic roughly halves the damage while targeting nothing. So a DRR of 0.76
bought with a 54-point rise in rejection is not 0.76 of detection value.

`policy.type: BT` is the null hypothesis — a uniform quota on every node, no
detection at all. At a matched rejection rate, the targeting benefit of a policy
is its DRR *minus BT's*.

T3, f = 0.10, 20 seeds. `DRR_relay / ΔRR`:

| policy | P | G | L | E |
|---|---|---|---|---|
| BT ρ=0.90 | 0.094 / +0.20 | 0.060 / +0.18 | 0.094 / +0.20 | 0.077 / +0.20 |
| **B3 κ=3 ρ₀=0.26** | −0.076 / +0.04 | 0.001 / +0.05 | **0.980 / +0.40** | **0.965 / +0.29** |
| BT ρ=0.75 | 0.579 / +0.49 | 0.526 / +0.45 | 0.579 / +0.49 | 0.548 / +0.48 |
| B3 κ=3 ρ₀=0 | 0.757 / +0.54 | 0.717 / +0.50 | 0.995 / +0.63 | 0.992 / +0.59 |
| BT ρ=0.60 | 0.831 / +0.65 | 0.810 / +0.61 | 0.831 / +0.65 | 0.819 / +0.64 |

Three things follow, and two of them are corrections to the obvious reading:

**Evidence-driven throttling is genuinely targeted, and by a wide margin.** B3 at
`ρ₀ = 0.26` reaches 0.98 (L) and 0.965 (E) for ΔRR ≈ +0.3, while the blind
control at the *same* cost reaches 0.08–0.09. That is an order of magnitude, and
it is the paper's real result. On P and G it correctly returns ≈ 0: it does not
pretend to protect against what it cannot see.

**The prior-driven variant's headline number is mostly volume.** B3 at `ρ₀ = 0`
scores 0.757 on the undetectable profile P — but BT reaches 0.579 at ΔRR +0.49
and 0.831 at +0.65, so interpolated to B3's +0.54 the blind control already gives
≈ 0.68. The genuine prior contribution is roughly **+0.08**, not +0.76. Real,
consistently positive, and an order of magnitude smaller than the raw figure
suggests. Quote it against BT or not at all.

**The routing-weight lever κ is doing almost nothing.** `κ=0` and `κ=3` sit
inside each other's confidence intervals on every profile (P: 0.774 vs 0.757;
L: 0.996 vs 0.995). The quota lever ρ carries essentially the whole effect. With
only K=4 cached candidates, re-ranking them by `exp(κS̄)` has little room to act;
before the phase 1 κ sweep is run, either K should be raised or κ reported as
inert on this topology and load.

### Damage has a floor no policy can touch

A session whose own **endpoint** is compromised leaks however it is routed — the
attacker owns one end of it. That share of `D_eff` is set by the traffic matrix,
not by the policy, and with f = 0.2 on 14 nodes it is roughly 40 % of all
exposure. Mixing it in caps DRR at a value that has nothing to do with the
policy, so it is tracked separately:

```
D_eff = D_eff_relay + D_eff_endpoint
```

`D_eff` stays the headline damage metric; **`D_eff_relay` is what measures a
policy**, and DRR is reported on it. The oracle's `D_eff_total` reduction is only
0.21 while its relay reduction is 0.99 — the gap is exactly this floor.

### Three corrections phase 5 forced

**1. `net50` was not 2-connected.** The MST-plus-shortest-fill generator left 4
degree-1 nodes and 15 articulation points, so only **17.4 %** of node pairs had
two node-disjoint paths (T1 and T2: 100 %). B2 on the central config would have
rejected 83 % of demands, and the partition guard would have vetoed about a third
of B1's isolations — both artifacts of the generator rather than results. A
biconnectivity augmentation pass now runs after the fill loop and trims the same
number of redundant long edges, so the edge count and mean degree are unchanged.
Every topology file records its own `frac_ge_2_disjoint` / `frac_ge_3_disjoint`,
which is the ceiling on B2 and is asserted by `tests/test_topology.py`.

**2. `disjoint_paths` was edge-disjoint.** The threat model is compromised
*nodes*, and two edge-disjoint paths can share a relay — which sees both XOR
shares and reconstructs the key. It is now an exact minimum-cost **node**-disjoint
set, solved as a min-cost flow on the node-split graph rather than greedily
filtered from the K Yen paths (Yen's paths are minor perturbations of each other
and share most of their interior, so a greedy filter would have reported the
disjointness of K rather than of the network).

**3. The ρ throttle was anchored at `S̄ = 0`.** Phase 1 writes
`ρ = max(0, 1 − S̄/S_iso)`, which gives `ρ = 1` only at zero score. With a
configuration prior in the score no node is ever at zero: measured on T1, every
healthy node sat at `S̄ ≈ 0.22`, so **every node in the network** got `ρ ≈ 0.56`.
B3 became a blanket 44 % throttle that raised rejection by 26 points while
separating compromised from healthy nodes by 0.006 — it throttled everything and
protected nothing. `policy.rho_start` re-anchors the ramp at the calibrated quiet
score, preserving the property phase 1 is built around (`ρ` reaches 0 exactly at
`S_iso`) while spending the lever on the tail. It defaults to `0.0`, the literal
formula, so the change is explicit and sweepable — and the sweep shows the two
anchors are genuinely different operating points, not a right and a wrong answer:

| anchor | what it is | profile P | profile E | efficiency (DRR per ΔRR) |
|---|---|---|---|---|
| `ρ_start = 0` | prior-driven precaution stays on | **0.28–0.32** | 0.82–0.84 | 2.3 |
| `ρ_start = 0.268` | evidence-driven only | ≈ 0 | 0.73–0.86 | **4.3–4.7** |

The prior-driven anchor is the only thing that works against profile P; the
quiet anchor is twice as efficient wherever evidence exists.

### Score operating point, solved rather than guessed

`calibrate.calibrate_score` derives `(λ_s, θ₀)` in closed form against two
targets: a quiet network at the **highest** prior must stay under 0.45, and a
**median**-prior node with one feature saturated must exceed 0.70.

"Quiet" means the *measured* healthy-node evidence floor, taken as a per-node
time average and then an upper quantile across nodes — not an idealised `e = 0`.
Calibrating against zero put every threshold a whole noise floor too low and
produced false positives at `S_iso = 0.7` on a run with no attack at all. The
quantile is over per-node averages rather than raw samples because `S̄` is EWMA
smoothed: what drives a false positive is a node that looks bad *persistently*.

| | T1 | T2 | T3-OTP | T3-AES |
|---|---|---|---|---|
| evidence floor `e_k` | 0.19/0.24/0.15 | 0.14/0.16/0.15 | 0.16/0.13/0.15 | 0.15/0.13/0.15 |
| `λ_s` | 7.5 | 8.0 | 11.0 | 11.0 |
| `θ₀` | 0.252 | 0.212 | 0.258 | 0.257 |
| quiet at highest prior | 0.411 | 0.410 | 0.395 | 0.390 |
| one feature, median prior | 0.733 | 0.733 | 0.745 | 0.749 |

The prior keeps its job: on a quiet network it still spreads `S` over roughly
[0.05, 0.41], which is what the `κ` lever consumes, while never on its own
reaching an isolation threshold.

The spec's `λ_s = 6.0, θ₀ = 0.35` remain a sweep point. Note that changing
`λ_s` changes the cross-node ordering of `S` (the prior is per node), so **AUC is
not invariant** to this calibration — the phase 4 tables were re-run.

### Other corrections

- **The policy ran on the prior alone for the first 5400 s.** `policy.apply` was
  called from step 0 while the detector still returns the prior until
  `warmup + W_base`. Under `selection=top_keyflow` the prior correlates with the
  compromised set by construction, so a policy driven by it would have looked
  effective for exactly the reason the phase 4 analysis warns against. Policies
  that read `S̄` now wait; B0/B2/B4 need no detector and act from the end of
  warm-up. `policy_active_from` is recorded in every summary.
- **The quota coin was drawn per candidate path, not per node.** A node on 3 of
  the 4 candidates got three independent trials, so its effective pass-through
  was `1−(1−ρ)³` — a function of `K`, not of `ρ`. Memoised per demand; measured
  pass-through is now 0.503 at `ρ = 0.5`.
- **`detector.ablation` must be off from phase 5 on.** Scoring all seven feature
  subsets from one evidence matrix is exact *only* while nothing feeds back, and
  a policy closes that loop.
- **`RoutingConfig.T_route` was dead code** that was nonetheless hashed into
  every `run_id`. It now gates the rebuild, which is also the model statement
  that a controller pushes flow rules on a period rather than reacting
  continuously.

## The headline claim, and the network it is made on

**The primary robustness result of this paper is the 12-member topology family,
not any single network.** Three named topologies with one sample each answer
"did it work on NSFNET", which is a question about NSFNET. Connectivity is the
one structural property the whole response argument rests on — it decides
whether disjoint paths exist, whether the partition guard can isolate anything,
and how much a throttle costs — so it is swept as an axis with replication:
12 members, 50 nodes, mean degree 2.2 to 4.4, three independent samples per
point, **each calibrated on its own** (its own `λ` bisected to 10 % rejection,
its own `φ`, its own `(λ_s, θ₀)` solved against its own evidence floor, its own
quiet score as the throttle anchor).

`DRR_relay`, 20 seeds per member:

| mean degree | ≥3 disjoint | B1 (G) | **B3 (G)** | B1 (E) | **B3 (E)** |
|---|---|---|---|---|---|
| 2.2 | 0.4 % | 0.071 | **0.460** | 0.804 | **0.988** |
| 2.8 | 2.9 % | 0.035 | **0.469** | 0.775 | **0.988** |
| 3.6 | 10.7 % | 0.023 | **0.383** | 0.786 | **0.985** |
| 4.4 | 18.3 % | 0.040 | **0.211** | 0.669 | **0.976** |

The graded policy is ahead on every member at every density, and on the greedy
profile — the one the detector finds hardest and the one an operator should
expect — it is ahead by a factor of 5 to 13. That is the claim, and it is made
across twelve networks rather than inside one.

### Why `net50` is still in the paper

It is the **lever-study network**, and that is a different job. The
`S_iso × ρ_start` grid, the κ ablation, the weight sensitivity and the
attack-intensity sweeps are parameter studies: they need one network held fixed
so that the parameter is the only thing moving. Running the 4 × 4 lever grid on
all twelve members would multiply the matrix by twelve to answer a question the
family already answers more directly.

So the division is explicit, and the paper should state it in the method
section rather than leave a reader to infer it:

| question | evidence |
|---|---|
| does the ordering hold across networks? | **the 12-member family** |
| does it hold when one factor moves? | the OFAT block, 14 cells |
| does it hold against an adaptive attacker? | the minimax table |
| where should the levers be set? | the `net50` grid |
| which feature detects which attack? | the `net50` ablation, prior off |

`net50` itself is a generated 50-node metropolitan mesh rather than a deployed
network, and the justification is Hefei: a 46-node quantum metropolitan area
network exists, so the scale is not hypothetical. The family is generated the
same way, which makes `net50` one member of it at mean degree 3.6 rather than a
special case.

## Against published methods

B0–B4 and BT are all constructions of this paper. Showing that our graded
policy beats our own binary one is an **ablation, not a comparison with the
state of the art**, and a reviewer is right to say so. Three published methods
are therefore reimplemented and run on the same cell, the same seeds and the
same metric.

```bash
python -m sim.sweep --sweep config/sweep_baselines.yaml --no-assert
python scripts/analyze.py --baselines
```

| | source | what it does |
|---|---|---|
| **B5** | Luo & Li, *Entropy* 27(11):1100, 2025 | derates a link's capacity by an endpoint-report trust score |
| **B6** | Kiktenko, Tayduganov & Fedorov, *Entropy* 26(12):1102, 2024 | XOR shares over node-disjoint paths, selected by key-rate deficiency |
| **B7** | Bi, Miao & Di, *Appl. Sci.* 13:8690, 2023 | key-availability link weighting, no detection at all |

Each needed adaptations; they are listed in full in the class docstrings in
[`sim/policy.py`](sim/policy.py) and must be restated in the paper. The largest
is that B5's first factor counts Byzantine witness acknowledgements from a
consensus layer this simulator does not have, so what is ported is their trust
*mechanism*, not their protocol. **Each baseline is reported at its own best
operating point**, found by sweeping its own tuning parameter, not at ours.

### Result — net50/OTP, f = 0.10, 20 seeds, `DRR_relay`

| policy | source | P passive | G greedy | L liar | E tap |
|---|---|---|---|---|---|
| B1 binary τ=.5 | ours | −0.002 | 0.038 | 0.879 | 0.745 |
| **B3 graded** | **ours** | **0.287** | **0.224** | **0.979** | **0.979** |
| B2 XOR m=2 | ours | 0.853 | 0.842 | 0.853 | 0.848 |
| **B5 Luo & Li 2025** | published | 0.001 | 0.001 | **0.995** | −0.003 |
| B6 Kiktenko+ 2024 | published | 0.849 | 0.837 | 0.849 | 0.845 |
| B7 Bi+ 2023 | published | −0.034 | −0.026 | −0.034 | −0.033 |
| B4 oracle | bound | 0.980 | 0.977 | 0.980 | 0.978 |

Three findings, and the first is not in our favour.

**B5 matches the oracle on the liar — and is blind to everything else.** At its
own best point (`β = 50`) Luo & Li's trust rule reaches **0.995** on profile L,
above our graded policy's 0.979, for a comparable price (ΔRR +0.365 against
+0.327). We do not beat it there and the paper should say so plainly. What it
does on the other three profiles is 0.001, 0.001 and −0.003, at a cost of
ΔRR +0.014 — it correctly does nothing, because it correctly sees nothing.

That is the whole argument of this paper, produced by the published method
rather than against it. Their trust is built on **one observable** — the
disagreement between a link's two endpoint reports, which is exactly our `x2` —
and one observable covers one attack. The method is **narrow, not weak**. Adding
`x1` (key accounting) and `x3` (the quantum layer) is what turns a liar detector
into a compromised-relay detector, and profile E is where that shows: 0.979
against −0.003.

**Key-aware routing is not a substitute for detection — it is slightly worse
than nothing.** B7 returns a *negative* DRR on every profile and at every
`α_key` we swept (−0.026 to −0.043), at essentially zero cost (|ΔRR| < 0.005).
Routing around key-poor links lengthens paths, a longer path crosses more
relays, and more relays is more exposure. Bi et al.'s algorithm is not defective
— it optimises key utilisation and latency, and does that well — but it is not
aimed at exposure, and optimising the wrong objective moves the right one the
wrong way. This is the cleanest available answer to *"could a good key-aware
routing algorithm make a detector unnecessary?"*

**B6 and B2 are indistinguishable here, and that is a finding about their
contribution rather than ours.** 0.849 against 0.853 on P, 0.837 against 0.842
on G, at identical cost — well inside each other's confidence intervals.
Kiktenko et al.'s deficiency-driven path selection and our minimum-hop selection
choose equally good path sets at this load, because key is not the binding
constraint on *which* disjoint paths to take: on a biconnected graph the
disjoint pairs are few enough that their key rates barely differ. Their
selection rule should help where key is scarce and paths are plentiful, which is
not this regime. Reported neutrally; we have no reason to claim it does not work
where they claim it does.

### What the comparison does and does not establish

It establishes that the contribution is **`x1` and `x3`, and the graded
response built on all three** — not multipath (Kiktenko et al.'s, and credited
to them), and not the idea of trust-weighted derating (Luo & Li's).

It does **not** establish superiority on profile L, where the published method
is at least as good. The paper's claim has to be the one the table supports:
*across the threat model, not at its easiest point.*

## The objection this paper has to answer first

A reader reaching the headline result — *the graded policy's advantage over the
binary one is largest where the detector is weakest* — can turn it around:

> The proposed policy shows its advantage precisely where the authors' own
> detector is close to blind. That looks less like evidence that graded response
> is better and more like compensation for a detector that does not work.

The objection is fair and the answer is not rhetorical, so it is set out here
rather than left to a rebuttal.

**First, the weakness is in the observable, not in the estimator.** `x1`
saturates at AUC ≈ 0.66 on the greedy profile and the attack-intensity sweep
shows it saturating *low*: at γ = 1.0, a relay drawing double its legitimate
key, AUC reaches only 0.705. Extra key draw is the one thing honest background
traffic also does, so the greedy relay is camouflaged by variation that is
genuinely there. No estimator recovers a signal the channel does not carry.

**Second, the obvious repair was tried and failed, and the failure is
informative.** Background traffic is episodic and zero-mean; a greedy relay
drains on every sample. Averaging the residual before it reaches the robust
baseline should therefore shrink the episodic part as `1/√n` and leave the
persistent part. It does not: AUC moves 0.657 → 0.666 while the false-positive
rate goes 0.078 → 0.329. The reason is general — *temporal aggregation cannot
help a statistic whose scale is estimated from its own history*, because
smoothing deflates the MAD by very nearly the factor it deflates the noise. The
knob is kept at its default of zero so the negative result stays reproducible.

**Third, and most important: the paper does not claim the detector is
sufficient.** It claims the opposite, and the experiment matrix is built around
it. B2 — XOR shares over node-disjoint paths, the scheme of Kiktenko et al. —
carries no detection at all and is the only policy that helps against the
passive profile, where detection is impossible **by construction**. The
division of labour is the result:

| | detection-driven (B1, B3, B5) | detection-free (B2, B6) |
|---|---|---|
| P passive | ≈ 0 by construction | works |
| G greedy | weak, `x1`-limited | works |
| L liar | strong | works |
| E tap | strong | works |
| cost | rejection only | **+59 % key per demand at m = 2** |

Redundancy works everywhere and costs a lot; detection works where there is
evidence and costs little. A network that can afford neither everywhere needs
both, which is the argument for the architecture rather than for either half.

**Fourth, the graded/binary comparison is decided where the detector is
strong, not only where it is weak.** On the liar and the tap, where AUC is 0.95
and above, B3 still beats B1 — 0.98 against 0.88 and 0.98 against 0.74 at the
spec intensity — and the minimax analysis, which lets the attacker choose its own
intensity, separates them further: B1's worst case is **zero on all three active
profiles** while B3 retains 0.23–0.41. A threshold policy fails not because the
evidence is absent but because it has to commit to a side of a line; the graded
policy reads the same numbers and does not.

The honest summary, which the paper states in the abstract rather than hiding in
the discussion: **a graded response converts weak evidence into partial
protection, and that is worth having exactly because the evidence about the most
plausible attack is weak.**

## Where the lever ablation changed the story

Phase 6 block A was rerun as a genuine two-lever grid rather than a single
operating point, and two of the phase 5 conclusions did not survive it.

### κ is inert, and the figure is `S_iso × ρ_start` instead

The κ reading in "Other corrections" above was hedged — *"before the phase 1 κ
sweep is run, either K should be raised or κ reported as inert"*. That sweep has
now been run, at `S_iso = 0.9`, where the partition guard isolates **no** node at
all and κ is therefore the only lever left acting:

| κ | 0 | 3 | 10 | 20 |
|---|---|---|---|---|
| DRR_relay | 0.451 | 0.421 | 0.437 | 0.447 |
| mean hops | — | +6 % | +17 % | **+21 %** |

(net50/OTP, f = 0.10, `S_iso` = 0.9, `ρ_start` = 0.1187, 20 seeds, pooled over
P/G/L/E. The four DRR values sit inside each other's confidence intervals.)

Routing weight buys nothing and costs path length. The reason is structural, not
a tuning failure: κ and ρ act on **the same node** through two different
mechanisms — κ makes the controller prefer paths around it, ρ makes it admit less
through it — and ρ is by far the stronger of the two, because a rejected demand
carries no key at all while a re-routed one still traverses *some* relay. Once ρ
has throttled a suspect node, there is nothing left for κ to steer away from.

This is reported rather than hidden, and the headline figure is the `S_iso ×
ρ_start` grid instead. A negative result about one of the proposal's own levers
is worth more than a figure that manufactures a gradient out of noise.

### Headline frontier — net50 / OTP, f = 0.10, 20 seeds, pooled over P/G/L/E

`DRR_relay / ΔRR`:

| ρ_start ＼ S_iso | 0.4 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|
| **0.0000** literal formula | **0.909** / +0.592 | 0.862 / +0.545 | 0.770 / +0.464 | 0.663 / +0.398 |
| 0.1316 quiet mean | 0.585 / +0.327 | 0.581 / +0.311 | 0.519 / +0.240 | 0.413 / +0.202 |
| 0.2150 quiet p90 | 0.520 / +0.185 | 0.499 / +0.170 | 0.466 / +0.145 | 0.385 / +0.128 |
| 0.2632 twice the mean | 0.496 / +0.139 | 0.498 / +0.139 | 0.459 / +0.124 | 0.376 / +0.110 |

Reference points in the same cell:

| policy | DRR_relay | ΔRR | ΔKPD |
|---|---|---|---|
| B1 τ = 0.5 | 0.419 [0.330, 0.513] | +0.069 | +0.01 |
| B2 m = 2 | 0.848 | +0.354 | **+0.587** |
| B2 m = 3 | 0.998 | +0.770 | **+2.735** |
| B4 oracle | 0.979 | +0.118 | −0.01 |
| BT ρ = 0.99 / 0.97 / 0.90 | 0.000 / −0.000 / 0.082 | +0.002 / +0.049 / +0.195 | ~0 |
| BT ρ = 0.75 / 0.60 | 0.558 / 0.823 | +0.477 / +0.635 | ~0 |

The grid is monotone in both levers and in the expected directions, which is
what makes it usable as a frontier: Every `ρ_start` is a measured quantity rather than a
multiple: the mean score of a quiet network, its 90th percentile, and twice the
mean.  `ρ_start` sets how much of the network the
throttle touches, `S_iso` sets how readily a node is removed outright, and every
cell is a defensible operating point rather than a tuned one.

### The quiet-attacker slice — the sharpest result in the project

Every number above is at the **spec** attack intensity. The same cell was rerun
with each profile's intensity parameter at the quietest point of the grid.  For
profile E that means `q` **and** `s` together, since one physical tap produces
both — see the minimax section for what happened when they were swept apart:

| profile | intensity | B1 τ=.5 | **B3 S=.5 ρ₀=.132** | B2 m=2 | B4 |
|---|---|---|---|---|---|
| G | spec γ=0.30 | 0.031 / −0.00 | 0.242 / +0.23 | 0.840 / +0.31 | 0.977 / +0.09 |
| G | **quiet γ=0.05** | **−0.000** / +0.01 | **0.271** / +0.26 | 0.850 / +0.36 | 0.979 / +0.13 |
| L | spec δ=0.50 | 0.879 / +0.11 | 0.979 / +0.33 | 0.852 / +0.36 | 0.980 / +0.13 |
| L | **quiet δ=0.02** | **0.016** / +0.01 | **0.414** / +0.28 | 0.852 / +0.36 | 0.981 / +0.13 |
| E | spec q=0.030 | 0.738 / +0.16 | 0.978 / +0.38 | 0.846 / +0.33 | 0.978 / +0.11 |
| E | **quiet q=0.002** | **−0.002** / +0.00 | **0.291** / +0.27 | 0.851 / +0.36 | 0.981 / +0.13 |

**Against a quiet attacker the binary policy collapses to zero on every
profile, while the graded policy keeps 0.27–0.41 at the same cost.** B1's ΔRR
also goes to zero, which says what the collapse *is*: the score never crosses τ,
no node is ever isolated, and B1 degenerates into B0. B3 has no threshold to
miss — partial evidence buys partial throttling — so it degrades smoothly instead
of switching off. B2 and B4 are flat by construction; neither reads the detector.

This is the argument for a graded response stated as sharply as the simulator can
state it, and it is a *robustness* argument rather than a peak-performance one:
at spec intensity B1 is already respectable on L and E (0.88, 0.76). The
difference only appears where it matters, against an attacker who is trying not
to be seen.

### The attack-intensity curve, per channel

Same runs, detector only (`policy.type: B0`, prior off, net50, 10 seeds). AUC
against the profile's own intensity knob:

| profile | knob | range | AUC |
|---|---|---|---|
| G greedy | `γ` | 0.05 → 1.00 | 0.557 → 0.705 |
| L liar | `δ` | 0.02 → 0.50 | 0.988 → 1.000 |
| E tap | `q` | 0.002 → 0.030 | 0.567 → **0.959** |

The three evidence channels have intrinsically different power and the curves say
why. `x2` (neighbour discrepancy) saturates immediately — a liar is detectable at
the smallest inconsistency it can report, because the discrepancy is a *logical*
contradiction and the only thing masking it is the report noise. `x3` (quantum
layer) gives the clean S-curve of a signal emerging from a noise floor. `x1`
(key accounting) saturates **low**, at 0.70: extra key draw is the one thing the
background traffic model also does, so the greedy profile is partly camouflaged
by honest variation no matter how greedy it gets. That ceiling is a property of
the observable, not of the estimator, and it is why the paper needs B2 as well.

### `x1` is signed too, and what that did and did not fix

`x1` was `|K_obs − K_exp| / σ_node`. The absolute value is the same structural
defect the `x2` fix removed: it makes the feature **two-sided**, so a node that
consumed *less* than the controller authorised — an idle one, or one whose buffer
report drifted upward — scores exactly as high as one that drained extra key,
though only the second is an attack. Every attack that moves `x1` moves it
upwards, so the sign carries the discrimination and the magnitude carries that
plus a symmetric noise channel.

Signing it (`detector.x1_mode: signed`, now the default) does **not** raise the
ceiling on profile G:

| | x1 AUC on G | node FPR @ τ=0.5, P | G | L | E |
|---|---|---|---|---|---|
| `abs` (the phase 1 formula) | 0.655 | 0.098 | 0.111 | 0.107 | 0.184 |
| `signed` | 0.657 | **0.078** | **0.089** | **0.078** | **0.164** |

What it buys is a **20 % cut in the false-positive rate on every profile at no
cost in AUC**, which is what a two-sided statistic costs when only one side is
informative. The measured evidence floor of `x1` fell with it, 0.16 → 0.127 on
net50 — half of that feature's noise floor was the absolute value — and `(λ_s,
θ₀)` were re-solved for every topology as a result.

### Temporal aggregation on `x1`: a measured dead end

The `x1` ceiling at AUC ≈ 0.66 is the weakest point in the detection chain, and
it is worth saying precisely why the obvious fix does not work, because it is the
fix a reader will propose.

The two sources `x1` cannot separate differ in their *time* signature: background
traffic is episodic and zero-mean — sessions arrive, consume, leave — while a
greedy relay drains on every sample from `t_c` onward. Averaging the residual
before it reaches the robust baseline should therefore shrink the episodic part
as `1/√n` and leave the persistent part alone. `detector.x1_ema` implements
exactly that. Measured over a 1500 s window, on 10 seeds:

| | x1 AUC on G | node FPR @ τ=0.5 on P |
|---|---|---|
| off | 0.657 | **0.078** |
| EMA, 600 s | 0.663 | 0.231 |
| EMA, 1500 s | 0.666 | **0.329** |

Nothing gained, false positives up by a factor of four. The reason is that **the
baseline is estimated from the same series**. Smoothing makes the series
autocorrelated, so the median/MAD taken over `W_base` underestimates its spread
by very nearly the factor the noise shrank by; numerator and denominator fall
together and `z` barely moves — except that every node now looks more extreme
against its own deflated MAD.

Stated generally: *temporal aggregation cannot help a statistic whose scale is
estimated from its own history.* A method that works has to hold the baseline
fixed while the evidence accumulates — a CUSUM on `e`, which is a different
detector rather than a tuning of this one. The knob is kept at its default of 0
so that the negative result stays reproducible instead of becoming folklore.

The honest conclusion is the one the intensity curve already gave: the `x1`
ceiling is a property of the **observable**, not of the estimator. Extra key draw
is the one thing honest background traffic also does. That is why the paper needs
B2, which never has to detect anything.

### F2 disambiguation — `x2` is signed, not absolute

`x2` compared the two endpoint reports of a link as `|rep_a − rep_b| / B_max`.
The absolute value makes the feature **symmetric**: a link where a liar
under-reports raises the feature on *both* endpoints equally, so the honest
neighbour is accused exactly as loudly as the liar, and the discrepancy is
evidence against the pair rather than against a node. Signing it by the node's
own side (`detector.x2_mode: signed`, now the default) keeps the direction of the
inconsistency, which is the part that identifies who lied:

| | AUC on L | node FPR @ τ=0.5 |
|---|---|---|
| `x2_mode: abs` | 0.963 | 0.389 |
| `x2_mode: signed` | **1.000** | **0.111** |

No other profile moves, which is the correct signature: only L produces an
asymmetric discrepancy in the first place. The statistic is deliberately **not**
clipped at zero inside the feature — the detector already clips `z` at zero, and
applying one-sidedness twice would discard the peer baseline's own spread.

### Isolation does not evict in-flight sessions

`test_isolation_actually_removes_the_node_as_a_relay` failed when it was first
written, and the failure was a real modelling fact rather than a bug: isolating a
node blocks new admissions through it but leaves sessions already routed through
it running to completion. Both readings are defensible — an SDN controller can
tear down flows, or it can drain them — so it is now a config flag,
`policy.tear_down_on_isolate`, defaulting to `False` (drain).

The measured difference is **±0.008–0.02 DRR**, inside seed noise at 20 seeds.
Sessions are short relative to the post-`t_c` window, so the in-flight population
at the moment of isolation is small. Worth knowing that it is small; not worth
choosing a default over.

### OFAT results — 14 cells, 2 profiles, 20 seeds

`DRR_relay / ΔRR`, with the detector's own AUC taken from the B0 row so that
detectability and response are not confounded. Every cell moves exactly one
factor off the centre (net50 / OTP / f = 0.10 / medium load / random selection /
noise 1.0 / α = 0.05).

| cell | AUC (G) | B1 (G) | B3 (G) | AUC (E) | B1 (E) | B3 (E) |
|---|---|---|---|---|---|---|
| CENTRE | 0.516 | 0.040 | **0.212** | 0.950 | 0.756 | **0.975** |
| topology nsfnet | 0.673 | 0.120 | 0.156 | 0.920 | 0.915 | 0.964 |
| topology usnet | 0.576 | 0.002 | **0.072** | 0.950 | 0.829 | **0.965** |
| km_mode AES | 0.513 | 0.017 | **0.297** | 0.951 | 0.767 | **0.983** |
| f = 0.05 | 0.556 | 0.027 | **0.142** | 0.976 | 0.789 | **0.976** |
| f = 0.20 | 0.522 | 0.025 | **0.208** | 0.909 | 0.553 | **0.946** |
| f = 0.30 | 0.475 | 0.028 | **0.188** | 0.606 | 0.028 | **0.348** |
| load low | 0.510 | 0.030 | **0.438** | 0.950 | 0.760 | **0.986** |
| load high | 0.527 | 0.042 | 0.063 | 0.951 | 0.715 | **0.967** |
| noise ×0.5 | 0.524 | 0.103 | **0.269** | 0.950 | 0.755 | **0.976** |
| noise ×2 | 0.499 | 0.004 | **0.200** | 0.949 | 0.602 | **0.936** |
| α = 0.01 | 0.484 | 0.050 | **0.231** | 0.946 | 0.696 | **0.969** |
| α = 0.20 | 0.534 | 0.039 | **0.223** | 0.951 | 0.749 | **0.982** |
| selection top_keyflow | 0.981 | 0.030 | **0.272** | 0.988 | 0.696 | **0.991** |

**B3 is ahead of B1 in all 28 cells** - though on the three profile-G cells where the detector sits at chance (usnet 0.005 vs 0.002, load=high 0.052 vs 0.037) both are indistinguishable from zero and the ordering there means nothing. More useful than the ordering is what sets the
size of the gap: it is the detector's AUC, and nothing else. Where AUC > 0.9
(every E cell but `f = 0.30`) both policies work and the gap is small — 0.915 vs
0.964 on nsfnet. Where AUC approaches 0.5 (every G cell) B1 returns essentially
zero while B3 keeps 0.06–0.44. The `f = 0.30` row on E is the clearest single
line in the table: AUC falls to 0.606, B1 falls from 0.553 to **0.028**, and B3
from 0.946 to **0.348** — a factor of 20 against a factor of 2.7.

That is the same finding as the quiet-attacker slice, produced by fourteen
independent axes instead of by lowering the attack intensity, which is the
strongest form the claim can take: *the graded policy's advantage is not a
tuning artefact of one operating point, it is what happens whenever the evidence
is weak, and the evidence is weak for many different reasons.*

Two rows are worth reading on their own:

* **`selection=top_keyflow` sends AUC to 0.981 on G** — against 0.516 at the
  centre. The prior is doing that, not the features: under top-flow selection the
  compromised set correlates with exposure by construction, which is exactly the
  confound the phase 4 analysis warns about. It belongs in the table as a
  sensitivity, never in a headline detection number.
* **`load=high` is the only cell where B3 nearly ties B1 on G** (0.063 vs
  0.042). At high load the key buffers are near-empty for everyone, so the `x1`
  residual of a greedy node stops standing out against honest starvation.

### The bug the OFAT sweep found: AES never charged the relay split

`D_eff_relay` was accrued only in the OTP branch of `_consume_continuous`. The
relay/endpoint split arrived with the phase 5 policies, after the AES re-key path
had already been written, and `_apply_rekey_pulses` was not updated — so **every
AES run reported `D_eff_relay = 0`**.

What makes it worth writing down is that nothing caught it, and nothing could
have, given how the metrics were defined:

* `D_eff` stayed exact — the pulse path charged it correctly;
* the split identity `D_eff = D_eff_relay + D_eff_endpoint` still held, because
  `D_eff_endpoint` is **defined** as the remainder rather than measured;
* the deep invariant checks `exposed_rate_relay`, which was also correct — the
  *rate* aggregate was fine, it simply was never integrated in AES;
* no assertion in the run path compares an accrual against its own rate.

The symptom only appeared two layers downstream, as `nan` in the `km_mode=AES`
row of the OFAT table, because `DRR_relay = 1 − D/D_b0` became `0/0`. Half the
experiment matrix was silently unmeasurable on the metric every policy is judged
on, and the only reason it surfaced is that the OFAT table puts the AES cell
directly beside cells that work.

The regression test asserts the thing the identity cannot: that the relay share
is non-zero and equals an independent recomputation from the session log, in
**both** key modes (`test_relay_endpoint_split_is_charged_in_both_key_modes`).
The general lesson is that a quantity defined as a remainder can never falsify
the quantity it is the remainder of.

## Minimax — the comparison nobody gets to choose

Every table above compares the policies at **one** attack intensity, and the
intensity is chosen by the person writing the paper. That is the objection a
referee should raise, and it is a fair one: loud attacks favour the binary
policy, quiet ones favour the graded one, so the choice decides the answer.

The fix is to stop choosing. The attacker optimises too — for each policy it
takes the intensity that maximises the damage it gets through — and every policy
is then reported at its own worst case:

```
worst(policy) = max over intensity of D_eff_relay(policy, intensity)
```

net50/OTP, f = 0.10, 20 seeds, five intensities per profile. `DRR_relay`:

| profile | policy | quietest | ... | spec | **worst** | attacker's best move |
|---|---|---|---|---|---|---|
| G | B1 τ=0.5 | −0.000 | 0.004 · 0.033 · 0.082 | 0.102 | **−0.000** | as quiet as possible |
| G | **B3** | 0.271 | 0.255 · 0.234 · 0.228 | 0.258 | **0.228** | γ = 0.6, *interior* |
| L | B1 τ=0.5 | 0.016 | 0.286 · 0.729 · 0.876 | 0.879 | **0.016** | as quiet as possible |
| L | **B3** | 0.414 | 0.753 · 0.965 · 0.980 | 0.979 | **0.414** | as quiet as possible |
| E | B1 τ=0.5 | −0.002 | 0.005 · 0.216 · 0.609 | 0.749 | **−0.002** | as quiet as possible |
| E | **B3** | 0.291 | 0.399 · 0.626 · 0.949 | 0.972 | **0.291** | as quiet as possible |

B2 (0.83–0.85) and B4 (0.96–0.98) are flat by construction: neither reads the
detector, so there is no worst case for the attacker to find.

Two things follow, and the second is the more interesting:

**The worst-case ranking is the same as the chosen-point ranking, by a wider
margin.** B1's worst case is zero on all three profiles — the score never crosses
τ, so it degenerates into B0. B3 keeps 0.23–0.41. Against an adversary allowed to
adapt, the graded policy is not somewhat better, it is the difference between a
defence and no defence.

**The shape of the attacker's optimum differs between the two.** Against the
*threshold* policy the attacker's best move is always to go as quiet as the grid
allows — B1's curve falls monotonically to the left edge on all three profiles.
Against the *graded* policy the optimum on G is **interior** (γ = 0.6): going
quieter stops helping, because a quieter greedy node steals less key, and below
some point the reduction in theft outweighs the reduction in detection. A defence
that degrades instead of switching off changes what the attacker wants to do,
which is the game-theoretic statement of the same claim the DRR tables make.

### The bug this sweep found: profile E has two intensity knobs

The first version of the table had a non-monotone E row — DRR at `q = 0.005`
came out *below* `q = 0.002`. Profile E is driven by two parameters, `q` (the
QBER it adds to the tapped link) and `s` (the secret-key-rate it costs that
link), and one physical tap produces both. The generator swept `q` and left `s`
at the spec value, which is not a quieter attacker but an attacker invisible in
one observable and loud in the other. The pairs are now tied together, matching
`sweep_intensity.yaml`, and the row is monotone.

## Connectivity as an axis — the detail behind the headline

Three named topologies with one sample each answer "did it work on nsfnet" and
nothing more. Connectivity is the one structural property the whole response
argument rests on, so it is now a family: 12 members, 50 nodes, mean degree 2.2
to 4.4, three independent samples per point.

Each member is calibrated **on its own** — `λ` bisected to 10 % rejection, its
own `φ`, its own `(λ_s, θ₀)` solved against its own evidence floor, its own quiet
score as the throttle anchor. Holding any of those fixed would compare a lightly
loaded dense network against a saturated sparse one and call the difference
connectivity.

`DRR_relay / ΔRR`, 20 seeds per member:

| mean degree | ≥3 disjoint | B1 τ=.5 (G) | B3 (G) | B2 m=2 | B2 m=3 | B4 |
|---|---|---|---|---|---|---|
| 2.2 | 0.4 % | 0.070 / +0.042 | **0.458** / +0.363 | 0.863 / +0.481 | 1.000 / **+0.874** | 0.962 |
| 2.8 | 2.9 % | 0.035 / +0.008 | **0.475** / +0.359 | 0.833 / +0.423 | 0.999 / **+0.855** | 0.975 |
| 3.6 | 10.7 % | 0.022 / +0.003 | **0.382** / +0.316 | 0.815 / +0.379 | 0.994 / +0.792 | 0.979 |
| 4.4 | 18.3 % | 0.038 / +0.006 | **0.208** / +0.233 | 0.865 / +0.392 | 0.994 / +0.716 | 0.976 |

**Every member is biconnected by construction**, so by Menger's theorem every
pair has two node-disjoint paths and B2 with `m = 2` is feasible everywhere —
its row is flat and density buys it nothing. What density actually buys is the
*third* path, and `m = 3` is the row that tracks it: `frac_ge_3_disjoint` runs
0.4 % to 18.3 % across the family.

The `m = 3` column is also the clearest illustration in the project of why DRR
has to be read next to its cost. It reaches **1.000** at mean degree 2.2 — and
does it by rejecting 87 % of the traffic, because three node-disjoint paths
essentially never exist there. Perfect protection of almost nothing.

B3 on profile G *falls* with density, 0.458 → 0.208. The throttle is anchored to
each member's own quiet score, and that score drops as degree rises (0.174 →
0.137): a denser network gives the peer baseline more neighbours to compare
against, so healthy nodes look more alike and the anchor sits lower — which
leaves the ramp less room between the anchor and `S_iso`. Worth knowing as a
tuning fact; it is not a claim that sparse networks are safer.

## When is it worth evicting in-flight sessions?

Isolating a node blocks new admissions through it but lets sessions already
routed through it drain (`policy.tear_down_on_isolate: false`). On the central
cell the two settings differ by ±0.01 DRR, inside seed noise — but that is a fact
about a traffic model whose sessions last 60–600 s against a 3 h attack window,
not a fact about QKD networks.

`E[T_s]` now runs over two orders of magnitude with `λ` rescaled by its inverse,
so offered load is held and the axis is session length alone. `DRR_relay`, the
gain being evict minus drain:

| E[T_s] | B1 drain | B1 evict | **B1 gain** | B3 drain | B3 evict | **B3 gain** |
|---|---|---|---|---|---|---|
| 82 s | 0.766 | 0.769 | +0.003 | 0.985 | 0.984 | −0.001 |
| 330 s (spec) | 0.738 | 0.765 | +0.027 | 0.978 | 0.980 | +0.002 |
| 1320 s | 0.745 | 0.779 | +0.034 | 0.964 | 0.980 | +0.016 |
| 5400 s | 0.662 | 0.787 | **+0.125** | 0.922 | 0.976 | **+0.053** |

(profile E; profile G is the same shape at a tenth the scale, +0.045 / +0.017.)

**The gain grows monotonically with session length and is worth nothing below
it.** A controller that can tear down established flows is more complex than one
that drains them, and this is the table that says when to pay for the
complexity: at the spec traffic model, never; at session lengths comparable to
the detection delay, it is worth a fifth of what the whole policy is worth.

The spec default stays `false` — drain — because that is the weaker assumption
about what an SDN controller can do, and the paper's claims should not depend on
the stronger one.

## The prior is a confound, and here is how large

The configuration prior `π_i = σ(c0 + c1·Exposure + c2·Φ + c3·Age)` is a function
of exposure. Under `selection = top_keyflow` the attacker picks nodes by exposure
too, so the prior correlates with the ground truth *by construction*:

| selection | profile | AUC prior **off** | AUC prior **on** | inflation |
|---|---|---|---|---|
| random | P | 0.492 | 0.476 | −0.016 |
| random | G | 0.607 | 0.512 | −0.094 |
| random | L | 0.942 | 0.998 | +0.056 |
| random | E | 0.915 | 0.950 | +0.035 |
| **top_keyflow** | **P** | **0.420** | **0.898** | **+0.479** |
| top_keyflow | G | 0.899 | 0.980 | +0.081 |
| top_keyflow | L | 1.000 | 1.000 | +0.000 |
| top_keyflow | E | 0.973 | 0.989 | +0.015 |

The row that matters is **P with top-flow selection**: profile P is undetectable
by construction — it changes no observable at all — and the prior alone scores
**AUC 0.898** on it. That is a detector that has detected nothing, reporting near
perfect discrimination, and it is exactly what would have been published if the
prior had been left switched on in the phase 4 tables.

Every detection number in this project is therefore quoted with the prior **off**.
The prior stays in the model because the policy uses it — it is what `κ` and the
`ρ` ramp consume on a quiet network — but it is not evidence and it is never
allowed into an AUC.

## Phase 7 — analytical validation

```bash
python -m pytest tests/test_validation.py -q
```

No ns-3. Eleven closed-form checks, each chosen to test an *interaction* rather
than a component, because component-level behaviour is already covered by the
phase 2–5 suites:

| check | what it would catch |
|---|---|
| `M/G/∞` session population `= (1 + unmanaged)·λ·(E[T_s] + dt)` | arrival, lifetime and the unmanaged-traffic multiplier disagreeing |
| offered load is Poisson (dispersion test) | a seeded arrival stream that is not actually memoryless |
| fluid limit of the buffer against the reflected model | the `min`/`max` clipping order in the buffer update |
| key conservation with attack **and** policy **and** multipath all on | the three feedback paths double-counting a draw |
| mean first-candidate hops vs `nx.average_shortest_path_length` | a routing cost function that is not the one being reported |
| admitted mean vs offered mean (length bias) | reporting an admission-biased statistic as a network property |
| `f = 1.0` closed form for `D_eff` | the damage accounting, against an independent recomputation |
| no attack ⇒ no damage, for **every** policy | a policy that manufactures exposure |
| slow-reference mode equals the incremental aggregates | every `O(|E|)` incremental update in the hot loop |
| key rate model and buffer fill time | `R_e = R_max·exp(−L_e/L_0)` and the fill integral |

The `M/G/∞` check is the one that earns its keep: it failed twice, once for the
missing `+ dt` (a session admitted at step *k* is counted at step *k* as well as
for its whole lifetime) and once for the unmanaged multiplier, and neither would
have shown up in any single-component test.

## Phase 8 — sensitivity to the detector weights

```bash
python scripts/make_phase8.py            # regenerates the sweep, recalibrating
python -m sim.sweep --sweep config/sweep_p8_weights.yaml --no-assert
python scripts/analyze.py --weights
```

Ten weight points — uniform, one feature favoured, one de-weighted, one dropped —
each run **twice**: with `(λ_s, θ₀)` re-solved for that weighting, and at the
phase 1 defaults.

The caveat that forces the doubling: `θ₀` is calibrated against the largest
evidence a *single-feature* attack can produce, which is `w_k`, not `1/3`. Hold
`θ₀` fixed and a weight sweep silently becomes a threshold sweep. The closed-form
solution moves `λ_s` from **11.0** at uniform weights to **18.0** when one
feature is favoured and **7.0** when one is dropped — a factor of 2.6 in the
score's slope, entirely as a side effect of renormalising three numbers that sum
to 1. The gap between the two surfaces is how much of any apparent weight
sensitivity is really this.

### Phase 8 results — the weights barely matter, the threshold does

Ten weightings × recalibrated / fixed × 4 profiles × 10 seeds. AUC:

| weights (w₁,w₂,w₃) | λ_s | P | G | L | E |
|---|---|---|---|---|---|
| (0.33, 0.33, 0.33) | 11.0 | 0.493 | 0.620 | 1.000 | 0.958 |
| (0.6, 0.2, 0.2) | 17.5 | 0.510 | 0.645 | 0.999 | 0.956 |
| (0.2, 0.6, 0.2) | 18.0 | 0.506 | 0.622 | 1.000 | 0.958 |
| (0.2, 0.2, 0.6) | 18.0 | 0.478 | 0.578 | 0.999 | 0.958 |
| (0.4, 0.4, 0.2) | 18.0 | 0.511 | 0.644 | 1.000 | 0.957 |
| (0.4, 0.2, 0.4) | 17.5 | 0.495 | 0.620 | 0.999 | 0.958 |
| (0.2, 0.4, 0.4) | 18.0 | 0.488 | 0.601 | 1.000 | 0.958 |
| (0.5, 0.5, **0.0**) | 7.5 | 0.519 | 0.656 | 1.000 | **0.512** |
| (0.5, **0.0**, 0.5) | 7.5 | 0.487 | 0.613 | **0.485** | 0.957 |
| (**0.0**, 0.5, 0.5) | 7.0 | 0.474 | **0.466** | 1.000 | 0.961 |

Three results:

**Re-weighting a feature that is present changes almost nothing.** Across the
seven weightings that keep all three features, AUC varies by 0.067 on G, 0.033 on
P, 0.002 on E and 0.001 on L — despite w₁ running from 0.2 to 0.6. The score is
dominated by *which* feature fires, not by how loudly it is weighted, because the
per-feature evidence `e_k` is already a clipped standardised z and saturates.

**Dropping a feature destroys exactly one profile, and the right one.** `w₃ = 0`
takes E from 0.958 to 0.512; `w₂ = 0` takes L from 1.000 to 0.485; `w₁ = 0` takes
G from 0.620 to 0.466. This is the phase 4 ablation table rederived from a
completely different direction — the ablation scored feature *subsets* post hoc
from one run, while this changes the live scoring function — and the one-to-one
feature↔profile correspondence comes out identical. That agreement is the best
evidence in the project that the correspondence is a property of the model rather
than of the evaluation code.

`w₁ = 0` puts G at 0.466, *below* chance, which is not noise at 10 seeds: a
greedy node drains its own buffers faster, so with `x1` removed the remaining two
features see it as slightly *healthier* than its peers. Dropping the feature that
matches the attack is worse than having no detector.

**All of the threshold sensitivity lands on the operating point, none on the
ranking.** Re-solved and fixed `(λ_s, θ₀)` differ by ≤ 0.007 AUC everywhere,
which is what the theory demands — AUC is a ranking statistic and `λ_s` is a
monotone rescaling. But the node false-positive rate at τ = 0.5 moves by a factor
of two, 0.078 → 0.158 at (0.2, 0.6, 0.2):

| weights | FPR@τ=0.5 re-solved | FPR@τ=0.5 fixed |
|---|---|---|
| (0.6, 0.2, 0.2) | 0.158 | 0.116 |
| (0.2, 0.6, 0.2) | 0.158 | 0.078 |
| (0.2, 0.4, 0.4) | 0.118 | 0.098 |
| all others | unchanged | unchanged |

So the caveat that motivated running both surfaces was real, but it is narrower
than feared: **a weight sweep reported on AUC alone would have been safe; one
reported on any threshold-dependent quantity — FPR, isolation count, DRR under
B1 — would have been measuring `θ₀` and calling it `w`.**

## Figures

```bash
python scripts/analyze.py --all          # -> results/analysis.parquet
python scripts/figures.py                # -> results/figures/*.png
```

`analyze.py` reports on the current feature set only.  `x1_mode` and `x2_mode`
are hashed into the run id, so the absolute-value forms of both features - the
literal phase 1 formulas, kept as sweep points - sit in `results/raw` beside the
signed ones instead of overwriting them.  That is what makes the before/after
panels possible, and it is also a trap: any report that is not *about* the
feature form has to say which form it means, or it silently averages two
different detectors.  `current()` does that for the reports; the parquet keeps
everything, because F3 needs the `abs` rows.

One rule governs `scripts/figures.py` and it is the only thing that makes the
numbers comparable. `results/raw` holds sixteen sweeps stacked on top of each
other and most of them share the same central cell, so **a figure may only pool
runs that differ in the axis it is plotting**. `cell()` pins everything not being
varied — topology, key mode, `f`, load, α, noise, selection, attack intensity,
prior, detector weights — and each figure relaxes exactly one pin. Three real
errors were caught by that discipline while the figures were being built:

* F1/F2/F3 initially pooled the **phase 8** runs, which sit at the central cell
  with the weights moved. The spec point of the intensity curve came out *below*
  its neighbours, which would have read as the attack becoming harder to detect
  as it got louder.
* F10's topology cells came out **empty**, because relaxing the topology pin and
  then filtering kept net50's `demand.lam` — which is re-derived per topology to
  hold offered load constant, so no nsfnet row survives it.
* F13 filed the uniform-weight **recalibrated** point under "fixed", because
  uniform weights solve back to `λ_s = 11.0` and differ only in `θ₀`. Keying on
  the `(λ_s, θ₀)` pair fixed it.

| figure | question it answers |
|---|---|
| F1 `f01_ablation` | which feature detects which attack |
| F2 `f02_intensity` | how detectability scales with attack intensity, per channel |
| F3 `f03_x2_mode` | what signing the neighbour discrepancy bought |
| F4 `f04_score_separation` | where `S̄` actually sits relative to the thresholds |
| F6 `f06_frontier_grid` | the graded policy's two levers as surfaces |
| F7 `f07_targeting` | how much of each policy's benefit is detection rather than volume |
| F8 `f08_cost_axes` | the price on its three separate axes |
| F9 `f09_quiet_attacker` | **the central claim**: binary collapses, graded degrades |
| F10 `f10_ofat` | does the ordering survive moving one factor at a time |
| F11 `f11_f_curve` | behaviour against the compromise fraction |
| F12 `f12_kappa` | the routing lever, measured where it is the only lever |
| F13 `f13_weights` | phase 8: weight sensitivity vs threshold sensitivity |
| F14 `f14_damage_floor` | why DRR is reported on the relay share |
| F15 `f15_stability` | controller-side behaviour: isolation load, flapping, vetoes |
| F16 `f16_minimax` | **every policy at its own worst attack intensity** |
| F17 `f17_family` | DRR and cost against mean node degree, 12 calibrated members |
| F18 `f18_session_len` | drain vs evict against session length |
| F19 `f19_baselines` | **against three published methods**, per profile, with cost |

Three readings that only became visible once the data was plotted:

**F4 explains F9 mechanically.** Compromised nodes under L and E sit at
`S̄ ≈ 0.60` — above `τ = 0.5` but below `τ = 0.7` — while under G they sit at
0.17, barely above the healthy 0.12. A binary policy is therefore balanced on a
knife edge at the spec intensity and falls off it the moment the attacker
quietens. The graded policy is reading the same numbers; it just does not have to
commit to a side of a line.

**F15 shows the throttle, not isolation, is doing the work.** At `S_iso = 0.9`
B3 isolates 0.08 nodes on average and never flaps once, yet F12 measures it at
DRR 0.45. Essentially all of the graded policy's benefit at that operating point
comes from `ρ`, which is also why κ has nothing left to act on.

**F8 shows B3's path stretch is negative** (−0.16 at `S_iso = 0.4`). Throttling
rejects long paths preferentially, because a long path crosses more relays and so
is more likely to meet a throttled one. The admitted traffic is therefore
*shorter* than B0's — a selection effect, not a routing improvement, and it has
to be read next to the rejection rate in the left panel rather than on its own.

## Definition of done

**Phase 2**

| # | criterion | status |
|---|---|---|
| 1 | `python -m sim.runner --config config/base_t1_otp.yaml` produces a parquet | done |
| 2 | all pytest green, including the four invariants | 129 tests green |
| 3 | same seed → byte-identical output file | `test_byte_identical_parquet` |
| 4 | 24 h run on T3 under 60 s on one core | 2.2 s |
| 5 | `sweep.py` runs 20+ runs in parallel, interruptible and resumable | 160-run sweep, resume verified |
| 6 | `calibrate.py` produces λ for the three load levels | T1, T2, T3-OTP, T3-AES |

**Phase 3**

| # | criterion | status |
|---|---|---|
| 1 | all four profiles P/G/L/E run and report `D_eff` | done, see the damage table |
| 2 | `D_eff` on P relates to `D_raw` as expected and no policy changes it | equal in OTP by construction; `NullPolicy` throughout |
| 3 | noisy observables recorded in `events.parquet` | `qber_obs_mean`, `x1..x3`, `e1..e3` |
| 4 | `test_damage_accounting` green | green, against an independent recomputation from the session log |
| 5 | runtime within +20 % of phase 2 | +16 % on the heaviest profile (2.58 s vs 2.23 s) |

**Phase 5-6**

| # | criterion | status |
|---|---|---|
| 1 | five competing policies B0-B4 implemented | done |
| 2 | B0 numerically identical to the phase 4 path | `test_b0_is_numerically_identical_to_no_policy` |
| 3 | the oracle isolates the whole compromised set | DRR_relay 0.988, `test_oracle_removes_almost_all_relay_damage`. NOT a bound on damage prevented: it never throttles, so a throttling policy can pass it by admitting less traffic. See REPRODUCE.md. |
| 4 | B2 relays XOR shares over node-disjoint legs | exhaustive truth table, `tests/test_multipath.py` |
| 5 | the policy never sees ground truth except in B4 | `test_policy_isolation` |
| 6 | DRR / PSI / C joined to matched baselines | `scripts/analyze.py`, hash join on `scenario_id` |
| 7 | Pareto frontier over the operating points | `--pareto`, non-dominated set marked |
| 8 | the graded policy's benefit is targeting, not volume | +0.9 over the blind control at matched cost on L/E; ~+0.08 on P |
| 9 | a volume control exists so the two can be told apart | `policy.type: BT` |

**Phase 4**

| # | criterion | status |
|---|---|---|
| 1 | `S̄` for every node in `events.parquet` | done |
| 2 | ROC/AUC, detection delay and FPR produced | `sim/evaluate.py`, in-process and as an offline CLI |
| 3 | AUC on profile P near 0.5 | 0.43–0.53 across topologies and seeds |
| 4 | AUC on G/L/E in 0.7–0.95, never 1.0 | G 0.72 (T1) / 0.80 (T3, top-flow), L 0.96, E 0.95; `test_auc_not_perfect` guards the ceiling |
| 5 | ablation table shows the one-to-one feature↔profile correspondence | yes, see above |
| 6 | `test_detector_isolation` green | green |

**Phase 7**

| # | criterion | status |
|---|---|---|
| 1 | closed-form checks on interactions, not components | 11 checks, `tests/test_validation.py` |
| 2 | queueing: session population matches `M/G/inf` | green, after the `+dt` and unmanaged-multiplier corrections |
| 3 | routing: mean hops match `networkx` | green, with admission length bias reported separately |
| 4 | damage: `f = 1.0` closed form | green, incl. the multipath exposure case |
| 5 | every incremental aggregate matched by a slow reference | `--slow-reference`, green |
| 6 | no attack implies no damage under **every** policy | green for B0-B4 and BT |

**Phase 8**

| # | criterion | status |
|---|---|---|
| 1 | weight sweep over 10 points, uniform to one-dropped | `config/sweep_p8_weights.yaml`, generated |
| 2 | `(lam_s, theta_0)` re-solved at every weight point | `scripts/make_phase8.py`; lam_s moves 7.0 - 18.0 |
| 3 | both surfaces reported so the two sensitivities separate | `--weights`; AUC gap <= 0.007, FPR gap up to 2x |
| 5 | the feature-profile correspondence survives a live re-weighting | dropping w_k kills exactly profile k |
| 4 | the `C`-weight simplex sensitivity costs no extra runs | `--cost`, reports how often each point wins |

### Performance

Routing is lazy in the pair *and* in K (admission usually stops at the cheapest
candidate, while materialising all K costs 17–32× more than one Dijkstra), and
the isolation-pruned graph is built once per rebuild instead of copied per query.

| 24 h run | wall time |
|---|---|
| T1, attack + detector + policy | 3.3 s |
| T3, attack + detector + policy | 5.4 s |
| 840-run policy sweep, 23 workers | 2 min 32 s |

---

## Future work

**Cross-validation against QKDNetSim / ns-3 — still open, and deliberately so.**
ns-3 is not installed here and building it on Windows is a project of its own, so
this was not attempted rather than attempted and abandoned. Phase 7 is analytical
only by design — the closed forms test this simulator against the model it claims to
implement, which is a different question from testing the model against a packet
level one. The natural next step is a single shared scenario (T1, one profile,
no policy) run in both and compared on buffer occupancy and rejection rate.

**Raising `K`.** κ was measured inert with `K = 4` cached candidates. The
structural argument above says raising `K` should not rescue it, since ρ acts on
the same node more strongly — but that is an argument, not a measurement.

**A CUSUM detector.** The `x1` ceiling at AUC ≈ 0.66 is the weakest link in the
chain, and temporal aggregation was measured and rejected: it cannot help a
statistic whose scale is estimated from its own history. Accumulating evidence
against a *fixed* baseline — a CUSUM on `e` — is the textbook answer to
"persistent small shift against episodic noise" and is the one version of the
idea this simulator has not tried. It is a different detector, not a tuning of
this one, so it belongs in its own phase.

**The attacker's move set.** The minimax table lets the attacker choose an
intensity. It does not let it choose a *profile*, switch profiles over time, or
coordinate across compromised nodes. Each of those is a larger move set and each
can only lower the worst case; the ordering between B1 and B3 should widen rather
than close, since all three make the evidence weaker, but that is a prediction,
not a result.

**A learned detector.** The score is a fixed weighted sum by construction, so
that the operating point can be solved for rather than fitted, and so that the
three features remain individually interpretable. The ablation table gives the
labelled data a supervised variant would need.

Nothing in `runner.py` changes for any of these. The registries in
`interfaces.py` (`ATTACKS`, `DETECTORS`, `POLICIES` plus the deferred `_LAZY`
table) are the only place a new component is named.
