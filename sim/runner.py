"""Main simulation loop.

Phases 3-4: the loop body below is unchanged in shape - the attack, the noise
model and the detector all arrive through the factories in ``interfaces``.

Fixed step loop, dt = 1 s. Three distinct cadences (phase 4, section 4.1):

    every step               key generation, attack effect, consumption, demand
    every detector.T_sample observation noise, feature extraction, score, EWMA
    every telemetry_period   policy update and the node level event log

Collapsing the last two into one period is what the phase 4 spec explicitly
forbids: the robust baseline needs W_base / T_sample = 120 samples, and at a
300 s sampling period it would get 6.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
import yaml

from . import keygen
from .config import SimConfig, load_config
from .demand import DemandGenerator, SessionManager
from .interfaces import build_attack, build_detector, build_policy, ewma
from .logging_io import write_run
from .metrics import MetricsAccumulator
from .noise import ObservationModel
from .routing import RouteTable
from .state import assert_invariants, init_state
from .topology import load_topology

DEEP_ASSERT_EVERY = 1000     # the edge_rate consistency check is expensive
BACKGROUND_RID_OFFSET = 1_000_000_000


@dataclass
class RunResult:
    """Everything a run produced - the tests inspect the internals, callers of
    :func:`run` only need ``summary``."""
    summary: dict
    state: object
    metrics: MetricsAccumulator
    topo: object
    routes: RouteTable
    buffers_initial: np.ndarray
    attack: object = None
    detector: object = None


def run(cfg: SimConfig, *, write: bool = True, asserts: bool = True,
        progress: bool = False, events: bool = True) -> dict:
    """One complete run. Returns the summary dict and writes the parquet files."""
    return simulate(cfg, write=write, asserts=asserts, progress=progress,
                    events=events).summary


def simulate(cfg: SimConfig, *, write: bool = True, asserts: bool = True,
             progress: bool = False, events: bool = True,
             keep_session_log: bool = False) -> RunResult:
    topo = load_topology(cfg.topology, cfg.noise.qber_base_min,
                         cfg.noise.qber_base_max)
    state = init_state(topo)

    # Five independent streams. Sharing one would mean that changing lambda
    # moves the compromised node selection, or that changing the attack profile
    # moves the observation noise - a methodological bug that is very hard to
    # find after the fact.
    rng_demand = np.random.default_rng(cfg.seed_demand)
    rng_policy = np.random.default_rng(cfg.seed_policy)
    rng_attack = np.random.default_rng(cfg.seed_attack)
    noise_seq = np.random.SeedSequence(cfg.seed_noise).spawn(2)
    rng_noise = np.random.default_rng(noise_seq[0])
    rng_background = np.random.default_rng(noise_seq[1])

    key_flow = load_phi(cfg)
    observer = ObservationModel(cfg.noise, topo, rng_noise)

    attack = build_attack(cfg.attack, n_nodes=topo.n_nodes, rng=rng_attack,
                          horizon=cfg.horizon, topo=topo, dt=cfg.dt,
                          key_flow=key_flow, noise=observer)
    detector = build_detector(cfg.detector, n_nodes=topo.n_nodes, topo=topo,
                              key_flow=key_flow, seed_topology=cfg.seed_topology)
    policy = build_policy(cfg.policy, n_nodes=topo.n_nodes, topo=topo,
                          oracle=attack.compromised_nodes, detector=detector)
    state.S_bar = detector.initial_S_bar(topo.n_nodes)
    state.n_paths = policy.apply(state.S_bar, 0.0).n_paths
    # A score driven policy must wait for the detector's baseline; one that
    # needs no detector (B0, B2, B4) acts from the end of the metric warm-up.
    policy_active_from = float(cfg.warmup + (
        cfg.detector.W_base if getattr(policy, "needs_detector", False) else 0.0))

    routes = RouteTable(topo, cfg.routing.K, cfg.routing.disjoint_K,
                        T_route=cfg.routing.T_route)
    routes.rebuild(state.node_weight, state.isolated, t=0.0)

    metrics = MetricsAccumulator(cfg.warmup, topo, keep_session_log=keep_session_log)
    demand_gen = DemandGenerator(cfg.demand, topo.n_nodes, rng_demand)
    lam_bg = cfg.noise.unmanaged_fraction * cfg.demand.lam
    background_gen = DemandGenerator(cfg.demand, topo.n_nodes, rng_background,
                                     lam=lam_bg, rid_offset=BACKGROUND_RID_OFFSET)
    session_mgr = SessionManager(cfg.demand, topo, metrics, rng_policy)

    alpha = float(cfg.detector.alpha)
    dt = cfg.dt
    n_steps = cfg.n_steps
    sample_every = max(1, int(round(cfg.detector.T_sample / dt)))
    tel_every = max(1, int(round(cfg.telemetry_period / dt)))
    B_max = topo.B_max
    buffers = state.buffers
    buffers_initial = buffers.copy()
    # only profile G moves the buffers, so only then is the extra sum worth it
    attack_drains = getattr(attack, "profile", None) == "G"

    iterator = range(n_steps)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=cfg.run_id, unit="step")
        except ImportError:
            pass

    n_cut = 0                      # sessions torn down by isolation
    t0 = time.perf_counter()
    for step_i in iterator:
        t = step_i * dt
        state.t = t

        # 1. key generation, at the rate the attack has left intact
        before = float(buffers.sum())
        keygen.step(buffers, state.R_eff, dt, B_max)
        after_gen = float(buffers.sum())
        metrics.note_generated(after_gen - before, t)

        # 2. attack effect on the true state
        attack.apply(state, t)
        if attack_drains and attack.last_drain:
            # siphoned key never reaches the ledger: that is the x1 residual
            metrics.note_consumed(attack.last_drain, 0.0, t)

        # 3. consumption of active sessions + AES pulses + starvation + damage
        session_mgr.step(state, t, dt)

        # 4. new demands - managed first, then the N4 background stream
        for req in demand_gen.arrivals(t, dt):
            s = session_mgr.try_admit(req, state, routes, topo, cfg.demand)
            if s is None:
                reason = session_mgr.last_reject
                if reason == "key" and not routes.reachable(req.src, req.dst):
                    reason = "unreachable"
                metrics.on_reject(req, t, reason)
            else:
                metrics.on_admit(s, t)
        for req in background_gen.arrivals(t, dt):
            s = session_mgr.try_admit(req, state, routes, topo, cfg.demand,
                                      managed=False)
            if s is not None:
                metrics.on_background_admit(s, t)

        # 5. observation and scoring, on the detector's own cadence
        if step_i % sample_every == 0:
            observer.observe(state, t)          # honest noisy measurement
            attack.observe(state, t)            # profile L lies about it
            if t >= cfg.warmup:
                # the robust baseline must be built on steady state behaviour,
                # not on the initial transient of buffers draining from B_max
                S = detector.update(state, topo, t)
                state.S_bar = ewma(S, state.S_bar, alpha)

        # 6. policy update and event log, on the coarser telemetry cadence
        if step_i % tel_every == 0:
            if t >= policy_active_from:
                decision = policy.apply(state.S_bar, t, state=state)
                state.node_weight = decision.node_weight
                state.rho = decision.rho
                state.isolated = decision.isolated
                state.n_paths = decision.n_paths
                state.hybrid_tau, state.hybrid_m, state.hybrid_agg = decision.hybrid
                if cfg.policy.tear_down_on_isolate and decision.isolated.any():
                    n_cut += session_mgr.tear_down_through(
                        state, np.flatnonzero(decision.isolated), t)
                if routes.is_stale(decision.node_weight, decision.isolated, t,
                                   edge_cost=decision.edge_cost):
                    routes.rebuild(decision.node_weight, decision.isolated, t,
                                   edge_cost=decision.edge_cost)
            metrics.log_telemetry(state, t, detector)

        # 7. invariants
        if asserts:
            assert_invariants(state, topo, deep=(step_i % DEEP_ASSERT_EVERY == 0))

        metrics.on_step(state, t)

    state.t = n_steps * dt
    elapsed = time.perf_counter() - t0

    summary = metrics.summary(state)
    summary["route_table_versions"] = routes.version
    summary["n_route_rebuilds"] = routes.n_rebuilds
    summary["n_sessions_open_at_end"] = len(state.sessions)
    summary["phi_source"] = "calibration" if key_flow is not None else "missing"
    summary["policy_active_from"] = policy_active_from
    summary["n_torn_down_on_isolate"] = n_cut
    summary.update(getattr(policy, "counters", dict)())
    if cfg.detector.enabled:
        from .evaluate import evaluate_detector
        summary.update(evaluate_detector(detector, state))
    summary["wall_time_s"] = round(elapsed, 4)

    if write:
        paths = write_run(cfg, summary,
                          metrics.telemetry_table() if events else None)
        summary["summary_path"] = paths["summary"]

    return RunResult(summary=summary, state=state, metrics=metrics, topo=topo,
                     routes=routes, buffers_initial=buffers_initial,
                     attack=attack, detector=detector)


def load_phi(cfg: SimConfig, out_dir: str = "config") -> np.ndarray | None:
    """Baseline key flow per node, from the phase 2 calibration file.

    Returns None when the file is missing. The caller reports that as
    ``phi_source='missing'`` in the summary rather than silently substituting
    zeros, because a missing Phi quietly flattens the prior.
    """
    import os
    path = os.path.join(
        out_dir, f"calibration_{cfg.topology.name}_{cfg.demand.km_mode}.yaml")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    phi = doc.get("phi")
    return np.asarray(phi, dtype=float) if phi else None


def _parse_overrides(pairs: list[str] | None) -> dict:
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


PRINT_KEYS = ("n_offered", "n_admitted", "n_rejected", "n_completed",
              "n_interrupted", "n_background_admitted", "RR", "KPD",
              "mean_path_len", "mean_buffer_util", "mean_active_sessions",
              "total_key_generated", "total_key_consumed",
              "n_compromised", "n_compromised_relays", "exposed_session_ratio",
              "D_eff", "D_eff_relay", "D_eff_endpoint", "D_raw",
              "policy_n_isolation_events", "policy_n_nodes_ever_isolated",
              "policy_n_nodes_ever_vetoed", "policy_n_flaps",
              "auc", "det_rate_tau07", "median_delay_tau07",
              "fpr_node_tau07", "wall_time_s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.runner",
                                 description="Run one QKD SDN simulation.")
    ap.add_argument("--config", required=True, help="path to the YAML config")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override a config field, e.g. --set demand.lam=0.2")
    ap.add_argument("--no-assert", action="store_true",
                    help="skip the runtime invariants (used by sweeps)")
    ap.add_argument("--no-write", action="store_true", help="do not write parquet")
    ap.add_argument("--no-events", action="store_true",
                    help="write only the summary row, not the node event log")
    ap.add_argument("--ablation", action="store_true",
                    help="print the feature subset x AUC table")
    ap.add_argument("--progress", action="store_true", help="show a progress bar")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)

    overrides = _parse_overrides(args.set)
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    cfg = load_config(args.config, overrides)

    result = simulate(cfg, write=not args.no_write, asserts=not args.no_assert,
                      progress=args.progress, events=not args.no_events)
    summary = result.summary

    print(f"run_id           {cfg.run_id}")
    print(f"attack           {cfg.attack.profile if cfg.attack.enabled else 'none'}"
          f"  f={cfg.attack.f}  t_c={cfg.t_compromise:g}  "
          f"selection={cfg.attack.selection}")
    for key in PRINT_KEYS:
        if key not in summary:
            continue
        value = summary[key]
        print(f"{key:22s}{value:,.4f}" if isinstance(value, float)
              else f"{key:22s}{value}")

    if args.ablation and cfg.detector.enabled:
        from .evaluate import ablation_table
        print("\nablation (AUC by feature subset)")
        for name, rep in ablation_table(result.detector, result.state).items():
            print(f"  {name:10s} AUC={rep['auc']:.3f}  "
                  f"det@0.7={rep['det_rate_tau07']:.2f}  "
                  f"FPR@0.7={rep['fpr_node_tau07']:.2f}")

    if "summary_path" in summary:
        print(f"\nwritten          {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
