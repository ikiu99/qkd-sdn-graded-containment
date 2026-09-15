"""Load calibration and baseline key flow.

Phase 2 reference: section 4.11 of phase2-build-spec.

Two products, both written to config/calibration_{topology}_{km_mode}.yaml so
that phases 3-5 read them instead of re-deriving them:

  lambda   the arrival rate that puts B0 without attack at ~1 / 10 / 30 percent
           rejection (phase 1, section 3: low / medium / saturated load)
  Phi_i    normalised key flow through each node in the baseline, the prior
           component of phase 1, section 7.  Deliberately not degree centrality.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np
import yaml

from .config import SimConfig, load_config, make_run_id
from .runner import simulate

LAM_CAP = 200.0          # refuse to search beyond this arrival rate
MAX_BISECT = 18


def _with_lam(cfg: SimConfig, lam: float) -> SimConfig:
    demand = dataclasses.replace(cfg.demand, lam=float(lam))
    out = dataclasses.replace(cfg, demand=demand, run_id="")
    return dataclasses.replace(out, run_id=make_run_id(out))


def rejection_rate(cfg: SimConfig, lam: float, verbose: bool = False,
                   cache: dict | None = None) -> float:
    """RR of one B0 run at this lambda. ``cache`` avoids repeating the bracket
    runs across the three targets - the runs are deterministic, so a hit is
    exact, not an approximation."""
    if cache is not None and lam in cache:
        return cache[lam]
    res = simulate(_with_lam(cfg, lam), write=False, asserts=False)
    rr = float(res.summary["RR"])
    if verbose:
        print(f"    lam={lam:9.4f}  RR={rr:6.3f}  offered={res.summary['n_offered']:7d} "
              f"admitted={res.summary['n_admitted']:7d} "
              f"interrupted={res.summary['n_interrupted']:6d} "
              f"buf={res.summary['mean_buffer_util']:.3f} "
              f"({res.summary['wall_time_s']:.1f}s)")
    if cache is not None:
        cache[lam] = rr
    return rr


def calibrate_lambda(cfg_base: SimConfig, targets=(0.01, 0.10, 0.30),
                     tol: float = 0.1, verbose: bool = True) -> dict[str, float]:
    """Bisect on lambda until the B0 rejection rate hits each target.

    ``tol`` is relative: the search stops once |RR - target| <= tol * target.
    Names follow phase 1: low / medium / saturated.
    """
    names = ("low", "medium", "saturated")
    out: dict[str, float] = {}
    achieved: dict[str, float] = {}
    cache: dict[float, float] = {}

    for name, target in zip(names, targets):
        if verbose:
            print(f"  target RR = {target:.0%}")
        lo, rr_lo = 0.0, 0.0
        hi = max(cfg_base.demand.lam, 1e-3)
        rr_hi = rejection_rate(cfg_base, hi, verbose, cache)

        # bracket from above
        while rr_hi < target and hi < LAM_CAP:
            lo, rr_lo = hi, rr_hi
            hi = min(hi * 2.0, LAM_CAP)
            rr_hi = rejection_rate(cfg_base, hi, verbose, cache)
        if rr_hi < target:
            print(f"    ! RR={rr_hi:.3f} at the lambda cap {LAM_CAP}: target "
                  f"{target:.0%} is unreachable in this configuration")
            out[name], achieved[name] = hi, rr_hi
            continue

        best, best_rr = hi, rr_hi
        for _ in range(MAX_BISECT):
            if abs(best_rr - target) <= tol * target:
                break
            mid = 0.5 * (lo + hi)
            rr = rejection_rate(cfg_base, mid, verbose, cache)
            if abs(rr - target) < abs(best_rr - target):
                best, best_rr = mid, rr
            if rr < target:
                lo, rr_lo = mid, rr
            else:
                hi, rr_hi = mid, rr
        out[name], achieved[name] = round(best, 6), round(best_rr, 4)
        if verbose:
            print(f"    -> lam={out[name]}  RR={achieved[name]}")

    out["_achieved_RR"] = achieved
    return out


def compute_key_flow(cfg_base: SimConfig, normalise: bool = True) -> np.ndarray:
    """One B0 pilot run without attack; cumulative key flow per node.

    Returns Phi_i, min-max normalised to [0,1] over the nodes (phase 1, table in
    section 7).
    """
    res = simulate(cfg_base, write=False, asserts=False)
    flow = res.metrics.node_key_flow(res.state)
    if not normalise:
        return flow
    lo, hi = float(flow.min()), float(flow.max())
    if hi - lo < 1e-12:
        return np.zeros_like(flow)
    return (flow - lo) / (hi - lo)


def logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def measure_evidence_floor(cfg: SimConfig, phi: np.ndarray,
                           q: float = 0.95) -> np.ndarray:
    """Per feature evidence level of a HEALTHY node, from one clean pilot run.

    The operating point has to be set against this, not against e = 0.  A
    healthy node does not sit at zero evidence: report noise, the N4 background
    stream and window-straddling sessions put e_k at a floor of roughly 0.1-0.2,
    and calibrating as if it were zero puts every threshold a whole noise floor
    too low - which is exactly how the first attempt produced false positives at
    S_iso = 0.7 on a run with no attack at all.
    """
    det = dataclasses.replace(cfg.detector, enabled=True, ablation=False,
                              use_prior=False, keep_evidence=True)
    attack = dataclasses.replace(cfg.attack, enabled=False)
    pilot = dataclasses.replace(cfg, detector=det, attack=attack, run_id="")
    pilot = dataclasses.replace(pilot, run_id=make_run_id(pilot))

    res = simulate(pilot, write=False, asserts=False)
    if not res.detector.hist_e:
        return np.zeros(3, dtype=float)
    e = np.stack(res.detector.hist_e)            # (samples, nodes, 3)
    # Per node time average first, then the upper quantile ACROSS nodes.  The
    # score is smoothed by an EWMA, so what drives a false positive is a node
    # that looks bad persistently, not a single unlucky sample.  Taking the
    # quantile over raw samples instead would pick up the 5% tail that the dual
    # baseline produces by construction on every run, and would push the
    # operating point far too high.
    return np.quantile(e.mean(axis=0), q, axis=0)


def calibrate_score(cfg: SimConfig, topo, phi: np.ndarray,
                    e_floor: np.ndarray | None = None,
                    s_lo: float = 0.45, s_hi: float = 0.70,
                    margin: float = 1.15) -> dict:
    """Solve (lam_s, theta_0) in closed form so the score has a usable range.

    The score is S = sigmoid(lam_s*(sum_k w_k e_k - theta_0) + logit(pi_i)).  The
    prior's logit spans ~3 units across nodes while a single saturated feature
    under uniform weights moves the evidence term by only lam_s/3, so with the
    spec's lam_s=6, theta_0=0.35 the prior dominates: a median-prior node with
    one feature fully triggered reaches S=0.43 while a healthy high-prior node
    with no evidence at all reaches 0.45.  Every policy threshold then fires on
    the prior rather than on evidence.

    Two operating-point targets pin it down, and which ones are chosen is the
    whole question - the obvious "worst case at both ends" pair is infeasible
    together with an informative prior:

      (a) quiet network, HIGHEST prior   -> S < s_lo   (never isolate without
                                                         evidence)
      (b) one feature saturated,
          MEDIAN prior                   -> S > s_hi   (evidence can isolate)

    "Quiet" means the measured healthy-node evidence floor ``e_floor``, not
    zero.  With ``ew_bg = w . e_floor`` and ``ew_hit = w_j + sum_{k!=j} w_k
    e_floor_k`` for the weakest-weighted feature j, (a) gives
    ``lam*(theta_0 - ew_bg) > L_max - logit(s_lo)`` and (b) gives
    ``lam*(ew_hit - theta_0) > logit(s_hi) - L_med``.  Adding them bounds lam_s
    from below by the *separation* ``ew_hit - ew_bg``; theta_0 then sits at the
    midpoint of its feasible interval so both targets hold with equal slack.

    The prior keeps its role: it still spreads S over roughly [0.05, 0.4] on a
    quiet network, which is exactly what the kappa routing-weight lever
    consumes, while never on its own reaching an isolation threshold.  That is
    the paper's claim about residual protection on the undetectable profile P,
    made quantitative.
    """
    from .detector import build_prior

    prior = build_prior(topo, cfg.detector, phi, cfg.seed_topology)
    L = np.log(prior / (1.0 - prior))
    L_max, L_med, L_min = float(L.max()), float(np.median(L)), float(L.min())

    w = np.asarray(cfg.detector.weights, dtype=float)
    w = w / w.sum()
    e_bg = (np.zeros(3) if e_floor is None else np.asarray(e_floor, dtype=float))
    ew_bg = float(w @ e_bg)
    # Weakest ACTIVE feature: the hardest single-feature attack to catch.  A
    # weight of zero means the feature is switched off, not that it is the
    # hardest case - taking argmin over all three would make every ablation
    # subset infeasible.
    active = np.flatnonzero(w > 1e-12)
    j = int(active[np.argmin(w[active])])
    ew_hit = float(w[j] * 1.0 + (w @ e_bg - w[j] * e_bg[j]))
    sep = ew_hit - ew_bg
    if sep <= 1e-9:
        raise RuntimeError("no separation between the quiet and triggered "
                           "operating points; the evidence floor saturates")

    need = (logit(s_hi) - logit(s_lo) + L_max - L_med) / sep
    lam = float(np.ceil(margin * need * 2.0) / 2.0)          # round up to 0.5
    lo = ew_bg + (L_max - logit(s_lo)) / lam
    hi = ew_hit - (logit(s_hi) - L_med) / lam
    if hi <= lo:
        raise RuntimeError("score targets are infeasible; widen [s_lo, s_hi] "
                           "or shrink the prior coefficients c1..c3")
    theta_0 = round(0.5 * (lo + hi), 4)

    def S(ew, logit_prior):
        return float(1.0 / (1.0 + np.exp(-(lam * (ew - theta_0) + logit_prior))))

    return {
        "lam_s": lam,
        "theta_0": theta_0,
        "targets": {"s_lo": s_lo, "s_hi": s_hi},
        "evidence_floor": [round(float(x), 4) for x in e_bg],
        "operating": {"quiet": round(ew_bg, 4), "one_feature": round(ew_hit, 4)},
        "prior_logit": {"min": round(L_min, 4), "median": round(L_med, 4),
                        "max": round(L_max, 4)},
        "check": {
            "quiet_max_prior": round(S(ew_bg, L_max), 4),        # must be < s_lo
            "quiet_med_prior": round(S(ew_bg, L_med), 4),
            "quiet_min_prior": round(S(ew_bg, L_min), 4),
            "one_feature_med_prior": round(S(ew_hit, L_med), 4),  # must be > s_hi
            "one_feature_min_prior": round(S(ew_hit, L_min), 4),
            "two_features_med_prior": round(S(min(1.0, 2 * ew_hit), L_med), 4),
        },
    }


def out_path(cfg: SimConfig, out_dir: str = "config") -> str:
    return os.path.join(out_dir,
                        f"calibration_{cfg.topology.name}_{cfg.demand.km_mode}.yaml")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.calibrate",
                                 description="Calibrate lambda and compute Phi_i.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--horizon", type=int, default=21600,
                    help="calibration horizon in seconds (default: the 6 h sweep horizon)")
    ap.add_argument("--targets", default="0.01,0.10,0.30")
    ap.add_argument("--tol", type=float, default=0.1, help="relative tolerance on RR")
    ap.add_argument("--flow-lambda", type=float, default=None,
                    help="lambda for the Phi_i pilot run (default: the calibrated "
                         "medium load)")
    ap.add_argument("--out-dir", default="config")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg = dataclasses.replace(cfg, horizon=int(args.horizon))
    targets = tuple(float(x) for x in args.targets.split(","))
    verbose = not args.quiet

    print(f"calibrating {cfg.topology.name} / {cfg.demand.km_mode} "
          f"over {cfg.horizon}s (warmup {cfg.warmup}s)")
    lam = calibrate_lambda(cfg, targets, tol=args.tol, verbose=verbose)
    achieved = lam.pop("_achieved_RR")

    flow_lam = args.flow_lambda if args.flow_lambda is not None else lam["medium"]
    print(f"pilot run for Phi_i at lam={flow_lam}")
    phi = compute_key_flow(_with_lam(cfg, flow_lam))

    from .topology import load_topology
    topo = load_topology(cfg.topology, cfg.noise.qber_base_min, cfg.noise.qber_base_max)
    print("measuring the healthy-node evidence floor...")
    e_floor = measure_evidence_floor(_with_lam(cfg, flow_lam), phi)
    print(f"    e_floor (95th pct) = {e_floor.round(4).tolist()}")
    score = calibrate_score(cfg, topo, phi, e_floor)
    print(f"score operating point: lam_s={score['lam_s']} theta_0={score['theta_0']}")
    for k, v in score["check"].items():
        print(f"    S({k:24s}) = {v}")

    doc = {
        "topology": cfg.topology.name,
        "km_mode": cfg.demand.km_mode,
        "horizon": cfg.horizon,
        "warmup": cfg.warmup,
        "seed_demand": cfg.seed_demand,
        "targets": {n: float(t) for n, t in zip(("low", "medium", "saturated"), targets)},
        "lam": {k: float(v) for k, v in lam.items()},
        "achieved_RR": {k: float(v) for k, v in achieved.items()},
        "phi_lambda": float(flow_lam),
        "phi": [round(float(x), 6) for x in phi],
        # Recorded, not applied: the authoritative values live in the run configs
        # so that a sweep axis on theta_0 or lam_s is never silently overridden.
        "score": score,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = out_path(cfg, args.out_dir)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
    print(f"written {path}")
    print(f"  lam      {doc['lam']}")
    print(f"  achieved {doc['achieved_RR']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
