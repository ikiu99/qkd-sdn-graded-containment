"""Build a parametric family of 50-node topologies and calibrate each one.

    python scripts/make_family.py

Three named topologies with one sample each answer "did it work on nsfnet" and
nothing more. Connectivity is the single structural property the whole response
argument depends on - B2 needs node-disjoint paths to exist, the partition guard
needs the graph to survive an isolation, and the cost of both is set by how much
longer the second path is than the first - so it deserves to be an axis rather
than three anecdotes.

The family spans mean degree 2.2 to 4.4 with three independent samples at each
point. Every member is biconnected by construction, exactly as ``net50`` is, so
**every** member has two node-disjoint paths between every pair (Menger) and the
interesting quantities are the *third* path and the *cost* of the second:

    frac_ge_3_disjoint   the ceiling on B2 with m=3
    detour ratio         mean(second path length) / mean(shortest path length)

Each member is calibrated on its own, because none of this is comparable
otherwise:

* ``demand.lam`` is bisected to put B0 at 10 % rejection. A denser graph carries
  more traffic at the same lambda, so holding lambda fixed would compare a lightly
  loaded dense network against a saturated sparse one and call the difference
  connectivity.
* ``phi`` (the key-flow prior input) is re-measured, since it is a property of
  the routing.
* ``(lam_s, theta_0)`` are re-solved against the member's own evidence floor,
  because the floor depends on how many neighbours a node has to be compared
  against - the peer branch of the dual baseline is a cohort statistic.

Calibration is the expensive part (a bisection plus two pilot runs per member),
so it is done once here and written into the sweep as literal values.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_topologies import (biconnect, connectivity_stats, rescale,   # noqa: E402
                              _exposure, _qber_unit)
from sim.calibrate import (calibrate_lambda, calibrate_score,           # noqa: E402
                           compute_key_flow, measure_evidence_floor)
from sim.config import load_config                                      # noqa: E402
from sim.runner import simulate                                         # noqa: E402
from sim.topology import load_topology                                  # noqa: E402

N = 50
DEGREES = (2.2, 2.8, 3.6, 4.4)
SAMPLES = 3
TOPO_DIR = "data/topologies/family"
BASE = "config/attack_t3_otp.yaml"
OUT = "config/sweep_family.yaml"
SEEDS = list(range(3, 3 + 20 * 10, 10))


def build_member(name: str, n: int, target_edges: int, seed: int) -> dict:
    """Same generator as ``net50``: Euclidean MST, shortest fill, biconnect."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))

    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = d[0].copy()
    parent = np.zeros(n, dtype=int)
    mst: list[tuple[int, int]] = []
    for _ in range(n - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        mst.append((min(j, int(parent[j])), max(j, int(parent[j]))))
        in_tree[j] = True
        closer = d[j] < best
        parent[closer] = j
        best = np.minimum(best, d[j])

    chosen = set(mst)
    for _, u, v in sorted((float(d[u, v]), u, v) for u in range(n)
                          for v in range(u + 1, n) if (u, v) not in mst):
        if len(chosen) >= target_edges:
            break
        chosen.add((u, v))
    chosen, n_added, n_trimmed = biconnect(chosen, d, n, target_edges)

    pairs = sorted(chosen)
    lens = rescale([float(d[u, v]) for u, v in pairs])
    return {
        "name": name,
        "nodes": list(range(n)),
        "edges": [[u, v, L] for (u, v), L in zip(pairs, lens)],
        "coords": [[round(float(x), 5), round(float(y), 5)] for x, y in pos],
        "exposure": _exposure(n, rng),
        "qber_unit": _qber_unit(len(pairs)),
        "meta": {
            "generator": f"family member, euclidean MST + fill to {target_edges}"
                         f" edges, biconnect (+{n_added}, -{n_trimmed}), "
                         f"seed_topology={seed}",
            "length_model": "linear rescale of unit-square distance into "
                            "[8.7, 52.1] km",
            **connectivity_stats(pairs, n),
        },
    }


def calibrate(name: str, path: str) -> dict:
    """lambda at 10 % rejection, phi, evidence floor, and (lam_s, theta_0)."""
    cfg = load_config(BASE, {"topology.name": name, "topology.path": path,
                             "attack.enabled": False, "policy.type": "B0",
                             "horizon": 21600, "warmup": 1800, "seed_demand": 2})
    lam = calibrate_lambda(cfg, targets=(0.10,), verbose=False)["low"]
    cfg = dataclasses.replace(
        cfg, demand=dataclasses.replace(cfg.demand, lam=lam))
    phi = compute_key_flow(cfg)
    floor = measure_evidence_floor(cfg, phi)
    topo = load_topology(cfg.topology, cfg.noise.qber_base_min,
                         cfg.noise.qber_base_max)
    sc = calibrate_score(cfg, topo, phi, floor)

    # The throttle anchor is a property of the member, not a constant. A denser
    # graph has a different quiet score, and pinning every member to net50's
    # would mean rho_start throttles the whole of a sparse member and none of a
    # dense one - the lever would stop being comparable across the very axis the
    # sweep exists to vary.
    quiet_cfg = dataclasses.replace(
        cfg, detector=dataclasses.replace(cfg.detector, enabled=True,
                                          ablation=False, lam_s=sc["lam_s"],
                                          theta_0=sc["theta_0"]))
    res = simulate(dataclasses.replace(quiet_cfg, run_id=""), write=False,
                   asserts=False)
    _, sb = res.detector.score_history()
    quiet = float(np.mean(sb[:, res.detector.main_index, :]))
    return {"lam": round(float(lam), 6), "phi": [round(float(v), 5) for v in phi],
            "e_floor": [round(float(v), 5) for v in floor],
            "lam_s": sc["lam_s"], "theta_0": sc["theta_0"],
            "quiet": round(quiet, 4)}


def main() -> int:
    os.makedirs(TOPO_DIR, exist_ok=True)
    variants, rows = [], []
    for deg in DEGREES:
        target = int(round(N * deg / 2))
        for k in range(SAMPLES):
            name = f"fam{str(deg).replace('.', '')}s{k}"
            path = f"{TOPO_DIR}/{name}.json"
            doc = build_member(name, N, target, seed=1000 + int(deg * 10) * 10 + k)
            pathlib.Path(path).write_text(
                json.dumps(doc, indent=1) + "\n", encoding="utf-8")
            c = calibrate(name, path)
            m = doc["meta"]
            pathlib.Path(f"config/calibration_{name}_OTP.yaml").write_text(
                yaml.safe_dump({"topology": name, "km_mode": "OTP",
                                "horizon": 21600, "warmup": 1800,
                                "seed_demand": 2, "lam": {"medium": c["lam"]},
                                "phi_lambda": c["lam"], "phi": c["phi"]},
                               sort_keys=False, default_flow_style=None),
                encoding="utf-8")
            rows.append((name, deg, len(doc["edges"]), m["mean_degree"],
                         m["frac_ge_2_disjoint"], m["frac_ge_3_disjoint"],
                         c["lam"], c["lam_s"], c["theta_0"]))
            print(f"  {name:<12} edges={len(doc['edges']):3d} "
                  f"deg={m['mean_degree']:.2f}  >=2 {m['frac_ge_2_disjoint']:.1%}"
                  f"  >=3 {m['frac_ge_3_disjoint']:5.1%}  lam={c['lam']:.4f}  "
                  f"lam_s={c['lam_s']:.1f}  theta_0={c['theta_0']:.4f}  "
                  f"quiet={c['quiet']:.4f}")
            common = {"topology.name": name, "topology.path": path,
                      "demand.lam": c["lam"], "detector.lam_s": c["lam_s"],
                      "detector.theta_0": c["theta_0"]}
            for pol in ({"policy.type": "B0", "attack.enabled": True},
                        {"policy.type": "B0", "attack.enabled": False},
                        {"policy.type": "B1", "policy.tau": 0.5},
                        {"policy.type": "B2", "policy.m_paths": 2},
                        {"policy.type": "B2", "policy.m_paths": 3},
                        {"policy.type": "B3", "policy.S_iso": 0.5,
                         "policy.rho_start": c["quiet"], "policy.kappa": 0.0},
                        {"policy.type": "B4"}):
                variants.append({**common, **pol})

    doc = {"base": BASE, "detector": {"ablation": False},
           "output_dir": "results/raw", "variants": variants,
           "axes": {"attack.profile": ["G", "E"], "seed_attack": SEEDS}}
    header = ("# Connectivity as an axis: mean degree 2.2 - 4.4, 3 samples each.\n"
              "#\n# GENERATED by scripts/make_family.py - edit that, not this.\n"
              "# Every member is calibrated on its own (lambda at 10% rejection,\n"
              "# its own phi, its own (lam_s, theta_0)); holding any of the three\n"
              "# fixed across the family would confound density with load.\n"
              "#\n#   python -m sim.sweep --sweep " + OUT + " --no-assert\n")
    pathlib.Path(OUT).write_text(
        header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None,
                                width=110), encoding="utf-8")
    print(f"\nwrote {OUT}: {len(variants)} x 2 x {len(SEEDS)} = "
          f"{len(variants) * 2 * len(SEEDS)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
