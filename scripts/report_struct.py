"""The three structural changes, at 20 seeds.

    python scripts/report_struct.py

Reports, in order:

1.  B8, risk-triggered multipath, against the two policies it is built from.
    The question is not whether it beats them on damage - B2 is unbeatable on
    damage and ruinous on key - but whether it buys most of B2's protection for
    a fraction of B2's key.
2.  The routing lever measured ALONE, with the quota lever turned nearly off.
    This is the experiment behind the "kappa is inert" claim, which was only
    ever measured in the presence of rho; rho acts on the same node and
    dominates it, so the earlier reading confounded the two.
3.  route_mode=logrisk against the phase 1 exp form at matched kappa.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analyze import attach_baselines, bootstrap_ci, load  # noqa: E402

PROFILES = ("P", "G", "L", "E")


def cell(d: pd.DataFrame) -> pd.DataFrame:
    """The central cell, with every axis this sweep does not vary pinned."""
    m = ((d["topology.name"] == "net50") & (d["demand.km_mode"] == "OTP")
         & np.isclose(d["attack.f"], 0.10)
         & np.isclose(d["noise.scale"], 1.0)
         & np.isclose(d["detector.alpha"], 0.05)
         & np.isclose(d["demand.T_s_max"], 600.0)
         & (d["attack.selection"] == "random")
         & (d["detector.x1_mode"] == "signed")
         & (d["detector.x2_mode"] == "signed")
         & np.isclose(d["attack.gamma"], 0.30)
         & np.isclose(d["attack.delta"], 0.50)
         & np.isclose(d["attack.q"], 0.03)
         & np.isclose(d["attack.s"], 0.25))
    return d[m]


def pick(d, pol, **kw):
    d = d[d["policy.type"] == pol]
    for k, v in kw.items():
        if k in d.columns:
            d = d[np.isclose(d[k].astype(float), v)]
    return d


def report_hybrid(d: pd.DataFrame) -> None:
    rows = [("B2 m=2 always", "B2", {"policy.m_paths": 2}),
            ("B3 graded only", "B3", {"policy.S_iso": 0.5,
                                      "policy.rho_start": 0.1316,
                                      "policy.kappa": 0.0}),
            ("B8 hybrid tau=0.70", "B8", {"policy.hybrid_tau": 0.70}),
            ("B8 hybrid tau=0.50", "B8", {"policy.hybrid_tau": 0.50}),
            ("B8 hybrid tau=0.35", "B8", {"policy.hybrid_tau": 0.35}),
            ("B8 hybrid tau=0.20", "B8", {"policy.hybrid_tau": 0.20}),
            ("B4 oracle", "B4", {})]
    print("\n=== 1. risk-triggered multipath (B8) ===")
    print("    DRR_relay [95% CI] / dRR / dKPD / mean legs\n")
    for prof in PROFILES:
        b = d[d["attack.profile"] == prof]
        if b.empty:
            continue
        print(f"  profile {prof}")
        print(f"    {'policy':<22}{'DRR_relay':>22}{'dRR':>8}{'dKPD':>8}{'legs':>7}")
        for lab, pol, kw in rows:
            r = pick(b, pol, **kw)
            if r.empty:
                continue
            m, lo, hi = bootstrap_ci(r["drr_relay"].to_numpy())
            print(f"    {lab:<22}{m:>8.3f} [{lo:6.3f},{hi:6.3f}]"
                  f"{r['dRR'].mean():>8.3f}{r['dKPD'].mean():>8.3f}"
                  f"{r['mean_legs'].mean():>7.2f}")
        print()

    # the summary that decides whether the hybrid is worth having
    print("  pooled over the four profiles - protection bought per unit of key:")
    print(f"    {'policy':<22}{'mean DRR':>10}{'mean dKPD':>11}{'DRR/dKPD':>10}")
    for lab, pol, kw in rows:
        r = pick(d, pol, **kw)
        if r.empty:
            continue
        drr, k = r["drr_relay"].mean(), r["dKPD"].mean()
        ratio = drr / k if k > 0.02 else float("inf")
        print(f"    {lab:<22}{drr:>10.3f}{k:>11.3f}"
              f"{ratio:>10.2f}" if np.isfinite(ratio)
              else f"    {lab:<22}{drr:>10.3f}{k:>11.3f}{'free':>10}")
    print("\n    B2 is unbeatable on damage and ruinous on key; B3 is free and")
    print("    blind on half the threat model.  The hybrid is worth having only")
    print("    if it is close to B2 on damage at a fraction of B2's key.")


def report_routing(d: pd.DataFrame) -> None:
    """The routing lever with the quota lever nearly off."""
    lone = d[np.isclose(d["policy.S_iso"], 0.9)
             & np.isclose(d["policy.rho_start"], 0.8999)]
    if lone.empty:
        print("\n  routing-alone runs not on disk")
        return
    print("\n=== 2 and 3. the routing lever measured ALONE ===")
    print("    quota lever nearly off (rho_start just below S_iso), so anything")
    print("    that moves here is the routing weight and nothing else.\n")
    print(f"    {'mode':<10}{'kappa':>7}{'DRR_relay':>22}{'dRR':>8}{'hops':>7}")
    for mode in ("exp", "logrisk"):
        sub = lone[lone["policy.route_mode"] == mode] \
            if "policy.route_mode" in lone.columns else lone
        for k in sorted(sub["policy.kappa"].dropna().unique()):
            r = sub[np.isclose(sub["policy.kappa"], k)]
            if r.empty:
                continue
            m, lo, hi = bootstrap_ci(r["drr_relay"].to_numpy())
            print(f"    {mode:<10}{k:>7.0f}{m:>8.3f} [{lo:6.3f},{hi:6.3f}]"
                  f"{r['dRR'].mean():>8.3f}{r['mean_leg_len'].mean():>7.2f}")
    print("\n    The earlier 'kappa is inert' reading was taken with rho active.")
    print("    rho acts on the same node and is the stronger instrument, so it")
    print("    had already rejected the demands kappa would have re-routed.")


def main() -> int:
    d = attach_baselines(load("results/raw/*.parquet"))
    d = cell(d)
    print(f"{len(d)} runs in the central cell")
    report_hybrid(d)
    report_routing(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
