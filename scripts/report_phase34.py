"""Tables for phases 3 and 4, built from the summary parquets.

    python scripts/report_phase34.py --ablation
    python scripts/report_phase34.py --damage
    python scripts/report_phase34.py --noise --alpha

The ablation table is the gate of phase 4 section 4.7: each profile must be
picked up by its own feature and by no other.  If the diagonal is not there,
either the attack has no effect or the feature cannot see it, and no policy
result built on top would mean anything.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.logging_io import read_summaries  # noqa: E402

SUBSETS = ("x1", "x2", "x3", "x1x2", "x1x3", "x2x3", "x1x2x3")
PROFILES = ("P", "G", "L", "E")
EXPECTED = {"P": None, "G": "x1", "L": "x2", "E": "x3"}


def load(pattern: str) -> pd.DataFrame:
    df = read_summaries(pattern).to_pandas()
    return df[df.get("attack.enabled", False) == True]        # noqa: E712


def ablation(df: pd.DataFrame) -> None:
    cols = [f"abl_{s}_auc" for s in SUBSETS]
    if not set(cols).issubset(df.columns):
        print("no ablation columns; run a sweep with detector.enabled and ablation")
        return
    group_keys = [k for k in ("topology.name", "attack.selection",
                              "detector.use_prior") if k in df.columns]
    for key, block in df.groupby(group_keys):
        tags = dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))
        selection = tags.get("attack.selection", "random")
        prior_on = bool(tags.get("detector.use_prior", True))
        label = "  ".join(f"{k.split('.')[-1]}={v}" for k, v in tags.items())
        print(f"\n=== AUC by feature subset | {label} "
              f"({block['seed_attack'].nunique()} attack seeds, mean) ===")
        print("  features   " + "".join(f"{p:>8s}" for p in PROFILES))
        table = block.groupby("attack.profile")[cols].mean()
        for s, col in zip(SUBSETS, cols):
            row = "".join(f"{table.loc[p, col]:8.3f}" if p in table.index else "     n/a"
                          for p in PROFILES)
            print(f"  {s:10s} {row}")

        print("  diagonal check:")
        for p in PROFILES:
            if p not in table.index:
                continue
            single = {s: table.loc[p, f"abl_{s}_auc"] for s in ("x1", "x2", "x3")}
            best = max(single, key=single.get)
            want = EXPECTED[p]
            if want is None:
                # profile P is undetectable by construction; anything above
                # chance here comes from the prior, not from evidence
                ok = max(single.values()) < 0.65
                note = ("all near chance" if ok else
                        "prior driven, not evidence" if prior_on else
                        "profile P should be undetectable")
            else:
                ok = best == want and single[best] > 0.65
                note = f"best = {best} ({single[best]:.3f}), expected {want}"
            print(f"    {p}: {'OK ' if ok else '** '}{note}")
        if selection == "top_keyflow" and prior_on:
            print("  NOTE: top_keyflow + prior. The compromised nodes ARE the high-Phi\n"
                  "  nodes, so the prior identifies them with no evidence at all - which\n"
                  "  is why even profile P scores high in this block.  It measures the\n"
                  "  prior, not the features.  Never average it with 'random'.")
        elif selection == "random" and prior_on:
            print("  NOTE: random + prior. Here the prior is uncorrelated with the\n"
                  "  compromised set, so it acts as a per node offset that dilutes a\n"
                  "  weak feature instead of helping.  Read feature quality off the\n"
                  "  use_prior=False block; this one is the end to end operating figure.")


def damage(df: pd.DataFrame) -> None:
    """Baseline damage under B0. Never mix topologies or selections in one row -
    D_eff scales with the offered load, so an average across them is meaningless."""
    for key, block in df.groupby(["topology.name", "attack.selection"]):
        print(f"\n=== baseline damage under B0 | topology={key[0]} "
              f"selection={key[1]} (mean over seeds) ===")
        agg = block.groupby(["attack.profile", "attack.f"]).agg(
            D_eff=("D_eff", "mean"), D_raw=("D_raw", "mean"),
            exposed=("exposed_session_ratio", "mean"),
            relays=("n_compromised_relays", "mean"),
            RR=("RR", "mean"), n=("D_eff", "size")).reset_index()
        print(f"  {'prof':>5s}{'f':>7s}{'D_eff':>13s}{'D_raw':>13s}"
              f"{'exposed':>9s}{'relays':>8s}{'RR':>7s}{'runs':>6s}")
        for _, r in agg.iterrows():
            print(f"  {r['attack.profile']:>5s}{r['attack.f']:7.2f}{r['D_eff']:13.3e}"
                  f"{r['D_raw']:13.3e}{r['exposed']:9.3f}{r['relays']:8.1f}"
                  f"{r['RR']:7.3f}{int(r['n']):6d}")
    print("\n  D_eff is the protected data exposed; D_raw the key material, counted\n"
          "  once per session however many compromised nodes its path crosses.\n"
          "  In KM-OTP the two are equal by construction (one key bit per data bit);\n"
          "  they separate in KM-AES, which is the whole reason D_eff is the primary\n"
          "  metric.  These numbers are the denominator of DRR in phase 5.")


def operating_point(df: pd.DataFrame) -> None:
    """How close the score actually gets to the policy thresholds.

    Phase 5 acts at S_iso in {0.5, 0.7, 0.9}.  If the score a real attack
    produces never reaches those, no policy lever ever fires and every phase 5
    comparison collapses to B0.
    """
    cols = [c for c in df.columns if c.startswith("det_rate_tau")]
    if not cols or "attack.profile" not in df.columns:
        return
    print("\n=== operating point: fraction of compromised nodes reaching S_iso ===")
    sub = df[df.get("detector.use_prior", True) == True]           # noqa: E712
    t = sub.groupby("attack.profile")[sorted(cols)].mean().round(3)
    print(t.to_string())
    print("  With uniform weights w=(1/3,1/3,1/3) and theta_0=0.35, a profile that\n"
          "  triggers exactly ONE of three features - which is what P/G/L/E are\n"
          "  designed to do - reaches at most S = sigmoid(6*(1/3 - 0.35)) ~ 0.60.\n"
          "  So S_iso = 0.7 and 0.9 are structurally unreachable for a single\n"
          "  feature attack, and only S_iso = 0.5 can fire.  Decide in phase 5\n"
          "  whether to lower theta_0, reweight, or keep S_iso at 0.5.")


def noise(df: pd.DataFrame) -> None:
    if "noise.scale" not in df.columns or df["noise.scale"].nunique() < 2:
        print("\nno noise sweep in these results")
        return
    print("\n=== AUC against observation noise (mandatory axis, section 3.3) ===")
    t = df.pivot_table(index="noise.scale", columns="attack.profile",
                       values="auc", aggfunc="mean")
    print(t.round(3).to_string())
    print("  An AUC that does not fall as the noise rises means the noise model\n"
          "  is not reaching the features.")


def alpha(df: pd.DataFrame) -> None:
    if "detector.alpha" not in df.columns or df["detector.alpha"].nunique() < 2:
        print("\nno alpha sweep in these results")
        return
    print("\n=== detection delay against false positives, by EWMA alpha ===")
    t = df.groupby(["detector.alpha", "attack.profile"]).agg(
        auc=("auc", "mean"),
        det_rate=("det_rate_tau07", "mean"),
        delay_s=("median_delay_tau07", "mean"),
        fpr_node=("fpr_node_tau07", "mean")).round(3)
    print(t.to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/raw/*.parquet")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--damage", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--alpha", action="store_true")
    ap.add_argument("--operating", action="store_true")
    args = ap.parse_args()
    if not any((args.ablation, args.damage, args.noise, args.alpha)):
        args.ablation = args.damage = args.operating = True

    df = load(args.pattern)
    print(f"{len(df)} runs with an attack enabled")
    if args.ablation:
        ablation(df)
    if args.damage:
        damage(df)
    if args.noise:
        noise(df)
    if args.alpha:
        alpha(df)
    if args.operating:
        operating_point(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
