"""Phase 6 analysis: DRR, PSI, the composite cost, and the Pareto frontier.

    python scripts/analyze.py --pareto --drr --psi --cost

All three of DRR, PSI and C are *cross-run* quantities: each needs a baseline
run to divide by. Two joins do that, each on a single hash column written into
every summary row by ``config.scenario_id`` / ``config.base_scenario_id``:

    DRR  scenario_id       everything except policy.*   -> the matched B0 run
    PSI  base_scenario_id also except attack.*         -> matched B0, no attack

Joining on "every column except policy.*" instead would be silently fragile:
adding one config field later changes the key set and invalidates every
previously computed number.

Two things the numbers need stating with:

* **DRR is reported on relay exposure.**  A session whose own endpoint is
  compromised leaks however it is routed, so that share of D_eff is a floor set
  by the traffic matrix rather than by the policy. Mixing it in caps DRR at a
  value no policy can influence. ``drr_total`` is reported alongside for
  completeness.
* **Ratios are paired, then averaged.**  Each seed contributes its own
  DRR = 1 - D_policy/D_B0 and the mean is taken over those, with a bootstrap CI.
  Dividing mean by mean is a different estimator and a worse one here.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.logging_io import read_summaries  # noqa: E402

POLICY_ORDER = ["B0", "B1", "B2", "B3", "B4"]


def load(pattern: str) -> pd.DataFrame:
    df = read_summaries(pattern).to_pandas()
    for col in ("scenario_id", "base_scenario_id", "policy.type"):
        if col not in df.columns:
            raise SystemExit(f"summaries lack '{col}'; re-run the sweep with the "
                             "current code")
    return df


def attach_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add the matched B0 and B0-no-attack columns, then DRR and PSI."""
    df = df.copy()

    b0 = df[df["policy.type"].isin(("B0", "none"))]
    dup = b0["scenario_id"].duplicated().sum()
    if dup:
        print(f"  warning: {dup} scenarios have more than one B0 row; taking the first")
    ref = b0.drop_duplicates("scenario_id").set_index("scenario_id")

    for src, dst in (("D_eff_relay", "D_eff_relay_b0"), ("D_eff", "D_eff_b0"),
                     ("RR", "RR_b0"), ("KPD", "KPD_b0")):
        df[dst] = df["scenario_id"].map(ref[src]) if src in ref.columns else np.nan

    clean = df[(df["policy.type"].isin(("B0", "none")))
               & (~df.get("attack.enabled", False).astype(bool))]
    cref = clean.drop_duplicates("base_scenario_id").set_index("base_scenario_id")
    col = "mean_leg_len" if "mean_leg_len" in cref.columns else "mean_path_len"
    df["leg_len_clean"] = df["base_scenario_id"].map(cref[col])

    # DRR - guarded. A zero denominator happens whenever the attack is off, f=0,
    # or every compromised node turned out never to relay; NaN there is honest,
    # a silent 0/0 dropped by a plotting library is not.
    for name, num, den in (("drr_relay", "D_eff_relay", "D_eff_relay_b0"),
                           ("drr_total", "D_eff", "D_eff_b0")):
        ok = df[den].to_numpy(dtype=float) > 0
        df[name] = np.where(ok, 1.0 - df[num] / df[den].replace(0, np.nan), np.nan)

    # PSI - relative stretch of ONE leg. The m-fold replication of B2 is a key
    # cost and already shows up in KPD and RR; folding it in here as well would
    # correlate the two Pareto axes.
    ref_len = df["leg_len_clean"].to_numpy(dtype=float)
    df["PSI"] = np.where(ref_len > 0, df[col] / df["leg_len_clean"] - 1.0, np.nan)
    df["dRR"] = df["RR"] - df["RR_b0"]
    df["dKPD"] = np.where(df["KPD_b0"] > 0, df["KPD"] / df["KPD_b0"] - 1.0, np.nan)
    return df


def bootstrap_ci(v: np.ndarray, n: int = 2000, seed: int = 0):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def composite(df: pd.DataFrame, a=(1 / 3, 1 / 3, 1 / 3)) -> pd.Series:
    """C = a1*dRR + a2*dKPD + a3*PSI, on comparable scales.

    All three are already relative to the matched baseline, so they share units
    of "fractional degradation" and the weights mean something. C is secondary
    to the Pareto frontier: it collapses three axes onto one with a choice of a
    that nobody can justify, which is exactly why ``--cost`` also reports how
    much the ranking moves as a varies over the simplex.
    """
    return (a[0] * df["dRR"].fillna(0.0)
            + a[1] * df["dKPD"].fillna(0.0)
            + a[2] * df["PSI"].fillna(0.0))


def current(df: pd.DataFrame) -> pd.DataFrame:
    """Drop superseded feature variants before reporting.

    ``x1_mode`` and ``x2_mode`` are hashed into the run id, so the absolute-value
    forms of both features - the literal phase 1 formulas, kept as sweep points -
    sit in results/raw alongside the signed ones instead of overwriting them.
    That is deliberate and it is what makes the before/after comparison possible,
    but it means every report that is NOT about the feature form has to say which
    form it is reporting, or it silently averages two different detectors.
    """
    for col in ("detector.x1_mode", "detector.x2_mode"):
        if col in df.columns:
            df = df[df[col] == "signed"]
    return df


def report_drr(df: pd.DataFrame) -> None:
    print("\n=== damage reduction, paired per seed then averaged (95% bootstrap CI) ===")
    keys = [k for k in ("topology.name", "demand.km_mode", "attack.profile",
                        "attack.f", "attack.selection") if k in df.columns]
    sub = df[~df["policy.type"].isin(("B0", "none"))]
    if sub.empty:
        print("  no non-B0 runs found")
        return
    for key, block in sub.groupby(keys):
        tags = "  ".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, key))
        print(f"\n  {tags}")
        print(f"    {'policy':<10}{'DRR_relay':>22}{'DRR_total':>12}"
              f"{'dRR':>9}{'dKPD':>9}{'PSI':>8}{'n':>5}")
        for pol, rows in block.groupby("policy.type"):
            m, lo, hi = bootstrap_ci(rows["drr_relay"].to_numpy())
            label = pol
            if pol == "B2" and "policy.m_paths" in rows.columns:
                label = f"B2(m={int(rows['policy.m_paths'].iloc[0])})"
            print(f"    {label:<10}{m:>8.3f} [{lo:6.3f},{hi:6.3f}]"
                  f"{rows['drr_total'].mean():>12.3f}{rows['dRR'].mean():>9.3f}"
                  f"{rows['dKPD'].mean():>9.3f}{rows['PSI'].mean():>8.3f}{len(rows):>5}")


def report_pareto(df: pd.DataFrame) -> None:
    """Security against cost. The headline figure of the paper."""
    print("\n=== Pareto frontier: DRR_relay against cost ===")
    sub = df[df["drr_relay"].notna()]
    if sub.empty:
        print("  nothing to plot")
        return
    rows = []
    gkeys = ["policy.type", "policy.kappa", "policy.S_iso", "policy.m_paths"]
    if "policy.rho_start" in sub.columns:
        gkeys.append("policy.rho_start")
    for key, block in sub.groupby(gkeys):
        pol, kappa, s_iso, m = key[:4]
        rows.append({
            "policy": pol, "kappa": kappa, "S_iso": s_iso, "m": m,
            "rho0": key[4] if len(key) > 4 else 0.0,
            "DRR_relay": block["drr_relay"].mean(),
            "dRR": block["dRR"].mean(), "dKPD": block["dKPD"].mean(),
            "PSI": block["PSI"].mean(), "C": composite(block).mean(),
            "n": len(block)})
    tab = pd.DataFrame(rows).sort_values("C")

    # non-dominated set: no other point is both cheaper and more protective
    front = []
    for _, r in tab.iterrows():
        if not ((tab["C"] <= r["C"]) & (tab["DRR_relay"] >= r["DRR_relay"])
                & ((tab["C"] < r["C"]) | (tab["DRR_relay"] > r["DRR_relay"]))).any():
            front.append(True)
        else:
            front.append(False)
    tab["frontier"] = front
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n  'frontier' marks the non-dominated operating points: the set an\n"
          "  operator can actually choose between. Everything else is strictly\n"
          "  worse on both axes.")


def report_frontier_grid(df: pd.DataFrame) -> None:
    """The headline figure: DRR against cost over the S_iso x rho_start grid.

    Not S_iso x kappa. The lever ablation found kappa inert even at S_iso = 0.9,
    where no node is isolated at all and kappa is therefore the only lever left -
    it moves path length by 15% and damage by nothing, because it penalises the
    same node rho already throttles and rho is the stronger instrument.
    """
    # The family sweep anchors every member's throttle to that member's own quiet
    # score and the session sweep varies T_s, so a dozen extra rho_start values
    # and four session lengths live in the frame. None of them is a point of
    # this grid - they are different networks and different traffic - and pooling
    # them turns a 4x4 surface into a list.
    b3 = one_intensity(df)
    b3 = b3[(b3["policy.type"] == "B3") & b3["drr_relay"].notna()
            & (b3["topology.name"] == "net50")
            & (b3["demand.km_mode"] == "OTP")
            & np.isclose(b3["demand.T_s_max"], 600.0)]
    grid = (0.0, 0.1316, 0.2150, 0.2632)
    b3 = b3[[any(np.isclose(v, g) for g in grid)
             for v in b3["policy.rho_start"].astype(float)]]
    if b3.empty:
        return
    print("\n=== headline frontier: B3 over S_iso x rho_start  (DRR_relay / dRR) ===")
    isos = sorted(b3["policy.S_iso"].unique())
    print(f"    {'rho_start':>10}" + "".join(f"{f'S_iso={v:g}':>16}" for v in isos))
    for r0 in sorted(b3["policy.rho_start"].unique()):
        row = ""
        for si in isos:
            c = b3[(b3["policy.rho_start"] == r0) & (b3["policy.S_iso"] == si)]
            row += (f"{c['drr_relay'].mean():7.3f}/{c['dRR'].mean():+6.3f}"
                    if len(c) else f"{'-':>16}")
        print(f"    {r0:>10.4f}{row}")
    print("    rho_start 0.0 is the literal phase 1 formula; larger values move the")
    print("    start of throttling up towards the isolation threshold, so the lever")
    print("    acts on the tail instead of on the whole network.")


def one_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a single attack-intensity setting.

    The sweep deliberately contains two: the phase 1 defaults and a much quieter
    attacker. Pooling them would average a trivially detectable profile with a
    barely detectable one and report the mean of two different experiments.
    """
    cols = [c for c in ("attack.gamma", "attack.delta", "attack.q")
            if c in df.columns]
    if not cols:
        return df
    mode = df[cols].mode().iloc[0]
    keep = np.ones(len(df), dtype=bool)
    for c in cols:
        keep &= (df[c] == mode[c]).to_numpy()
    return df[keep]


def report_targeting(df: pd.DataFrame) -> None:
    """DRR minus the blind control at a matched rejection rate.

    Any throttling policy reduces damage two ways: by targeting, and by simply
    admitting fewer sessions. BT applies a uniform quota with no detection at
    all, so interpolating its curve to a policy's own dRR gives the share that is
    pure volume; what is left is the targeting benefit. Reporting DRR without
    this subtraction credits a policy for traffic it merely refused.
    """
    bt = df[(df["policy.type"] == "BT") & df["drr_relay"].notna()]
    if bt.empty:
        print("\n  no blind-throttle control runs; targeting cannot be separated")
        return
    curve = bt.groupby("policy.rho_blind")[["dRR", "drr_relay"]].mean()
    curve = curve.sort_values("dRR")
    # anchor at the origin: with no extra rejection a blind quota removes no
    # damage, exactly. Without it np.interp clamps to the cheapest measured
    # control point and understates the targeting benefit of every cheap policy.
    x = np.concatenate(([0.0], curve["dRR"].to_numpy()))
    y = np.concatenate(([0.0], curve["drr_relay"].to_numpy()))
    print("\n=== targeting benefit: DRR minus the blind control at matched cost ===")
    print("    blind control: " + "  ".join(f"dRR {a:.2f} -> {b:.2f}"
                                            for a, b in zip(x, y)))
    print(f"\n    {'policy':<26}{'dRR':>8}{'DRR':>8}{'blind':>8}{'targeting':>11}")
    sub = df[(~df["policy.type"].isin(("B0", "none", "BT")))
             & df["drr_relay"].notna()]
    rows = []
    for key, b in sub.groupby(["policy.type", "policy.tau", "policy.m_paths",
                               "policy.S_iso", "policy.rho_start"]):
        pol = key[0]
        if pol == "B1":
            lbl = f"B1 tau={key[1]:g}"
        elif pol == "B2":
            lbl = f"B2 m={int(key[2])}"
        elif pol == "B3":
            lbl = f"B3 S_iso={key[3]:g} rho0={key[4]:.3g}"
        else:
            lbl = pol
        d, drr = b["dRR"].mean(), b["drr_relay"].mean()
        blind = float(np.interp(d, x, y))
        rows.append((lbl, d, drr, blind, drr - blind))
    for lbl, d, drr, blind, gain in sorted(rows, key=lambda r: -r[4]):
        print(f"    {lbl:<26}{d:>8.3f}{drr:>8.3f}{blind:>8.3f}{gain:>+11.3f}")
    print("    'targeting' is the column that matters: a policy whose benefit is")
    print("    indistinguishable from the blind control is not detecting anything.")


OFAT_CENTRE = {"topology.name": "net50", "demand.km_mode": "OTP",
               "attack.f": 0.10, "attack.selection": "random",
               "noise.scale": 1.0, "detector.alpha": 0.05}
# demand.lam is not in the dict above because its centre value is topology- and
# km_mode-dependent; the load axis is recognised by elimination instead.
OFAT_ORDER = ["topology.name", "demand.km_mode", "attack.f", "attack.selection",
              "noise.scale", "detector.alpha"]


def ofat_cell(row: pd.Series, lam_centre: float) -> str:
    """Name the single dimension this row moves off the OFAT centre.

    The sweep varies one factor per variant by construction, so first difference
    wins; the order matters only because ``demand.lam`` is re-derived per
    topology and per key-management mode to hold offered load fixed, and would
    otherwise register as a second moved factor in every non-central topology.
    """
    for col in OFAT_ORDER:
        if col not in row.index:
            continue
        want, got = OFAT_CENTRE[col], row[col]
        if isinstance(want, float):
            if not np.isclose(float(got), want):
                return f"{col.split('.')[-1]}={got:g}"
        elif str(got) != want:
            return f"{col.split('.')[-1]}={got}"
    if "demand.lam" in row.index and not np.isclose(float(row["demand.lam"]),
                                                    lam_centre):
        return f"load={'high' if row['demand.lam'] > lam_centre else 'low'}"
    return "CENTRE"


def report_ofat(df: pd.DataFrame) -> None:
    """One factor at a time around the central cell, per profile.

    The question this answers is not "which cell is best" - it is whether the
    ordering of the policies survives moving each dimension on its own. A
    conclusion that only holds at one topology, one key-management mode and one
    compromise fraction is not a conclusion about QKD networks.
    """
    need = {"attack.profile", "drr_relay", "policy.type"}
    if not need.issubset(df.columns):
        return
    lam_c = float(df[(df.get("topology.name") == "net50")
                     & (df.get("demand.km_mode") == "OTP")]["demand.lam"].median())
    d = df.copy()
    d["cell"] = d.apply(ofat_cell, axis=1, lam_centre=lam_c)
    # The family sweep adds twelve topologies with anchors of their own and
    # the session sweep four session lengths; neither is an OFAT cell, and
    # both have a report to themselves.
    d = d[d["attack.profile"].isin(("G", "E"))
          & ~d["topology.name"].astype(str).str.startswith("fam")
          & np.isclose(d["demand.T_s_max"], 600.0)]

    # The centre cell is also where the intensity and ablation sweeps live, and
    # those deliberately move the attack intensity and switch the prior off.
    # Pooling them into CENTRE would report an AUC averaged over attackers of
    # different loudness and make the centre look worse than every cell that
    # moves off it - an artefact of the pooling, not of the factor.
    for col, spec in (("attack.gamma", 0.30), ("attack.delta", 0.50),
                      ("attack.q", 0.03)):
        if col in d.columns:
            d = d[np.isclose(d[col].astype(float), spec)]
    if "detector.use_prior" in d.columns:
        d = d[d["detector.use_prior"].astype(bool)]

    # Block A shares the centre cell with this sweep and carries B3 at four
    # rho_start anchors and B2 at m in {2,3}; pooling those into the CENTRE row
    # would compare an averaged operating point against a single one everywhere
    # else, which is not an OFAT contrast. Pin each policy to the operating
    # point the OFAT variants actually use.
    keep = ~d["policy.type"].isin(("B1", "B2", "B3"))
    if "policy.tau" in d.columns:
        keep |= (d["policy.type"] == "B1") & np.isclose(d["policy.tau"], 0.5)
    if "policy.m_paths" in d.columns:
        keep |= (d["policy.type"] == "B2") & (d["policy.m_paths"] == 2)
    if "policy.S_iso" in d.columns:
        # rho_start is anchored to each topology's own quiet score, so a single
        # pinned value selects B3 in the net50 cells and nothing anywhere else -
        # which is how the nsfnet, usnet and AES rows first came out blank.
        anchors = (0.1316, 0.1301, 0.2528, 0.2207)
        keep |= ((d["policy.type"] == "B3")
                 & np.isclose(d["policy.S_iso"], 0.5)
                 & np.array([any(np.isclose(v, a) for a in anchors)
                              for v in d["policy.rho_start"].astype(float)]))
    d = d[keep]

    print("\n=== OFAT: does the policy ordering survive moving one factor? ===")
    print("    DRR_relay / dRR;  AUC is the detector alone, from the B0 row")
    labels = [("B1", "B1 tau=.5"), ("B2", "B2 m=2"), ("B3", "B3 S=.5"),
              ("B4", "B4 oracle")]
    for prof in ("G", "E"):
        blk = d[d["attack.profile"] == prof]
        if blk.empty:
            continue
        print(f"\n  profile {prof}")
        print(f"    {'cell':<22}" + "".join(f"{lbl:>16}" for _, lbl in labels)
              + f"{'AUC':>7}")
        cells = sorted(blk["cell"].unique(), key=lambda c: (c != "CENTRE", c))
        for cell in cells:
            rows = blk[blk["cell"] == cell]
            line = ""
            for pol, _ in labels:
                r = rows[rows["policy.type"] == pol]
                line += (f"{r['drr_relay'].mean():8.3f}/{r['dRR'].mean():+6.3f}"
                         if len(r) else f"{'-':>16}")
            b0 = rows[(rows["policy.type"].isin(("B0", "none")))
                      & rows.get("attack.enabled", True).astype(bool)]
            auc = f"{b0['auc'].mean():>7.3f}" if len(b0) else f"{'-':>7}"
            print(f"    {cell:<22}{line}{auc}")
    print("    A row where B1 collapses but B3 does not is the same finding as the")
    print("    attack-intensity sweep, reached by a different route.")


def report_weights(df: pd.DataFrame) -> None:
    """Phase 8: how much of the weight sensitivity is really threshold movement.

    ``theta_0`` is calibrated against the largest evidence a single-feature
    attack can produce, which is ``w_k``, not 1/3. Re-weighting therefore moves
    the operating point even when the features themselves are unchanged. Each
    weight point appears twice - re-solved and at the phase 1 default - and the
    gap between the two AUC columns is the part of any "weight sensitivity" that
    is an artefact of holding the threshold fixed.
    """
    if "detector.weights" not in df.columns:
        return
    # The uniform point is the *default* weighting, so without these filters it
    # pools every policy run in results/raw and stops being the phase 8 cell at
    # all - it came out at L = 0.915 against 1.000 for every other point, which
    # is the pooling, not the weights. Phase 8 runs with no policy, no prior
    # and the ablation scorer, which is also exactly the ablation sweep's cell,
    # so the uniform point and that sweep legitimately share rows.
    d = df[df["detector.weights"].notna() & df["auc"].notna()].copy()
    for col, want in (("policy.type", ("B0", "none")),
                      ("detector.use_prior", (False,)),
                      ("detector.ablation", (True,)),
                      ("topology.name", ("net50",)),
                      ("demand.km_mode", ("OTP",))):
        if col in d.columns:
            d = d[d[col].isin(want)]
    for col, spec in (("attack.gamma", 0.30), ("attack.delta", 0.50),
                      ("attack.q", 0.03), ("attack.f", 0.10)):
        if col in d.columns:
            d = d[np.isclose(d[col].astype(float), spec)]
    if d.empty:
        return
    def _w(v):
        # parquet round-trips the tuple as a string on some writers and as a
        # list on others; accept both rather than depend on the arrow version
        if isinstance(v, str):
            v = v.strip("[]() ").split(",")
        return tuple(round(float(x), 2) for x in v)

    d["w"] = d["detector.weights"].map(_w)
    if d["w"].nunique() < 3:
        return
    fixed_lam = float(d["detector.lam_s"].mode().iloc[0])
    d["recal"] = ~np.isclose(d["detector.lam_s"].astype(float), fixed_lam)

    print("\n=== phase 8: detector weights, re-solved vs fixed threshold (AUC) ===")
    profs = [p for p in ("P", "G", "L", "E") if p in set(d["attack.profile"])]
    print(f"    {'weights':>18}{'lam_s':>7}" +
          "".join(f"{p + ' recal':>11}{p + ' fixed':>11}" for p in profs))
    for w, blk in sorted(d.groupby("w"), key=lambda kv: -kv[0][0]):
        lam = blk[blk["recal"]]["detector.lam_s"]
        line = ""
        for p in profs:
            for flag in (True, False):
                s = blk[(blk["attack.profile"] == p) & (blk["recal"] == flag)]
                line += f"{s['auc'].mean():>11.3f}" if len(s) else f"{'-':>11}"
        # a weighting whose re-solved lam_s lands back on the default has no
        # separate recal row, and saying so is more honest than a dash
        lam_s = f"{lam.iloc[0]:>7.1f}" if len(lam) else f"{fixed_lam:>6.1f}="
        print(f"    {str(w):>18}{lam_s}{line}")
    print("    AUC ranks nodes by score, so it is invariant to any monotone")
    print("    rescaling: the recal/fixed columns differ only where a feature is")
    print("    dropped and the ranking itself changes. The policy-facing effect")
    print("    of the threshold is in the FPR column below, not here.")
    print(f"\n    {'weights':>18}{'FPR@0.5 recal':>16}{'FPR@0.5 fixed':>16}")
    for w, blk in sorted(d.groupby("w"), key=lambda kv: -kv[0][0]):
        q = blk[blk["attack.profile"] == "P"] if "P" in profs else blk
        line = ""
        for flag in (True, False):
            s = q[q["recal"] == flag]
            line += (f"{s['fpr_node_tau05'].mean():>16.3f}" if len(s)
                     else f"{'-':>16}")
        print(f"    {str(w):>18}{line}")


def report_minimax(df: pd.DataFrame) -> None:
    """Each policy at its own worst attack intensity.

    Comparing policies at one intensity lets whoever writes the paper choose the
    operating point, and the choice decides the answer: loud attacks favour the
    binary policy, quiet ones favour the graded one. Here the attacker chooses
    instead, taking the intensity that maximises the damage it gets through, and
    every policy is reported at its own worst case:

        worst(policy) = max over intensity of  D_eff_relay(policy, intensity)

    Read as DRR, that is the *minimum* DRR over the intensity grid, and it is a
    minimax quantity: no operating point is chosen by anybody, and the ranking
    it produces is the one a defender would actually get from an adversary who
    is allowed to adapt.
    """
    knobs = {"G": "attack.gamma", "L": "attack.delta", "E": "attack.q"}
    d = df[(df["topology.name"] == "net50") & (df["demand.km_mode"] == "OTP")
           & np.isclose(df["attack.f"], 0.10)
           & np.isclose(df["detector.alpha"], 0.05)
           & np.isclose(df["noise.scale"], 1.0)
           & (df["attack.selection"] == "random")
           & (~df["detector.ablation"].astype(bool))
           & df["drr_relay"].notna()]
    if d.empty:
        return
    pols = [("B1", {"policy.tau": 0.5}, "B1 tau=0.5"),
            ("B3", {"policy.S_iso": 0.5, "policy.rho_start": 0.1316},
             "B3 S=.5 r=.119"),
            ("B2", {"policy.m_paths": 2}, "B2 m=2"),
            ("BT", {"policy.rho_blind": 0.9}, "BT rho=0.9"),
            ("B4", {}, "B4 oracle")]

    def sel(b, pol, kw):
        b = b[b["policy.type"] == pol]
        for k, v in kw.items():
            if k in b.columns:
                b = b[np.isclose(b[k].astype(float), v)]
        return b

    print("\n=== minimax: every policy at its own worst attack intensity ===")
    print("    the attacker picks the intensity, not the author\n")
    for prof, knob in knobs.items():
        blk = d[d["attack.profile"] == prof]
        lv = sorted(blk[knob].dropna().unique())
        if len(lv) < 3:
            continue
        print(f"  profile {prof}   {knob.split('.')[-1]} in "
              f"[{lv[0]:g}, {lv[-1]:g}]")
        print(f"    {'policy':<16}" + "".join(f"{v:>9g}" for v in lv)
              + f"{'WORST':>9}{'at':>8}")
        for pol, kw, lab in pols:
            row, vals = "", []
            for v in lv:
                c = sel(blk[np.isclose(blk[knob].astype(float), v)], pol, kw)
                m = c["drr_relay"].mean() if len(c) else np.nan
                vals.append(m)
                row += f"{m:>9.3f}" if np.isfinite(m) else f"{'-':>9}"
            if not np.any(np.isfinite(vals)):
                continue
            j = int(np.nanargmin(vals))
            print(f"    {lab:<16}{row}{vals[j]:>9.3f}{lv[j]:>8g}")
        print()
    print("    A policy whose row is flat has no worst case to find - B2 and B4")
    print("    never read the detector, so intensity cannot move them. The two")
    print("    that do read it are the only ones the attacker can play against,")
    print("    and the WORST column is the honest comparison between them.")


def report_prior_confound(df: pd.DataFrame) -> None:
    """Every detection number, with the configuration prior on and off.

    The prior is a function of exposure, and under ``selection=top_keyflow`` the
    attacker picks nodes by exposure too - so the prior correlates with the
    ground truth by construction and inflates AUC without any feature firing.
    Reporting one column would be a choice about how much of that to keep. Both
    columns is the only honest presentation, and the gap between them is itself
    the measurement of how much the prior is worth.
    """
    d = df[(df["topology.name"] == "net50") & (df["demand.km_mode"] == "OTP")
           & np.isclose(df["attack.f"], 0.10)
           & np.isclose(df["noise.scale"], 1.0)
           & np.isclose(df["detector.alpha"], 0.05)
           & (df["detector.x2_mode"] == "signed")
           & df["policy.type"].isin(("B0", "none"))
           & df["attack.enabled"].astype(bool) & df["auc"].notna()]
    for col, spec in (("attack.gamma", 0.30), ("attack.delta", 0.50),
                      ("attack.q", 0.03)):
        if col in d.columns:
            d = d[np.isclose(d[col].astype(float), spec)]
    if d.empty or d["detector.use_prior"].nunique() < 2:
        return
    print("\n=== the configuration prior, on and off ===")
    print(f"    {'selection':<14}{'profile':<9}{'AUC prior off':>15}"
          f"{'AUC prior on':>14}{'inflation':>11}")
    for seln in sorted(d["attack.selection"].unique()):
        for prof in ("P", "G", "L", "E"):
            b = d[(d["attack.selection"] == seln) & (d["attack.profile"] == prof)]
            off = b[~b["detector.use_prior"].astype(bool)]["auc"].mean()
            on = b[b["detector.use_prior"].astype(bool)]["auc"].mean()
            if not (np.isfinite(off) and np.isfinite(on)):
                continue
            print(f"    {seln:<14}{prof:<9}{off:>15.3f}{on:>14.3f}"
                  f"{on - off:>+11.3f}")
    print("    Under random selection the prior is uncorrelated with the")
    print("    compromised set and the inflation is noise. Under top_keyflow it")
    print("    is the whole signal, which is why every detection table in this")
    print("    project is quoted with the prior OFF.")


def report_family(df: pd.DataFrame) -> None:
    """Connectivity as an axis rather than three named anecdotes.

    Twelve 50-node members, mean degree 2.2 to 4.4, three independent samples at
    each point, each calibrated on its own (lambda bisected to 10 % rejection,
    its own phi, its own (lam_s, theta_0), its own quiet-score throttle anchor).
    Holding any of those fixed would compare a lightly loaded dense network
    against a saturated sparse one and call the difference connectivity.

    Every member is biconnected by construction, so by Menger *every* pair has
    two node-disjoint paths and B2 with m=2 is always feasible. What density
    actually buys is the third path - ``frac_ge_3_disjoint`` runs 0.4 % to 19 %
    across the family - and the cost of the second one, which is what the PSI
    and dKPD columns measure.
    """
    fam = df[df["topology.name"].astype(str).str.startswith("fam")
             & df["drr_relay"].notna()
             & np.isclose(df["attack.f"], 0.10)
             & (df["attack.selection"] == "random")]
    for col, spec in (("attack.gamma", 0.30), ("attack.delta", 0.50),
                      ("attack.q", 0.03), ("attack.s", 0.25)):
        if col in fam.columns:
            fam = fam[np.isclose(fam[col].astype(float), spec)]
    if fam.empty:
        return
    fam = fam.copy()
    fam["deg"] = fam["topology.name"].str.extract(r"fam(\d+)s")[0].astype(float) / 10
    pols = [("B1", {"policy.tau": 0.5}, "B1 tau=.5"),
            ("B3", {"policy.S_iso": 0.5}, "B3 graded"),
            ("B2", {"policy.m_paths": 2}, "B2 m=2"),
            ("B2", {"policy.m_paths": 3}, "B2 m=3"),
            ("B4", {}, "B4 oracle")]

    def sel(b, pol, kw):
        b = b[b["policy.type"] == pol]
        for k, v in kw.items():
            if k in b.columns:
                b = b[np.isclose(b[k].astype(float), v)]
        return b

    print("\n=== connectivity as an axis: 12 members, mean degree 2.2 - 4.4 ===")
    for prof in ("G", "E"):
        blk = fam[fam["attack.profile"] == prof]
        if blk.empty:
            continue
        print(f"\n  profile {prof}    DRR_relay / dRR")
        print(f"    {'degree':>7}{'>=3 disj':>10}"
              + "".join(f"{lab:>17}" for _, _, lab in pols))
        for deg in sorted(blk["deg"].unique()):
            d = blk[np.isclose(blk["deg"], deg)]
            row = ""
            for pol, kw, _ in pols:
                c = sel(d, pol, kw)
                row += (f"{c['drr_relay'].mean():8.3f}/{c['dRR'].mean():+7.3f}"
                        if len(c) else f"{'-':>17}")
            # the m=3 feasibility ceiling is a property of the graph; read it off
            # the rejection reason rather than the topology file so the table
            # stays honest about what the simulator actually did
            nd = sel(d, "B2", {"policy.m_paths": 3})
            frac = (1 - nd["n_no_disjoint"].sum()
                    / max(nd["n_offered"].sum(), 1)) if len(nd) else np.nan
            print(f"    {deg:>7.1f}{frac:>10.1%}{row}")
    print("\n    Every member is biconnected, so m=2 never fails and B2's row is")
    print("    flat - density buys it nothing.  m=3 is the row that moves, and it")
    print("    moves with the third-path fraction, which is the honest statement")
    print("    of what redundancy costs on a sparse network.")


def report_session_len(df: pd.DataFrame) -> None:
    """Does isolating early matter, and when?

    Isolation blocks new admissions through a node but lets sessions already
    routed through it drain. On the central cell the drain-versus-evict choice
    is worth +-0.01 DRR, inside seed noise - but that is a fact about a traffic
    model whose sessions last 60-600 s against a 3 h post-t_c window, not a fact
    about QKD networks. Here E[T_s] runs from 82 s to 5400 s with lambda
    rescaled by its inverse, so offered load is held and the axis is length
    alone. If the two settings separate anywhere it will be at the long end,
    where the in-flight population at the moment of isolation stops being a
    rounding error.
    """
    d = df[df["demand.T_s_max"].notna() & df["drr_relay"].notna()
           & (df["topology.name"] == "net50")
           & (df["demand.km_mode"] == "OTP")
           & np.isclose(df["attack.f"], 0.10)
           & np.isclose(df["noise.scale"], 1.0)
           & np.isclose(df["detector.alpha"], 0.05)
           & (df["attack.selection"] == "random")]
    # The spec row is the central cell, which the intensity and minimax sweeps
    # also occupy; without pinning the intensity the E row averages an attacker
    # at q=0.002 with one at q=0.03 and comes out 0.2 below its own neighbours.
    for col, spec in (("attack.gamma", 0.30), ("attack.delta", 0.50),
                      ("attack.q", 0.03), ("attack.s", 0.25)):
        if col in d.columns:
            d = d[np.isclose(d[col].astype(float), spec)]
    if d.empty or d["demand.T_s_max"].nunique() < 3:
        return
    d = d.copy()
    d["ETs"] = (d["demand.T_s_min"] + d["demand.T_s_max"]) / 2
    print("\n=== session length vs the value of evicting in-flight sessions ===")
    print("    DRR_relay;  'drain' keeps them running, 'evict' tears them down")
    for prof in ("G", "E"):
        blk = d[d["attack.profile"] == prof]
        if blk.empty:
            continue
        print(f"\n  profile {prof}")
        print(f"    {'E[T_s]':>9}" + "".join(
            f"{p + ' ' + m:>14}" for p in ("B1", "B3") for m in ("drain", "evict"))
            + f"{'B1 gain':>10}{'B3 gain':>10}")
        for ets in sorted(blk["ETs"].unique()):
            b = blk[np.isclose(blk["ETs"], ets)]
            vals = {}
            # The spec row shares its cell with the whole frontier grid, so the
            # operating point has to be pinned: without this, "B1" at E[T_s]=330
            # is an average over three thresholds and "B3" over sixteen lever
            # settings, and the row stops being comparable with the other three.
            for pol, kw in (("B1", {"policy.tau": 0.5}),
                            ("B3", {"policy.S_iso": 0.5,
                                    "policy.rho_start": 0.1316})):
                for td in (False, True):
                    c = b[(b["policy.type"] == pol)
                          & (b["policy.tear_down_on_isolate"].astype(bool) == td)]
                    for k, v in kw.items():
                        c = c[np.isclose(c[k].astype(float), v)]
                    vals[(pol, td)] = c["drr_relay"].mean() if len(c) else np.nan
            row = "".join(f"{vals[(p, m)]:>14.3f}" for p in ("B1", "B3")
                          for m in (False, True))
            g1 = vals[("B1", True)] - vals[("B1", False)]
            g3 = vals[("B3", True)] - vals[("B3", False)]
            print(f"    {ets:>9.0f}{row}{g1:>+10.3f}{g3:>+10.3f}")
    print("\n    The 'gain' columns are the whole question: a controller that can")
    print("    tear down flows is more complex than one that drains them, and the")
    print("    complexity is only worth paying for where these numbers are large.")


BASELINE_LABELS = [
    ("B1", {"policy.tau": 0.5}, "B1 binary tau=.5", "ours"),
    ("B2", {"policy.m_paths": 2}, "B2 XOR m=2", "ours"),
    ("B3", {"policy.S_iso": 0.5, "policy.rho_start": 0.1316}, "B3 graded", "ours"),
    ("B4", {}, "B4 oracle", "bound"),
    ("B5", {"policy.beta_trust": 50.0}, "B5 Luo & Li 2025", "published"),
    ("B6", {"policy.m_paths": 2}, "B6 Kiktenko+ 2024", "published"),
    ("B7", {"policy.alpha_key": 0.5}, "B7 Bi+ 2023", "published"),
]


def report_baselines(df: pd.DataFrame) -> None:
    """The comparison against published methods, per profile.

    B0-B4 and BT are all constructions of this paper, so a table containing only
    them shows an ablation, not a comparison with the state of the art. B5, B6
    and B7 are reimplementations of published methods run on the same cell, the
    same seeds and the same metric; the adaptations each one needed are in the
    class docstrings in sim/policy.py and must be restated in the paper.

    The column that decides anything is not DRR but DRR minus the blind control
    at matched cost: a published method that is beaten only on raw DRR may
    simply have been throttling less.
    """
    d = one_intensity(df)
    d = d[(d["topology.name"] == "net50") & (d["demand.km_mode"] == "OTP")
          & np.isclose(d["attack.f"], 0.10)
          & np.isclose(d["demand.T_s_max"], 600.0)
          & np.isclose(d["noise.scale"], 1.0)
          & np.isclose(d["detector.alpha"], 0.05)
          & (d["attack.selection"] == "random")]
    if d.empty or not (d["policy.type"] == "B5").any():
        return

    bt = d[d["policy.type"] == "BT"]
    curve = (bt.groupby("policy.rho_blind")[["dRR", "drr_relay"]].mean()
             .sort_values("dRR")) if len(bt) else None
    bx = curve["dRR"].to_numpy() if curve is not None else None
    by = curve["drr_relay"].to_numpy() if curve is not None else None

    print("\n=== against published methods, net50/OTP, f=0.10, 20 seeds ===")
    print("    DRR_relay [95% CI] / dRR / targeting over the blind control\n")
    for prof in PROFILES_ORDER:
        blk = d[d["attack.profile"] == prof]
        if blk.empty:
            continue
        print(f"  profile {prof}")
        print(f"    {'policy':<20}{'source':<11}{'DRR_relay':>22}"
              f"{'dRR':>8}{'dKPD':>8}{'targeting':>11}")
        for pol, kw, lab, src in BASELINE_LABELS:
            rows = blk[blk["policy.type"] == pol]
            for k, v in kw.items():
                if k in rows.columns:
                    rows = rows[np.isclose(rows[k].astype(float), v)]
            if rows.empty:
                continue
            m, lo, hi = bootstrap_ci(rows["drr_relay"].to_numpy())
            cost = rows["dRR"].mean()
            tgt = (m - float(np.interp(cost, bx, by))) if bx is not None else np.nan
            print(f"    {lab:<20}{src:<11}{m:>8.3f} [{lo:6.3f},{hi:6.3f}]"
                  f"{cost:>8.3f}{rows['dKPD'].mean():>8.3f}{tgt:>+11.3f}")
        print()
    print("    B5's mechanism is one observable - the disagreement between a")
    print("    link's two endpoint reports, which is our x2 - so it should move")
    print("    profile L and nothing else. That it does is the argument for x1")
    print("    and x3, made with the published method rather than against it.")


PROFILES_ORDER = ("P", "G", "L", "E")


def report_cost_sensitivity(df: pd.DataFrame, n: int = 200) -> None:
    """How much does the C ranking depend on the arbitrary weights a?"""
    sub = df[df["drr_relay"].notna()]
    if sub.empty:
        return
    print("\n=== how stable is the C ranking as the weights a vary? ===")
    rng = np.random.default_rng(0)
    keys = ["policy.type", "policy.kappa", "policy.S_iso", "policy.m_paths"]
    if "policy.rho_start" in sub.columns:
        keys.append("policy.rho_start")
    groups = list(sub.groupby(keys))
    if len(groups) < 2:
        return
    wins: dict = {}
    for _ in range(n):
        a = rng.dirichlet((1.0, 1.0, 1.0))
        scores = {g: composite(b, a).mean() for g, b in groups}
        best = min(scores, key=scores.get)
        wins[best] = wins.get(best, 0) + 1
    print(f"  best-C operating point over {n} random weightings:")
    for g, c in sorted(wins.items(), key=lambda kv: -kv[1]):
        label = (f"{g[0]} kappa={g[1]} S_iso={g[2]} m={int(g[3])}"
                 + (f" rho0={g[4]}" if len(g) > 4 else ""))
        print(f"    {label:<40} {c / n:6.1%}")
    print("  A single winner across the simplex means the ranking is not an\n"
          "  artefact of the weights; a split means C should not be quoted alone.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/raw/*.parquet")
    ap.add_argument("--out", default="results/analysis.parquet")
    ap.add_argument("--drr", action="store_true")
    ap.add_argument("--psi", action="store_true")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--targeting", action="store_true")
    ap.add_argument("--ofat", action="store_true")
    ap.add_argument("--weights", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--minimax", action="store_true")
    ap.add_argument("--prior", action="store_true")
    ap.add_argument("--family", action="store_true")
    ap.add_argument("--session", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.all or not any((args.drr, args.psi, args.pareto, args.cost,
                            args.frontier, args.targeting, args.ofat,
                            args.weights, args.minimax, args.prior,
                            args.family, args.session, args.baselines)):
        args.drr = args.pareto = args.cost = args.ofat = args.weights = True
        args.baselines = True
        args.frontier = args.targeting = args.psi = True
        args.minimax = args.prior = args.family = args.session = True

    full = attach_baselines(load(args.pattern))
    # every report runs on the current feature set; the full frame - superseded
    # variants included - is what gets written out, because figures.py needs the
    # abs rows to draw the before/after panel
    df = current(full)
    n_pol = int((~df["policy.type"].isin(("B0", "none"))).sum())
    print(f"{len(df)} current runs of {len(full)}, {n_pol} with a policy, "
          f"{df['scenario_id'].nunique()} scenarios")
    missing = int(df["D_eff_relay_b0"].isna().sum())
    if missing:
        print(f"  warning: {missing} runs have no matched B0 baseline")

    if args.drr:
        report_drr(df)
    if args.psi:
        print("\n=== path stretch (PSI), relative to B0 without attack ===")
        print(df.groupby("policy.type")[["PSI", "mean_leg_len", "mean_legs"]]
              .mean().round(4).to_string())
    if args.frontier:
        report_frontier_grid(df)
    if args.targeting:
        report_targeting(df)
    if args.minimax:
        report_minimax(df)
    if args.prior:
        report_prior_confound(df)
    if args.family:
        report_family(df)
    if args.session:
        report_session_len(df)
    if args.baselines:
        report_baselines(df)
    if args.ofat:
        report_ofat(df)
    if args.weights:
        report_weights(df)
    if args.pareto:
        report_pareto(df)
    if args.cost:
        report_cost_sensitivity(df)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    full.to_parquet(args.out)
    print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
