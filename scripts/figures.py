"""Every figure in the paper, regenerated from results/analysis.parquet.

    python scripts/analyze.py --all            # writes results/analysis.parquet
    python scripts/figures.py                  # writes results/figures/*.png

One rule runs through the whole file and it is the only thing that makes the
numbers comparable: **a figure may only pool runs that differ in the axis it is
plotting**. results/raw holds fourteen sweeps on top of each other - the
frontier grid, the OFAT block, the intensity sweep, the ablation block, phase 8 -
and most of them share the same central cell. Averaging a policy over "every
row at net50" silently mixes four rho_start anchors, five attack intensities and
runs with the prior switched off. ``cell()`` below pins everything that is not
being varied; each figure then relaxes exactly one pin.

The figures are deliberately plain: no gradients, no 3-D, no dual-encoded colour.
Anything that has a confidence interval gets one.
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "results/figures"
SRC = "results/analysis.parquet"

SPEC = {"attack.gamma": 0.30, "attack.delta": 0.50, "attack.q": 0.03}
PROFILES = ["P", "G", "L", "E"]
# kept as the short form for dense axes; PROFILE_LABEL is the full one
PROF_LABEL = {"P": "Passive", "G": "Greedy", "L": "Liar", "E": "Tap"}
QUIET = 0.1316          # net50/OTP quiet score - the B3 throttle anchor
C = {"B0": "#9e9e9e", "B1": "#d62728", "B2": "#2ca02c", "B3": "#1f77b4",
     "B4": "#9467bd", "BT": "#8c8c8c"}

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
})


# Plain-language naming.
#
# The internal names are short because the code says them a thousand times; a
# figure says each one once, to a reader who has not read the code. Everything
# below exists so that a reader can take one image out of the paper and still
# know what it is about.
POLICY_LABEL = {
    "B0": "No defence",
    "B1": "Threshold\n(isolate at a line)",
    "B2": "Always redundant\n(2 paths, no detector)",
    "B3": "Graded (ours)",
    "B4": "Knows who is compromised\n(isolation only)",
    "B5": "Luo & Li 2025\n(published)",
    "B6": "Kiktenko+ 2024\n(published)",
    "B7": "Bi+ 2023\n(published)",
    "B8": "Hybrid (ours)",
    "BT": "Blind throttle\n(volume control)",
}

# What each adversary actually does, rather than its letter.
PROFILE_LABEL = {
    "P": "Passive\nreads, changes nothing",
    "G": "Greedy\ndraws extra key",
    "L": "Liar\nmisreports buffers",
    "E": "Tap\ntaps the fibre",
}

# What each metric means to an operator.
M_DAMAGE = "Damage prevented\n(1 = all relay exposure removed)"
M_DAMAGE_SHORT = "Damage prevented"
M_COST_RR = "Price: extra sessions refused"
M_COST_KEY = "Price: extra key per delivered demand"
M_DETECT = "Detection quality (AUC; 0.5 = coin flip)"


def frame_damage(ax, key=False):
    """Draw the floor and the ceiling so a number can be judged.

    Damage prevented is DRR = 1 - D/D_0: zero when the policy prevents nothing,
    one when it prevents all of it. Those two are the frame, and the band
    between them is the room a policy actually has.

    This used to draw the ORACLE as the ceiling, which was wrong. The oracle
    isolates the compromised set and throttles nothing, so a policy that also
    throttles prevents more damage than the oracle by admitting less traffic -
    ours does, on the tapping adversary - and a reference line the data crosses
    is worse than no reference line at all. The oracle is still reported, as a
    row in the tables, where it is a measurement rather than a claim about what
    is possible.

    ``key=True`` labels the two lines for the legend; every other panel draws
    them unlabelled, so a three-panel figure carries the caption once.
    """
    ax.axhline(0.0, color="0.35", lw=1.0, zorder=0,
               label="prevents nothing (no defence at all)" if key else None)
    ax.axhline(1.0, color="0.35", lw=1.0, ls=":", zorder=0,
               label="prevents all relay damage" if key else None)
    ax.axhspan(0.0, 1.0, color="0.5", alpha=0.05, zorder=-1)

def panel_tags(axes, y=1.02):
    """Tag panels (a), (b), (c) so the text can point at one of them."""
    for k, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.text(-0.02, y, f"({chr(97 + k)})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="right")


def cell(df, *, topo="net50", km="OTP", f=0.10, spec=True, prior=None,
         load=True, alpha=0.05, noise=1.0, selection="random", x2="signed",
         x1="signed", collude=False, teardown=False, route="exp",
         hybrid_agg="product"):
    """Pin the central cell; pass None to any argument to relax that pin.

    Every categorical mode belongs here. Each one was added by a study that
    selects on it itself, and each one, left unpinned, turns some unrelated
    figure's operating point into an average over a variant that study measured
    and rejected.
    """
    d = df
    if collude is not None and "attack.collude" in d.columns:
        d = d[d["attack.collude"].astype(bool) == collude]
    # Tearing down sessions already in flight is a controller design choice with
    # a measured effect (F18), so the central cell has to sit on one side of it.
    # The default is to drain rather than evict.
    if teardown is not None and "policy.tear_down_on_isolate" in d.columns:
        d = d[d["policy.tear_down_on_isolate"].astype(bool) == teardown]
    # logrisk is a measured failure, not an alternative: 0.136 against exp's
    # 0.317 at kappa = 10. Averaging it in halves every routing number.
    if route is not None and "policy.route_mode" in d.columns:
        d = d[d["policy.route_mode"].fillna("exp") == route]
    # the max and mean path-risk forms belong to the aggregation study alone
    if hybrid_agg is not None and "policy.hybrid_agg" in d.columns:
        d = d[d["policy.hybrid_agg"].fillna("product") == hybrid_agg]
    if x2 is not None:
        d = d[d["detector.x2_mode"] == x2]
    if x1 is not None:
        d = d[d["detector.x1_mode"] == x1]
    if topo is not None:
        d = d[d["topology.name"] == topo]
    if km is not None:
        d = d[d["demand.km_mode"] == km]
    if f is not None:
        d = d[np.isclose(d["attack.f"], f)]
    if spec:
        for col, v in SPEC.items():
            d = d[np.isclose(d[col].astype(float), v)]
    if prior is not None:
        d = d[d["detector.use_prior"].astype(bool) == prior]
    if alpha is not None:
        d = d[np.isclose(d["detector.alpha"], alpha)]
    if noise is not None:
        d = d[np.isclose(d["noise.scale"], noise)]
    if selection is not None:
        d = d[d["attack.selection"] == selection]
    if load:
        # demand.lam is re-derived per topology and per key mode to hold offered
        # load fixed, so "the central load" is the modal value within the cell
        if len(d):
            d = d[np.isclose(d["demand.lam"], d["demand.lam"].mode().iloc[0])]
    return d


# Policy knobs and the value a figure gets if it does not ask for one. Anything
# a caller leaves unpinned is pinned here instead, so that adding a sweep over a
# knob cannot silently turn some other figure's operating point into an average
# over it. Pass None for a knob to opt out and average deliberately.
OP_DEFAULTS = {
    "B1": {"policy.tau": 0.5},
    "B2": {"policy.m_paths": 2},
    "B3": {"policy.S_iso": 0.5, "policy.rho_start": QUIET, "policy.kappa": 0.0},
    "B5": {"policy.beta_trust": 50.0},
    "B6": {"policy.m_paths": 2},
    "B7": {"policy.alpha_key": 0.5},
    "B8": {"policy.S_iso": 0.5, "policy.rho_start": QUIET, "policy.kappa": 0.0,
           "policy.m_paths": 2},
    "BT": {},
}


def op(d, pol, **kw):
    """One policy at one operating point. Without this every B3 curve is an
    average over four rho_start anchors, which is not an operating point."""
    d = d[d["policy.type"] == pol]
    pins = dict(OP_DEFAULTS.get(pol, {}))
    pins.update(kw)
    for k, v in pins.items():
        if v is None or k not in d.columns:
            continue
        d = d[np.isclose(d[k].astype(float), v)]
    return d


def ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v[0]) if v.size else np.nan), 0.0, 0.0
    r = np.random.default_rng(seed)
    b = r.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    m = v.mean()
    return m, m - np.quantile(b, 0.025), np.quantile(b, 0.975) - m


PAPER = bool(os.environ.get("QKD_PAPER"))


def _strip_titles(fig):
    """Drop the in-figure headline, keep real subplot labels.

    The manuscript repeats it verbatim in the LaTeX caption directly below, and
    a reviewer reading both sees the same sentence twice. Axes titles that name
    a panel rather than the figure ("profile G greedy") are kept, so the
    heuristic is: remove the figure suptitle, and remove an axes title only if
    it opens with the figure tag Fn.
    """
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    # Titles open either with the old Fn tag or with the question the
    # figure answers; the LaTeX caption restates both.
    head = r"^(F\d+\b|Q:)"
    for ax in fig.axes:
        t = ax.get_title()
        if re.match(head, t):
            ax.set_title("")
        elif "\n" in t and re.match(head, t.split("\n")[0]):
            ax.set_title(t.split("\n", 1)[1])


def save(fig, name, caption):
    """Write the PNG for review and a vector PDF for the manuscript.

    Journals rasterise a 160 dpi PNG badly at print size; the PDF keeps the
    text selectable and the lines sharp, and LaTeX picks it over the PNG
    automatically when both are on the graphics path.
    """
    path = os.path.join(OUT, name)
    fig.savefig(path)
    if PAPER:
        _strip_titles(fig)
    fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    print(f"  {name:<34} {caption}")


def heat(ax, M, xlab, ylab, fmt="{:.3f}", cmap="RdYlGn", vmin=None, vmax=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlab)), xlab, rotation=0)
    ax.set_yticks(range(len(ylab)), ylab)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                lo, hi = (vmin, vmax) if vmin is not None else (
                    np.nanmin(M), np.nanmax(M))
                rel = (M[i, j] - lo) / max(hi - lo, 1e-9)
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=8,
                        color="white" if rel < 0.18 or rel > 0.93 else "black")
    return im


def _w(v):
    if isinstance(v, str):
        v = v.strip("[]() ").split(",")
    return tuple(round(float(x), 2) for x in v)


# 1. detection
SUBSETS = [("key accounting", "abl_x1_auc"),
           ("neighbour disagreement", "abl_x2_auc"),
           ("quantum layer", "abl_x3_auc"),
           ("key + neighbour", "abl_x1x2_auc"),
           ("key + quantum", "abl_x1x3_auc"),
           ("neighbour + quantum", "abl_x2x3_auc"),
           ("all three", "abl_x1x2x3_auc")]


def fig_ablation(df):
    """Which feature detects which attack - the phase 4 gate.

    Scored with the prior OFF. The prior is a per-node quantity correlated with
    exposure by construction, so leaving it in would put a floor under every
    cell and turn a feature table into a prior table.
    """
    d = cell(df, prior=False)
    d = d[d["detector.ablation"].astype(bool)
          & (d["policy.type"].isin(("B0", "none")))
          & d["attack.enabled"].astype(bool)
          ]
    # phase 8 also lives in this cell and moves the weights, which changes every
    # subset AUC as well as the combined one; this table is about the features
    d = d[d["detector.weights"].map(_w) == (0.33, 0.33, 0.33)]
    M = np.array([[d[d["attack.profile"] == p][c].mean() for p in PROFILES]
                  for _, c in SUBSETS])
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    im = heat(ax, M, [PROFILE_LABEL[p] for p in PROFILES],
              [s for s, _ in SUBSETS], vmin=0.4, vmax=1.0)
    fig.colorbar(im, ax=ax, label=M_DETECT, fraction=0.046)
    ax.set_title("Q: which signal catches which attacker?\n"
                 "A: one each, and nothing catches the silent one",
                 fontsize=10.5)
    ax.set_xlabel("what the attacker does")
    ax.set_ylabel("signals given to the detector")
    save(fig, "f01_ablation.png", "feature x profile AUC (the diagonal)")


def fig_intensity(df):
    """How detectable each attack is as a function of its own intensity knob.

    The three profiles are driven by three different parameters, so the x axis
    is normalised to each one's own spec value (1.0 = the phase 1 default).
    Without this the three curves cannot share a panel.
    """
    d = df[(df["topology.name"] == "net50") & (df["demand.km_mode"] == "OTP")
           & (~df["detector.use_prior"].astype(bool))
           & d_ok(df) & df["attack.enabled"].astype(bool)]
    # phase 8 sits at the spec intensity with the weights moved, so without this
    # the spec point alone is an average over ten weightings and dips below its
    # neighbours - an artefact that would read as the attack getting *harder*
    d = d[d["detector.weights"].map(_w) == (0.33, 0.33, 0.33)]
    knobs = [("G", "attack.gamma", 0.30), ("L", "attack.delta", 0.50),
             ("E", "attack.q", 0.03)]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    for ax, (prof, col, spec) in zip(axes, knobs):
        b = d[d["attack.profile"] == prof]
        xs = sorted(b[col].dropna().unique())
        m, lo, hi = zip(*[ci(b[np.isclose(b[col], x)]["auc"]) for x in xs])
        rel = np.array(xs) / spec
        ax.errorbar(rel, m, yerr=[lo, hi], marker="o", ms=4, lw=1.4,
                    color=C["B3"], capsize=2)
        ax.axhline(0.5, color="0.5", ls=":", lw=1)
        ax.axvline(1.0, color=C["B1"], ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(rel, [f"{v:g}" for v in xs], minor=False)
        ax.set_xticks([], minor=True)
        ax.set_title(PROFILE_LABEL[prof].replace(chr(10), " - "), fontsize=9)
        ax.set_xlabel(f"{col.split('.')[-1]}   (spec = {spec:g}, dashed)")
        ax.set_ylim(0.4, 1.03)
    axes[0].set_ylabel(M_DETECT)
    axes[0].text(0.72, 0.508, "chance", transform=axes[0].get_yaxis_transform(),
                 fontsize=7, color="0.4")
    fig.suptitle("Q: how loud does an attacker have to be before we see it?\n"
                 "A: it depends entirely on which signal it disturbs",
                 y=1.10, fontsize=11)
    save(fig, "f02_intensity.png", "AUC vs attack intensity, per channel")


def d_ok(df):
    """Rows where nothing but the intensity has been moved off centre."""
    return (np.isclose(df["attack.f"], 0.10)
            & (df["detector.x2_mode"] == "signed")
            & (df["detector.x1_mode"] == "signed")
            & np.isclose(df["detector.alpha"], 0.05)
            & np.isclose(df["noise.scale"], 1.0)
            & (df["attack.selection"] == "random")
            & df["policy.type"].isin(("B0", "none")))


def fig_x2_mode(df):
    """The F2 disambiguation: |a-b| accuses both endpoints, a-b accuses one.

    Only profile L can produce an asymmetric neighbour discrepancy, so only L
    should move - that the other three are unchanged is the check that the
    statistic was fixed rather than merely made larger.
    """
    d = cell(df, prior=False, x2=None)
    d = d[d["detector.ablation"].astype(bool)
          & d["policy.type"].isin(("B0", "none"))
          & d["attack.enabled"].astype(bool)]
    d = d[d["detector.weights"].map(_w) == (0.33, 0.33, 0.33)]
    modes = [m for m in ("abs", "signed") if (d["detector.x2_mode"] == m).any()]
    if len(modes) < 2:
        print("  f03 skipped: no x2_mode=abs runs on disk")
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    w, x = 0.36, np.arange(len(PROFILES))
    for k, (ax, col, lab, lo) in enumerate((
            (axes[0], "abl_x2_auc", "Detection quality of the\nneighbour signal alone", 0.4),
            (axes[1], "fpr_node_tau05",
             "False alarms: honest relays\nwrongly flagged", 0.0))):
        for i, m in enumerate(modes):
            b = d[d["detector.x2_mode"] == m]
            vals = [ci(b[b["attack.profile"] == p][col]) for p in PROFILES]
            ax.bar(x + (i - 0.5) * w, [v[0] for v in vals], w,
                   yerr=np.array([[v[1] for v in vals], [v[2] for v in vals]]),
                   label=("blame both ends" if m == "abs"
                          else "blame the one that disagrees"), capsize=2,
                   color=C["B1"] if m == "abs" else C["B3"])
        ax.set_xticks(x, PROFILES)
        ax.set_ylabel(lab)
        ax.set_xlabel("what the attacker does")
        ax.set_ylim(bottom=lo)
        if k == 0:
            ax.axhline(0.5, color="0.5", ls=":", lw=1)
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Q: does it matter WHICH of two disagreeing neighbours we "
                 "blame?\nA: yes. Keeping the direction of the disagreement "
                 "cuts false alarms by two thirds", y=1.10, fontsize=10.5)
    save(fig, "f03_x2_mode.png", "x2 absolute vs signed, AUC and FPR")


def fig_separation(df):
    """Mean smoothed score of compromised vs healthy nodes.

    AUC is a ranking statistic and says nothing about where the scores actually
    sit, which is what a threshold policy acts on. This is the same data in the
    units the policy sees.
    """
    d = cell(df, prior=True)
    d = d[d["policy.type"].isin(("B0", "none")) & d["attack.enabled"].astype(bool)
          & (~d["detector.ablation"].astype(bool))]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, p in enumerate(PROFILES):
        b = d[d["attack.profile"] == p]
        if b.empty:
            continue
        h, hlo, hhi = ci(b["mean_S_bar_healthy"])
        c, clo, chi = ci(b["mean_S_bar_compromised"])
        ax.plot([h, c], [i, i], color="0.7", lw=2, zorder=1)
        ax.errorbar(h, i, xerr=[[hlo], [hhi]], marker="o", ms=7, color="#2ca02c",
                    capsize=2, zorder=2,
                    label="honest relay" if i == 0 else None)
        ax.errorbar(c, i, xerr=[[clo], [chi]], marker="D", ms=7, color="#d62728",
                    capsize=2, zorder=2,
                    label="compromised relay" if i == 0 else None)
    for tau, ls in ((0.5, "--"), (0.7, ":")):
        ax.axvline(tau, color="0.4", ls=ls, lw=1)
        ax.text(tau, 0.99, rf"$\tau$={tau}", fontsize=7,
                color="0.4", ha="center", va="top",
                transform=ax.get_xaxis_transform())
    ax.set_yticks(range(len(PROFILES)),
                  [PROFILE_LABEL[p].replace(chr(10), " - ")
                   for p in PROFILES], fontsize=8)
    ax.set_xlabel("how suspicious the detector thinks a node is\n"
                  r"(smoothed risk score $\bar{S}$)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Q: does the risk score separate compromised relays?\n"
                 "A: yes, but for two attackers it lands between the "
                 "thresholds", fontsize=10.5)
    save(fig, "f04_score_separation.png", "S_bar separation vs the thresholds")


# 2. response - cost and benefit
def _points(d):
    """Every operating point in the central cell, as (label, policy, rows)."""
    out = []
    for tau in sorted(d[d["policy.type"] == "B1"]["policy.tau"].dropna().unique()):
        out.append((f"Threshold, cut at {tau:g}", "B1", op(d, "B1", **{"policy.tau": tau})))
    for m in sorted(d[d["policy.type"] == "B2"]["policy.m_paths"].dropna().unique()):
        out.append((f"Always redundant, {int(m)} paths", "B2", op(d, "B2", **{"policy.m_paths": m})))
    b3 = d[d["policy.type"] == "B3"]
    for si in sorted(b3["policy.S_iso"].dropna().unique()):
        for r0 in sorted(b3["policy.rho_start"].dropna().unique()):
            rows = op(d, "B3", **{"policy.S_iso": si, "policy.rho_start": r0})
            if len(rows):
                out.append((f"Graded: cut {si:g}, throttle from {r0:.2f}",
                            "B3", rows))
    if len(d[d["policy.type"] == "B4"]):
        out.append(("Knows who is compromised", "B4", d[d["policy.type"] == "B4"]))
    for rb in sorted(d[d["policy.type"] == "BT"]["policy.rho_blind"].dropna().unique()):
        out.append((f"Refuse traffic only, {1 - rb:.0%} of it", "BT", op(d, "BT", **{"policy.rho_blind": rb})))
    return [(l, p, r) for l, p, r in out if len(r)]


def fig_grid(df):
    """The two live levers of the graded policy, as a surface.

    rho_start sets how much of the network the throttle touches; S_iso sets how
    readily a node is removed outright. Both panels are needed: the left one
    alone would recommend the top-left corner, which is also the most expensive
    cell in the right one.
    """
    d = cell(df)
    b3 = d[(d["policy.type"] == "B3") & d["drr_relay"].notna()
           & np.isclose(d["demand.T_s_max"], 600.0)]
    grid = (0.0, QUIET, 0.2150, 0.2632)
    b3 = b3[np.array([any(np.isclose(v, g) for g in grid)
                      for v in b3["policy.rho_start"].astype(float)])]
    if b3.empty:
        return
    iso = sorted(b3["policy.S_iso"].unique())
    r0s = sorted(b3["policy.rho_start"].unique())
    A = np.full((len(r0s), len(iso)), np.nan)
    B = np.full_like(A, np.nan)
    for i, r in enumerate(r0s):
        for j, s in enumerate(iso):
            c = b3[np.isclose(b3["policy.rho_start"], r)
                   & np.isclose(b3["policy.S_iso"], s)]
            if len(c):
                A[i, j], B[i, j] = c["drr_relay"].mean(), c["dRR"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    fig.subplots_adjust(wspace=0.45)
    xl = [f"{v:g}" for v in iso]
    yl = [f"{v:.4g}" for v in r0s]
    im0 = heat(axes[0], A, xl, yl, vmin=0.3, vmax=0.95)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    axes[0].set_title("what it prevents")
    im1 = heat(axes[1], B, xl, yl, fmt="{:+.3f}", cmap="RdYlGn_r",
               vmin=0.10, vmax=0.60)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    axes[1].set_title("what it costs")
    for ax in axes:
        ax.set_xlabel("how sure before a relay is cut off  "
                      r"($S_{iso}$)", fontsize=8.5)
        ax.set_ylabel("how early throttling starts  "
                      r"($\rho_{start}$)", fontsize=8.5)
    fig.suptitle("Q: where should the two dials of the graded policy be set?\n"
                 "A: nowhere in particular; they trade protection against "
                 "price, so the operator chooses", y=1.08, fontsize=10.5)
    save(fig, "f06_frontier_grid.png", "S_iso x rho_start surfaces")


def fig_targeting(df):
    """DRR with the volume effect subtracted.

    Interpolating the blind-control curve to a policy's own dRR gives the share
    of its damage reduction that any uniform quota of the same cost would also
    have achieved. What is left is targeting. Reporting DRR without this
    subtraction credits a policy for traffic it merely refused.
    """
    d = cell(df)
    pts = _points(d)
    bt = sorted((r["dRR"].mean(), r["drr_relay"].mean())
                for l, p, r in pts if p == "BT")
    if not bt:
        return
    bx, by = map(np.array, zip(*bt))
    # Sixteen graded operating points is more rows than anyone reads, and the
    # frontier figure already carries the full grid; keep the ones the text
    # discusses so the bars stay distinguishable.
    # One isolation point per throttle anchor is enough: the anchor is what the
    # figure is about and the isolation point moves a row by a few hundredths.
    keep_b3 = {(0.5, 0.0), (0.5, QUIET)}
    keep_b1 = {0.5}
    rows = []
    for lab, pol, r in pts:
        if pol == "BT":
            continue
        if pol == "B1" and float(r["policy.tau"].iloc[0]) not in keep_b1:
            continue
        if pol == "B3":
            si = float(r["policy.S_iso"].iloc[0])
            r0 = float(r["policy.rho_start"].iloc[0])
            if not any(abs(si - a) < 1e-6 and abs(r0 - b) < 1e-6
                       for a, b in keep_b3):
                continue
        drr, cost = r["drr_relay"].mean(), r["dRR"].mean()
        rows.append((lab, pol, drr, float(np.interp(cost, bx, by))))
    rows.sort(key=lambda t: t[2] - t[3])
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    y = np.arange(len(rows))
    ax.barh(y, [r[3] for r in rows], color="0.82",
            label="free: any policy refusing as much traffic gets this")
    ax.barh(y, [r[2] - r[3] for r in rows], left=[r[3] for r in rows],
            color=[C[r[1]] for r in rows],
            label="earned: the part that came from aiming")
    for i, r in enumerate(rows):
        ax.text(r[2] + 0.012, i, f"{r[2] - r[3]:+.3f}", va="center", fontsize=7,
                color=C[r[1]])
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("Damage prevented, split by where it came from")
    ax.set_xlim(0, 1.14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              fontsize=7.5)
    ax.set_title("Q: how much of the benefit is aiming, and how much is just\n"
                 "admitting less traffic?", fontsize=10)
    save(fig, "f07_targeting.png", "targeting benefit over the blind control")


def fig_cost(df):
    """The three cost axes separately, because they are not interchangeable.

    Extra rejection is a service-level cost, extra key per demand is an
    operational one, and path stretch is a latency one. B2 is cheap on the
    first and ruinous on the second; collapsing them into one score would hide
    exactly the trade-off the paper is about.
    """
    d = cell(df)
    pts = [(l, p, r) for l, p, r in _points(d) if p != "BT"]
    pts = [t for t in pts if t[1] != "B3" or "/0.119" in t[0]]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), sharey=True)
    y = np.arange(len(pts))
    for ax, col, lab in ((axes[0], "dRR", "sessions refused"),
                         (axes[1], "dKPD", "key spent per delivered demand"),
                         (axes[2], "PSI", "how much longer each path gets")):
        ax.barh(y, [r[col].mean() for _, _, r in pts],
                color=[C[p] for _, p, _ in pts])
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_xlabel(lab)
    axes[0].set_yticks(y, [l for l, _, _ in pts], fontsize=6.5)
    fig.suptitle("Q: what does each defence cost?\nA: three different "
                 "currencies, and no policy is cheap in all three",
                 y=1.07, fontsize=10.5)
    save(fig, "f08_cost_axes.png", "dRR / dKPD / PSI per operating point")


# 3. robustness
def fig_quiet(df):
    """The central claim: what happens when the attacker stops being loud.

    Spec intensity is the phase 1 default; quiet is the smallest value in the
    intensity sweep. B1 and B3 read the same score through the same hysteresis
    band, so the only difference between the two panels is threshold versus ramp.
    """
    d = cell(df, spec=False)
    # the quiet level is the lowest intensity that was actually run *with the
    # policies on*; the detector-only intensity sweep goes lower, but a policy
    # point needs its own matched B0 baseline to form a DRR at all
    pol = d[~d["policy.type"].isin(("B0", "none"))]
    quiet = {c: float(pol[c].dropna().min()) for c in SPEC}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    labs = [("B1", {"policy.tau": 0.5}, r"Threshold"),
            ("B3", {"policy.S_iso": 0.5, "policy.rho_start": QUIET}, "Graded (ours)"),
            ("B2", {"policy.m_paths": 2}, "Always redundant"),
            ("B4", {}, "Knows who is compromised")]
    w, x = 0.36, np.arange(len(labs))
    for ax, (prof, knob) in zip(axes, (("G", "attack.gamma"),
                                       ("L", "attack.delta"),
                                       ("E", "attack.q"))):
        b = d[d["attack.profile"] == prof]
        for i, (tag, lvl) in enumerate((("spec", SPEC[knob]),
                                        ("quiet", quiet[knob]))):
            sub = b[np.isclose(b[knob].astype(float), lvl)]
            vals = [ci(op(sub, p, **kw)["drr_relay"]) for p, kw, _ in labs]
            ax.bar(x + (i - 0.5) * w, [v[0] for v in vals], w,
                   yerr=np.array([[max(v[1], 0) for v in vals],
                                  [max(v[2], 0) for v in vals]]),
                   capsize=2, label=("attacking at full strength" if tag == "spec"
                          else "attacking as quietly as it can"),
                   color=C["B1"] if tag == "spec" else C["B3"])
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xticks(x, [l for _, _, l in labs], fontsize=7.5, rotation=20)
        ax.set_title(PROFILE_LABEL[prof].replace(chr(10), " - "), fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylim(-0.15, 1.1)
    axes[0].set_ylabel(M_DAMAGE_SHORT)
    fig.suptitle("Q: what if the attacker simply turns itself down?\n"
                 "A: the threshold policy stops working; the graded one "
                 "keeps most of what it had", y=1.09, fontsize=10.5)
    save(fig, "f09_quiet_attacker.png", "spec vs quiet intensity, per policy")


# Each entry is a keyword override for ``cell()``. Passing the value as a pin
# rather than relaxing the pin and filtering afterwards matters: ``cell`` fixes
# demand.lam to the modal value of whatever survives the other filters, and lam
# is re-derived per topology and per key mode to hold offered load constant.
# Relaxing topo to None and then selecting nsfnet would therefore keep net50's
# lam and drop every nsfnet row - which is exactly how the topology cells went
# missing from the first version of this figure.
# NSFNET, USNET and the AES key mode were dropped from this figure: all three
# moved both policies together without separating them, so they cost a column
# each and answered nothing the remaining eleven do not. They are still swept,
# and scripts/analyze.py --ofat still prints them.
OFAT_CELLS = [
    ("baseline setup", {}),
    ("5% relays taken", {"f": 0.05}),
    ("20% relays taken", {"f": 0.20}),
    ("30% relays taken", {"f": 0.30}),
    ("half the noise", {"noise": 0.5}),
    ("twice the noise", {"noise": 2.0}),
    ("stricter alarm", {"alpha": 0.01}),
    ("looser alarm", {"alpha": 0.20}),
    ("attacker picks\nthe busiest relays", {"selection": "top_keyflow"}),
]


def fig_ofat(df):
    """One factor moved at a time, with the detector's own AUC beside it.

    The ordering of the policies is not the finding; the AUC line is, and what
    it says is narrower than it first looks. AUC tracks what either policy
    achieves (0.91 for B1, 0.90 for B3 over the 28 cells) but not the distance
    between them (0.14). The greedy column is the reason: the binary policy is
    on the floor in all fourteen of its cells across an AUC range of 0.55 to
    0.90, because AUC is a ranking and tau is a level. At the best greedy
    reading, 0.899, the compromised nodes still score 0.336 against tau = 0.5.

    The AUC comes from separate detector-only runs with the configuration prior
    switched off (config/sweep_ofat_prior_off.yaml), not from the policy runs
    plotted as bars. The policy runs keep the prior because the policy consumes
    it, and reading their AUC as a detection result is the confound the
    methodology warns about rather than a measurement.
    """
    df = df[~df["topology.name"].astype(str).str.startswith("fam")
            & np.isclose(df["demand.T_s_max"], 600.0)]
    cells = [(lab, cell(df, **kw)) for lab, kw in OFAT_CELLS]
    cells = [(l, d) for l, d in cells if len(d)]
    # the load axis is the one factor that is not a config value of its own
    free = cell(df, load=False)
    lams = sorted(free["demand.lam"].dropna().unique())
    if len(lams) >= 3:
        for lab, v in (("light traffic", lams[0]),
                       ("heavy traffic", lams[-1])):
            d = free[np.isclose(free["demand.lam"], v)]
            if len(d):
                cells.append((lab, d))

    fig, axes = plt.subplots(2, 1, figsize=(9.8, 6.4), sharex=True)
    x = np.arange(len(cells))
    for ax, prof in zip(axes, ("G", "E")):
        for i, (pol, kw, col, lab) in enumerate((
                ("B1", {"policy.tau": 0.5}, C["B1"], "Threshold"),
                ("B3", {"policy.S_iso": 0.5}, C["B3"], "Graded (ours)"))):
            v = [op(d[d["attack.profile"] == prof], pol, **kw)["drr_relay"].mean()
                 for _, d in cells]
            ax.bar(x + (i - 0.5) * 0.38, v, 0.38, color=col, label=lab)
        ax2 = ax.twinx()
        # use_prior MUST be off here. The prior is a susceptibility term built
        # from exposure and key-flow share; when the adversary selects its
        # relays by key-flow share it separates the compromised set on its own,
        # and the "busiest relays" cell reported 0.98 for a detector that reads
        # 0.90. It also costs AUC where the selection is random, since there it
        # is uncorrelated with the truth and only adds variance. Either way the
        # number is not the detector's.
        auc = [d[(d["attack.profile"] == prof)
                 & d["policy.type"].isin(("B0", "none"))
                 & d["attack.enabled"].astype(bool)
                 & (~d["detector.use_prior"].astype(bool))
                 & (~d["detector.ablation"].astype(bool))]["auc"].mean()
               for _, d in cells]
        ax2.plot(x, auc, color="0.15", marker="s", ms=4, lw=1.3, ls="--",
                 label="how well the detector sees (right axis)", zorder=5)
        ax2.axhline(0.5, color="0.5", lw=0.8, ls=":")
        ax2.set_ylim(0.4, 1.02)
        ax2.set_ylabel("detection quality (AUC)", fontsize=8.5)
        ax2.grid(False)
        ax.set_ylabel(M_DAMAGE_SHORT)
        ax.set_ylim(0, 1.05)
        ax.set_title(PROFILE_LABEL[prof].replace(chr(10), " - "),
                     loc="left", fontsize=9.5)
        if prof == "G":
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=3, loc="upper left")
    axes[1].set_xticks(x, [c for c, _ in cells], rotation=35, ha="right",
                       fontsize=7.5)
    fig.suptitle("Q: does the result survive changing the network, the "
                 "traffic, the noise?\nA: yes. Against the greedy attacker the "
                 "threshold policy is on the floor in all eleven,\n"
                 "even in the cell where the detector sees it best",
                 y=1.02, fontsize=10.5)
    save(fig, "f10_ofat.png", "11 OFAT cells, B1 vs B3, AUC overlaid")


def fig_f_curve(df):
    """Every policy against the compromise fraction, which is the one axis an
    operator does not control. The two families fail for different reasons and
    the crossing point is the useful part.
    """
    d = cell(df, f=None)
    fs = sorted(d["attack.f"].dropna().unique())
    labs = [("B1", {"policy.tau": 0.5}, "Threshold"),
            ("B3", {"policy.S_iso": 0.5, "policy.rho_start": QUIET},
             "Graded (ours)"),
            ("B2", {"policy.m_paths": 2}, "Always redundant"),
            ("B4", {}, "Knows who is compromised")]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    for ax, prof in zip(axes, ("G", "E")):
        b = d[d["attack.profile"] == prof]
        for pol, kw, lab in labs:
            m, lo, hi = zip(*[ci(op(b[np.isclose(b["attack.f"], f)], pol, **kw)
                                 ["drr_relay"]) for f in fs])
            ax.errorbar(fs, m, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)],
                        marker="o", ms=4, lw=1.4, capsize=2, color=C[pol],
                        label=lab)
        ax.set_xlabel("fraction of relays compromised")
        ax.set_title(PROFILE_LABEL[prof].replace(chr(10), " - "), fontsize=9)
        ax.set_ylim(-0.1, 1.05)
    axes[0].set_ylabel(M_DAMAGE_SHORT)
    axes[0].legend(fontsize=7.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.suptitle("Q: how many relays can be compromised before each defence "
                 "fails?\nA: they fail for different reasons, and at "
                 "different points", y=1.08, fontsize=10.5)
    save(fig, "f11_f_curve.png", "DRR vs compromise fraction")


def fig_kappa(df):
    """The routing-weight lever, measured where it is the only lever acting.

    At S_iso=0.9 no node is isolated, so any movement here is kappa's. It buys
    nothing and costs path length, because kappa and rho act on the same node
    and rho - which refuses the demand outright - is far the stronger of the two.
    """
    d = cell(df)
    b3 = d[(d["policy.type"] == "B3") & np.isclose(d["policy.S_iso"], 0.9)]
    ks = sorted(b3["policy.kappa"].dropna().unique())
    if len(ks) < 3:
        print("  f12 skipped: kappa sweep not on disk")
        return
    m, lo, hi = zip(*[ci(b3[np.isclose(b3["policy.kappa"], k)]["drr_relay"])
                      for k in ks])
    hops = [b3[np.isclose(b3["policy.kappa"], k)]["mean_leg_len"].mean() for k in ks]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.errorbar(ks, m, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], marker="o",
                ms=5, lw=1.6, capsize=3, color=C["B3"], label=M_DAMAGE_SHORT)
    ax.set_xlabel("how hard routing avoids suspect relays  "
                  r"($\kappa$)")
    ax.set_ylabel(M_DAMAGE_SHORT, color=C["B3"])
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    rel = np.array(hops) / hops[0] - 1
    ax2.plot(ks, 100 * rel, color=C["B1"], marker="s", ms=5, ls="--",
             label="price: longer paths")
    ax2.set_ylabel("how much longer paths get (%)", color=C["B1"])
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title("Q: is the routing dial worth having?\n"
                 "A: yes, but only once the admission dial is out of its way",
                 fontsize=10.5)
    save(fig, "f12_kappa.png", "the routing lever, measured in isolation")


# 4. sensitivity and accounting
def fig_weights(df):
    """Phase 8. Two panels because the weights and the threshold they force are
    two different sensitivities and only one of them turns out to matter.

    Left: re-weighting a feature that is present moves AUC by almost nothing;
    removing one destroys exactly the profile that feature detects. Right: the
    threshold the weights force moves the false-positive rate by a factor of two
    while leaving the ranking alone, so a weight sweep reported on any
    threshold-dependent quantity would have been measuring theta_0.
    """
    d = cell(df, prior=False)
    d = d[d["detector.ablation"].astype(bool)
          & d["policy.type"].isin(("B0", "none"))
          & d["detector.weights"].notna() & d["auc"].notna()]
    if d.empty:
        return
    d = d.copy()
    d["w"] = d["detector.weights"].map(_w)
    if d["w"].nunique() < 5:
        print("  f13 skipped: phase 8 sweep not on disk")
        return
    # "Re-solved" has to be keyed on the (lam_s, theta_0) PAIR. Uniform weights
    # solve back to lam_s = 11.0, the default, and differ only in theta_0 -
    # keying on lam_s alone silently files the uniform recalibrated point under
    # "fixed" and leaves that row with no comparison at all.
    d["pair"] = list(zip(d["detector.lam_s"].round(4),
                         d["detector.theta_0"].round(4)))
    fixed = d["pair"].mode().iloc[0]
    d["recal"] = d["pair"] != fixed
    ws = sorted(d["w"].unique(), key=lambda t: (min(t) > 0, -t[0], -t[1]))

    # the heatmap shows the re-solved surface, which is the operating point each
    # weighting actually implies; the fixed surface is the right-hand panel's job
    # Uniform weights re-solve to the phase 1 default, so they have no
    # separate "re-solved" row; fall back to the fixed one rather than
    # leaving a hole in the middle of the map.
    r = d[d["recal"]]
    def _auc(w, prof):
        v = r[(r["w"] == w) & (r["attack.profile"] == prof)]["auc"]
        if not len(v):
            v = d[(d["w"] == w) & (d["attack.profile"] == prof)]["auc"]
        return v.mean()
    M = np.array([[_auc(w, p) for p in PROFILES] for w in ws])
    gap = np.nanmax(np.abs(np.array(
        [[r[(r["w"] == w) & (r["attack.profile"] == p)]["auc"].mean()
          - d[~d["recal"]][(d[~d["recal"]]["w"] == w)
                           & (d[~d["recal"]]["attack.profile"] == p)]["auc"].mean()
          for p in PROFILES] for w in ws])))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(wspace=0.42)
    im = heat(axes[0], M, [PROF_LABEL[p] for p in PROFILES],
              [str(w) for w in ws], vmin=0.4, vmax=1.0)
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="AUC")
    axes[0].set_ylabel("weight on (key, neighbour, quantum)")
    axes[0].set_title("dropping $w_k$ kills exactly profile $k$;\n"
                      "re-weighting a present feature does almost nothing")

    y = np.arange(len(ws))
    for i, (flag, lab, col) in enumerate(((True, "threshold re-solved for these weights",
                                           C["B3"]),
                                          (False, "threshold left at its default",
                                           C["B1"]))):
        v = [d[(d["w"] == w) & (d["recal"] == flag)
               & (d["attack.profile"] == "P")]["fpr_node_tau05"].mean()
             for w in ws]
        axes[1].barh(y + (i - 0.5) * 0.38, v, 0.38, color=col, label=lab)
    axes[1].set_yticks(y, [str(w) for w in ws], fontsize=7.5)
    axes[1].invert_yaxis()          # match the heatmap row order
    axes[1].set_xlabel("false alarms on honest relays")
    axes[1].legend(fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.5, -0.13), ncol=2)
    axes[1].set_title("the same weights, judged on a\nthreshold-dependent metric")
    fig.suptitle("Q: how much does the detector depend on how we weight its "
                 "three signals?\nA: hardly at all, until one is dropped "
                 "entirely", y=1.06, fontsize=10.5)
    save(fig, "f13_weights.png", "weight sensitivity vs threshold sensitivity")


def fig_damage_floor(df):
    """Where the damage goes, and why DRR is reported on the relay share.

    A session whose own endpoint is compromised leaks however it is routed, so
    that block is set by the traffic matrix rather than by any policy. Quoting
    a policy against total damage caps it at a number it cannot influence - the
    oracle looks like a 20% improvement instead of a 98% one.
    """
    d = cell(df)
    labs = [("B0", {}, "No defence"), ("B1", {"policy.tau": 0.5}, "Threshold"),
            ("B3", {"policy.S_iso": 0.5, "policy.rho_start": QUIET}, "Graded (ours)"),
            ("B2", {"policy.m_paths": 2}, "Always redundant"), ("B4", {}, "Knows who is compromised")]
    d0 = d[d["policy.type"].isin(("B0", "none")) & d["attack.enabled"].astype(bool)]
    base = d0["D_eff"].mean()
    if not np.isfinite(base) or base <= 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    x = np.arange(len(labs))
    rel, endp = [], []
    for pol, kw, _ in labs:
        r = d0 if pol == "B0" else op(d, pol, **kw)
        rel.append(r["D_eff_relay"].mean() / base)
        endp.append(r["D_eff_endpoint"].mean() / base)
    axes[0].bar(x, rel, 0.6, color=C["B1"], label="relay exposure")
    axes[0].bar(x, endp, 0.6, bottom=rel, color="0.75",
                label="endpoint exposure")
    axes[0].set_xticks(x, [l for _, _, l in labs], rotation=20, fontsize=8)
    axes[0].set_ylabel(r"exposure, relative to no defence")
    axes[0].legend(fontsize=7.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.30), ncol=2)
    axes[0].set_title("where the exposure sits", fontsize=9)

    dr = [1 - r / rel[0] for r in rel]
    dt = [1 - (r + e) / (rel[0] + endp[0]) for r, e in zip(rel, endp)]
    axes[1].bar(x - 0.19, dr, 0.38, color=C["B3"],
                label="of relay exposure (what a policy can act on)")
    axes[1].bar(x + 0.19, dt, 0.38, color="0.6",
                label="of all exposure (includes the floor)")
    axes[1].set_xticks(x, [l for _, _, l in labs], rotation=20, fontsize=8)
    axes[1].set_ylabel("damage reduction")
    axes[1].legend(fontsize=7.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.30), ncol=1)
    axes[1].set_title("the same policies, judged on each")
    fig.suptitle("Q: why not just measure total exposure?\nA: part of it is "
                 "a floor that no routing decision can reach",
                 y=1.10, fontsize=10.5)
    save(fig, "f14_damage_floor.png", "relay vs endpoint exposure")


def fig_stability(df):
    """Did the policy behave, or did it thrash?

    An isolation policy that flaps is useless in a real controller whatever its
    DRR, because every flap is a routing rebuild and a burst of rerouted
    sessions. The hysteresis band and dwell timer exist for this, and B1 gets
    both so that the comparison is against a well-implemented binary policy.
    """
    d = cell(df)
    labs = [("B1", {"policy.tau": 0.35}, "Threshold, cuts early"),
            ("B1", {"policy.tau": 0.5}, "Threshold"),
            ("B1", {"policy.tau": 0.7}, "Threshold, cuts late"),
            ("B3", {"policy.S_iso": 0.4, "policy.rho_start": QUIET}, "Graded, cuts early"),
            ("B3", {"policy.S_iso": 0.5, "policy.rho_start": QUIET}, "Graded"),
            ("B3", {"policy.S_iso": 0.9, "policy.rho_start": QUIET}, "Graded, cuts late"),
            ("B4", {}, "Knows who is compromised")]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    x = np.arange(len(labs))
    for ax, col, lab in (
            (axes[0], "policy_mean_isolated", "relays cut off at any moment"),
            (axes[1], "policy_n_flaps",
             "times it changed its mind\nabout a relay"),
            (axes[2], "policy_n_veto_ticks",
             "times it wanted to cut a relay\nbut that would split the "
             "network")):
        v = [op(d, p, **kw)[col].mean() for p, kw, _ in labs]
        ax.bar(x, v, 0.6, color=[C[p] for p, _, _ in labs])
        ax.set_xticks(x, [l for _, _, l in labs], rotation=35, ha="right",
                      fontsize=7.5)
        ax.set_ylabel(lab)
    fig.suptitle("Q: is the defence stable enough for a real controller to "
                 "run?\nA: the graded policy cuts off fewer relays and "
                 "changes its mind less often", y=1.08, fontsize=10.5)
    save(fig, "f15_stability.png", "isolation load, flapping, guard vetoes")




# 5. the referee-facing figures
def fig_minimax(df):
    """F16. The central claim, stated as the question it answers.

    Every other comparison in the paper fixes the attack intensity, and the
    person fixing it is the author. Here the adversary picks, and the figure's
    job is to show what that does to a threshold.
    """
    d = cell(df, spec=False)
    d = d[np.isclose(d["demand.T_s_max"], 600.0)]
    knobs = [("G", "attack.gamma", 0.30), ("L", "attack.delta", 0.50),
             ("E", "attack.q", 0.03)]
    rows = [("B1", {"policy.tau": 0.5}, "Threshold (isolate at a line)", C["B1"]),
            ("B3", {"policy.S_iso": 0.5, "policy.rho_start": QUIET},
             "Graded (ours)", C["B3"]),
            ("B2", {"policy.m_paths": 2}, "Always redundant", C["B2"])]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), sharey=True)
    for ax, (prof, col, spec) in zip(axes, knobs):
        b = d[d["attack.profile"] == prof]
        xs = sorted(b[col].dropna().unique())
        frame_damage(ax, key=(prof == "G"))
        for pol, kw, lab, colour in rows:
            v = [op(b[np.isclose(b[col], x)], pol, **kw)["drr_relay"].mean()
                 for x in xs]
            ax.plot(range(len(xs)), v, marker="o", ms=4.5, lw=1.7, color=colour,
                    label=lab, zorder=3)
            j = int(np.nanargmin(v))
            ax.scatter([j], [v[j]], marker="v", s=110, color=colour,
                       edgecolor="white", linewidth=0.8, zorder=5)
            if pol in ("B1", "B3"):
                ax.annotate(f"worst {v[j]:+.2f}", (j, v[j]),
                            textcoords="offset points", xytext=(6, -13),
                            fontsize=7.5, color=colour, fontweight="bold")
        ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs], fontsize=8)
        ax.set_xlabel("quieter  " + r"$\longrightarrow$" + "  louder")
        ax.set_title(PROFILE_LABEL[prof].replace(chr(10), " - "), fontsize=9)
        ax.set_ylim(-0.12, 1.06)
    axes[0].set_ylabel(M_DAMAGE_SHORT)
    panel_tags(axes)
    axes[0].legend(fontsize=7.5, loc="upper center",
                   bbox_to_anchor=(1.72, -0.22), ncol=5)
    fig.suptitle("Q: what happens if the attacker chooses how loud to be?\n"
                 "A: the threshold policy drops to zero; the graded one does not",
                 y=1.10, fontsize=11)
    save(fig, "f16_minimax.png", "worst-case damage prevented, adversary chooses")


def fig_family(df):
    """F17. Does the lead survive a change of topology?

    Twelve 50-node members, three at each of four densities, every one
    calibrated on its own - its own lambda at 10 % rejection, its own phi, its
    own (lam_s, theta_0). Holding any of those fixed across the family would
    confound density with load.

    The oracle is not plotted. It isolates the compromised set and throttles
    nothing, so on the tapping adversary it sits below the graded policy, which
    also throttles; drawn as a line on a damage axis that reads as an ordering
    between methods rather than as the cost difference it is. Its numbers are
    in the results tables, beside its rejection rate, where they can be read
    for what they are. The frame here is 0 and 1, the two levels damage
    prevented genuinely cannot pass.

    Always-on redundancy is kept for context and leads on the greedy adversary.
    It is not free - it charges 0.43 to 0.66 extra key per delivered demand
    across this family, against the graded policy's -0.16 to +0.05 - but that
    price has two figures of its own and repeating it here only crowded the
    legend.
    """
    fam = df[df["topology.name"].astype(str).str.startswith("fam")
             & df["drr_relay"].notna()]
    if fam.empty:
        print("  f17 skipped: topology family not on disk")
        return
    for col, spec in (("attack.gamma", 0.30), ("attack.delta", 0.50),
                      ("attack.q", 0.03), ("attack.s", 0.25)):
        if col in fam.columns:
            fam = fam[np.isclose(fam[col].astype(float), spec)]
    fam = fam[np.isclose(fam["attack.f"], 0.10)
              & (fam["attack.selection"] == "random")]
    fam = fam.copy()
    fam["deg"] = fam["topology.name"].str.extract(r"fam(\d\d)").astype(float) / 10.0
    degs = sorted(fam["deg"].dropna().unique())

    rows = [("B1", {"policy.tau": 0.5}, "Threshold", C["B1"]),
            ("B3", {"policy.S_iso": 0.5}, "Graded (ours)", C["B3"]),
            ("B2", {"policy.m_paths": 2}, "Always redundant, 2 paths", C["B2"])]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    for ax, prof in zip(axes, ("G", "E")):
        b = fam[fam["attack.profile"] == prof]
        frame_damage(ax, key=(prof == "G"))
        for pol, kw, lab, col in rows:
            m, lo, hi = [], [], []
            for g in degs:
                r = op(b[np.isclose(b["deg"], g)], pol, **kw)
                v, l, h = ci(r["drr_relay"])
                m.append(v); lo.append(max(l, 0)); hi.append(max(h, 0))
            ax.errorbar(degs, m, yerr=[lo, hi], marker="o", ms=4.5, lw=1.6,
                        capsize=2.5, color=col, label=lab, zorder=3)
        ax.set_xlabel("network density (mean links per node)")
        ax.set_xticks(degs)
        ax.set_title(PROFILE_LABEL[prof].replace(chr(10), " - "), fontsize=9)
        ax.set_ylim(-0.10, 1.16)
    axes[0].set_ylabel(M_DAMAGE_SHORT)
    panel_tags(axes)
    axes[0].legend(fontsize=7.5, loc="upper center",
                   bbox_to_anchor=(1.05, -0.20), ncol=3)
    fig.suptitle("Q: does the graded policy keep its lead over the threshold "
                 "policy when the topology changes?\n"
                 "A: on all twelve members, and the margin is widest where the "
                 "detector struggles", y=1.06, fontsize=10.5)
    save(fig, "f17_family.png", "damage prevented against network density")

def fig_session_len(df):
    """When is it worth tearing down in-flight sessions instead of draining them?

    On the central cell the answer is "never, measurably" - but that cell has
    60-600 s sessions against a 3 h attack window, so almost nothing is in flight
    when a node is isolated. Sweeping E[T_s] over two orders of magnitude with
    lambda rescaled by its inverse holds the offered load and asks the question
    on the axis that actually decides it.
    """
    # cell() cannot be used here: it pins demand.lam, and lam is rescaled by the
    # inverse of E[T_s] at every point precisely so that this axis is length and
    # not length-times-load. Everything else cell() would pin is pinned by hand,
    # the attack intensity included - the spec row shares the central cell with
    # the intensity and minimax sweeps, and without that pin the E panel dips by
    # 0.2 at one point for no reason but the averaging.
    d = df[df["demand.T_s_max"].notna() & df["drr_relay"].notna()
           & (df["topology.name"] == "net50") & (df["demand.km_mode"] == "OTP")
           & (df["detector.x1_mode"] == "signed")
           & (df["detector.x2_mode"] == "signed")
           & np.isclose(df["attack.f"], 0.10)
           & np.isclose(df["noise.scale"], 1.0)
           & np.isclose(df["detector.alpha"], 0.05)
           & (df["attack.selection"] == "random")]
    for col, spec in (("attack.gamma", 0.30), ("attack.delta", 0.50),
                      ("attack.q", 0.03), ("attack.s", 0.25)):
        if col in d.columns:
            d = d[np.isclose(d[col].astype(float), spec)]
    if d.empty or d["demand.T_s_max"].nunique() < 3:
        print("  f18 skipped: no session-length runs on disk")
        return
    d = d.copy()
    d["ETs"] = (d["demand.T_s_min"] + d["demand.T_s_max"]) / 2
    ets = sorted(d["ETs"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharey=True)
    for ax, prof in zip(axes, ("G", "E")):
        b = d[d["attack.profile"] == prof]
        for pol, col, kw in (("B1", C["B1"], {"policy.tau": 0.5}),
                             ("B3", C["B3"], {"policy.S_iso": 0.5,
                                              "policy.rho_start": QUIET})):
            for td, ls, mk in ((False, "-", "o"), (True, "--", "s")):
                v = [op(b[(b["policy.tear_down_on_isolate"].astype(bool) == td)
                          & np.isclose(b["ETs"], e)], pol, **kw)
                     ["drr_relay"].mean() for e in ets]
                ax.plot(ets, v, marker=mk, ms=4, lw=1.5, ls=ls, color=col,
                        label=("Threshold" if pol == "B1"
                               else "Graded (ours)")
                              + (", tear down" if td else ", let them finish"))
        ax.set_xscale("log")
        ax.set_xticks(ets, [f"{e:.0f}" for e in ets], minor=False)
        ax.set_xticks([], minor=True)
        ax.set_xlabel("how long a session lasts (s)")
        ax.set_title(PROFILE_LABEL[prof].replace(chr(10), " - "), fontsize=9)
    axes[0].set_ylabel(M_DAMAGE_SHORT)
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper center",
                   bbox_to_anchor=(1.05, -0.20))
    fig.suptitle("Q: when a relay is cut off, should sessions already running "
                 "through it be torn down?\nA: only worth it once sessions "
                 "last as long as detection takes", y=1.08, fontsize=10.5)
    save(fig, "f18_session_len.png", "teardown value vs session length")




def fig_baselines(df):
    """F19. Coverage of the threat model, and what each method charges for it.

    The left panel is per adversary; the right panel collapses it to the two
    numbers an operator has to trade off, so the figure can be read without
    counting bars.
    """
    d = cell(df)
    d = d[np.isclose(d["demand.T_s_max"], 600.0)]
    if not (d["policy.type"] == "B5").any():
        print("  f19 skipped: baseline sweep not on disk")
        return
    rows = [("B7", {"policy.alpha_key": 0.5}, "Bi+ 2023\nkey-aware routing",
             "#bcbd22", True),
            ("B5", {"policy.beta_trust": 50.0}, "Luo & Li 2025\ntrust derating",
             "#e377c2", True),
            ("B1", {"policy.tau": 0.5}, "Threshold\n(ours)", C["B1"], False),
            ("B6", {"policy.m_paths": 2}, "Kiktenko+ 2024\nXOR multipath",
             "#17becf", True),
            ("B2", {"policy.m_paths": 2}, "Always redundant\n(ours)",
             C["B2"], False),
            ("B3", {"policy.S_iso": 0.5, "policy.rho_start": QUIET},
             "Graded\n(ours)", C["B3"], False),
            ("B8", {"policy.hybrid_tau": 0.5}, "Hybrid\n(ours)",
             "#ff7f0e", False)]
    rows = [r for r in rows if len(op(d, r[0], **r[1]))]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(wspace=0.30)

    # -- left: per adversary ------------------------------------------------ #
    x = np.arange(len(PROFILES))
    w = 0.82 / len(rows)
    frame_damage(axes[0], key=True)
    for i, (pol, kw, lab, col, pub) in enumerate(rows):
        vals = [ci(op(d[d["attack.profile"] == p], pol, **kw)["drr_relay"])
                for p in PROFILES]
        axes[0].bar(x + (i - (len(rows) - 1) / 2) * w, [v[0] for v in vals], w,
                    yerr=np.array([[max(v[1], 0) for v in vals],
                                   [max(v[2], 0) for v in vals]]),
                    capsize=1.2, color=col, label=lab.replace(chr(10), " "),
                    hatch="//" if pub else None, edgecolor="white",
                    linewidth=0.4, zorder=3)
    axes[0].set_xticks(x, [PROFILE_LABEL[p] for p in PROFILES], fontsize=8)
    axes[0].set_ylabel(M_DAMAGE_SHORT)
    axes[0].set_ylim(-0.10, 1.10)
    axes[0].legend(fontsize=7, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.20), columnspacing=0.9,
                   handlelength=1.2, title="hatched = published method",
                   title_fontsize=7)

    # -- right: coverage against price -------------------------------------- #
    placed: list[tuple[float, int]] = []
    for pol, kw, lab, col, pub in rows:
        r = op(d, pol, **kw)
        v = [op(d[d["attack.profile"] == p], pol, **kw)["drr_relay"].mean()
             for p in PROFILES]
        cov = sum(1 for a in v if a > 0.25)
        key = r["dKPD"].mean()
        axes[1].scatter([key], [cov], s=150, color=col, zorder=4,
                        marker="s" if pub else "o", edgecolor="white",
                        linewidth=1.0)
        seen_here = sum(1 for q in placed
                        if abs(q[0] - key) < 0.04 and q[1] == cov)
        placed.append((key, cov))
        short = lab.split(chr(10))[0]
        axes[1].annotate(short, (key, cov), textcoords="offset points",
                         xytext=(10, 5 - 14 * seen_here),
                         fontsize=7.5, color=col)
    axes[1].axhspan(3.5, 4.5, color="#2ca02c", alpha=0.06, zorder=0)
    # the band caption sits above the band, the "cheaper" arrow below it, so
    # neither lands on the other or on a point
    axes[1].text(0.99, 3.60, "covers the whole threat model", fontsize=7,
                 color="0.3", va="bottom", ha="right")
    axes[1].annotate("", xy=(0.06, 0.62), xytext=(0.55, 0.62),
                     arrowprops=dict(arrowstyle="->", color="0.45", lw=1.1))
    axes[1].text(0.305, 0.72, "cheaper", fontsize=8, color="0.45",
                 ha="center", va="bottom")
    axes[1].set_xlabel(M_COST_KEY)
    axes[1].set_ylabel("adversaries covered\n(damage prevented > 0.25, of 4)")
    axes[1].set_yticks(range(5))
    axes[1].set_ylim(-0.4, 4.7)
    axes[1].set_xlim(-0.09, 1.02)
    panel_tags(axes)

    fig.suptitle("Q: which methods cover the whole threat model, and what do "
                 "they charge?\nA: redundancy covers it and pays on every "
                 "session; the hybrid covers it and pays only where it matters",
                 y=1.09, fontsize=10.5)
    save(fig, "f19_baselines.png", "coverage against price, vs published methods")


def fig_hybrid(df):
    """F20. Protection against key spent, for the two parents and the hybrid.

    B2 buys its protection outright and pays for it on every session; B3 pays
    almost nothing and is blind wherever the detector is. The hybrid spends per
    session, so it traces a curve between them rather than sitting at a point,
    and the question the figure answers is whether that curve passes above the
    straight line joining its two parents.
    """
    d = cell(df)
    d = d[np.isclose(d["demand.T_s_max"], 600.0)]
    if not (d["policy.type"] == "B8").any():
        print("  f20 skipped: hybrid sweep not on disk")
        return
    taus = sorted(d[d["policy.type"] == "B8"]["policy.hybrid_tau"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # -- left: the pooled trade-off ---------------------------------------- #
    def pooled(rows):
        return rows["drr_relay"].mean(), rows["dKPD"].mean()

    b2 = pooled(op(d, "B2", **{"policy.m_paths": 2}))
    b3 = pooled(op(d, "B3", **{"policy.S_iso": 0.5, "policy.rho_start": QUIET,
                               "policy.kappa": 0.0}))
    hy = [pooled(op(d, "B8", **{"policy.hybrid_tau": t})) for t in taus]
    hx, hyv = [p[1] for p in hy], [p[0] for p in hy]

    axes[0].plot([b3[1], b2[1]], [b3[0], b2[0]], color="0.6", ls="--", lw=1.2,
                 zorder=1, label="linear blend of the two parents")
    axes[0].plot(hx, hyv, marker="o", ms=5, lw=1.8, color="#ff7f0e", zorder=3,
                 label="hybrid (ours), sweeping its trigger")
    for t, (y, x) in zip(taus, hy):
        axes[0].annotate(f"{t:g}", (x, y), textcoords="offset points",
                         xytext=(5, -9), fontsize=7, color="#ff7f0e")
    axes[0].scatter([b2[1]], [b2[0]], marker="s", s=48, color=C["B2"], zorder=4,
                    label="always redundant")
    axes[0].scatter([b3[1]], [b3[0]], marker="D", s=44, color=C["B3"], zorder=4,
                    label="graded only (ours)")
    axes[0].set_xlabel(M_COST_KEY)
    axes[0].set_ylabel(M_DAMAGE_SHORT + ", all four attackers")
    axes[0].legend(fontsize=7.5, loc="lower right")

    # -- right: where the key actually goes -------------------------------- #
    tau = 0.5 if 0.5 in taus else taus[len(taus) // 2]
    x = np.arange(len(PROFILES))
    w = 0.26
    for i, (pol, kw, lab, col) in enumerate((
            ("B2", {"policy.m_paths": 2}, "always redundant", C["B2"]),
            ("B3", {"policy.S_iso": 0.5, "policy.rho_start": QUIET,
                    "policy.kappa": 0.0}, "graded only", C["B3"]),
            ("B8", {"policy.hybrid_tau": tau}, "hybrid", "#ff7f0e"))):
        v = [op(d[d["attack.profile"] == p], pol, **kw)["mean_legs"].mean()
             for p in PROFILES]
        axes[1].bar(x + (i - 1) * w, v, w, color=col, label=lab)
    axes[1].axhline(1.0, color="0.4", lw=0.8)
    axes[1].set_xticks(x, [PROF_LABEL[p] for p in PROFILES], fontsize=8.5)
    axes[1].set_ylabel("paths used per session\n(2 = redundant, 1 = not)")
    axes[1].set_ylim(0.9, 2.12)
    axes[1].legend(fontsize=7.5, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, -0.13), columnspacing=1.1,
                   handlelength=1.4)
    axes[1].set_title("the hybrid spends where the detector is blind",
                      loc="left", fontsize=9)

    panel_tags(axes)
    fig.suptitle("Q: can we get redundancy's cover without paying for it "
                 "on every session?\nA: yes. It reaches all four "
                 "attackers for 40% of the key", y=1.07, fontsize=10.5)
    save(fig, "f20_hybrid.png", "hybrid trade-off and where its key goes")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(SRC)
    print(f"{len(df)} runs from {SRC}\n")
    for fn in (fig_ablation, fig_intensity, fig_x2_mode, fig_separation,
               fig_grid, fig_targeting, fig_cost, fig_quiet,
               fig_ofat, fig_f_curve, fig_kappa, fig_weights, fig_damage_floor,
               fig_baselines, fig_hybrid,
               fig_stability, fig_minimax, fig_family, fig_session_len):
        try:
            fn(df)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
    print(f"\nwritten to {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
