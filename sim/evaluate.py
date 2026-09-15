"""Detector evaluation, independent of any policy.

Phase 4 reference: section 4.5 of phase3-4-build-spec.

Everything here is a pure function over arrays, so the same code serves two
callers: the runner, which evaluates at full T_sample resolution while the run
is still in memory, and the CLI at the bottom, which works offline from
``{run_id}_events.parquet``.  The event log is sampled at ``telemetry_period``
(300 s) rather than ``T_sample`` (30 s), so the offline numbers are the coarse
view of the same thing - the summary row carries the authoritative ones.

Ground truth label per (node, sample):

    label[i, t] = 1 if compromised[i] and t >= t_compromise[i] else 0

Profile E labelling, stated explicitly because it is a choice: the *compromised
node* is the positive, not both ends of the tapped link.  The system scores
nodes and the policy levers act on nodes.  The healthy far end is labelled
negative, so if the detector lights it up that is a genuine false positive and
it should show up in the results as one.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

INF = float("inf")
DEFAULT_TAUS = (0.5, 0.7, 0.9)


# --------------------------------------------------------------------------- #
# core statistics
# --------------------------------------------------------------------------- #
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U statistic, with proper average ranks for ties.

    Ties matter here: during detector warm-up every node carries exactly its
    prior, and dropping tie correction would quietly bias those samples.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).astype(bool).ravel()
    n = scores.size
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    avg_rank = first + (counts - 1) / 2.0 + 1.0
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.repeat(avg_rank, counts)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(scores: np.ndarray, labels: np.ndarray,
              n_points: int = 101) -> dict[str, list[float]]:
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).astype(bool).ravel()
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"tau": [], "fpr": [], "tpr": []}
    taus = np.linspace(0.0, 1.0, n_points)
    tpr, fpr = [], []
    for tau in taus:
        hit = scores >= tau
        tpr.append(float(hit[labels].sum() / n_pos))
        fpr.append(float(hit[~labels].sum() / n_neg))
    return {"tau": taus.tolist(), "fpr": fpr, "tpr": tpr}


def labels_for(t: np.ndarray, compromised: np.ndarray,
               t_compromise: np.ndarray) -> np.ndarray:
    """(T, |V|) boolean ground truth."""
    t = np.asarray(t, dtype=float)[:, None]
    return np.asarray(compromised, dtype=bool)[None, :] & (t >= np.asarray(
        t_compromise, dtype=float)[None, :])


def detection_delay(t: np.ndarray, s_bar: np.ndarray, compromised: np.ndarray,
                    t_compromise: np.ndarray, tau: float) -> dict:
    """First crossing of tau after compromise, per compromised node.

    A node that never crosses is recorded as **censored**, never as infinity and
    never silently dropped: replacing it with infinity destroys the mean and
    dropping it inflates the apparent speed.  The reportable pair is
    'median delay among the detected' plus 'detection rate'.
    """
    t = np.asarray(t, dtype=float)
    comp = np.flatnonzero(np.asarray(compromised, dtype=bool))
    delays: list[float] = []
    censored = 0
    for i in comp:
        tc = float(t_compromise[i])
        after = t >= tc
        if not after.any():
            censored += 1
            continue
        crossed = np.flatnonzero(after & (s_bar[:, i] >= tau))
        if crossed.size == 0:
            censored += 1
        else:
            delays.append(float(t[crossed[0]] - tc))
    n_total = comp.size
    return {
        "tau": float(tau),
        "n_compromised": int(n_total),
        "n_detected": len(delays),
        "detection_rate": (len(delays) / n_total) if n_total else float("nan"),
        "n_censored": int(censored),
        "median_delay_s": float(np.median(delays)) if delays else float("nan"),
        "mean_delay_s": float(np.mean(delays)) if delays else float("nan"),
        "delays_s": delays,
    }


def false_positives(t: np.ndarray, s_bar: np.ndarray, compromised: np.ndarray,
                    tau: float) -> dict:
    """Healthy nodes that crossed tau, by node and by sample."""
    healthy = ~np.asarray(compromised, dtype=bool)
    n_h = int(healthy.sum())
    if n_h == 0:
        return {"fpr_node": float("nan"), "fpr_sample": float("nan"),
                "n_false_isolations": 0}
    block = s_bar[:, healthy] >= tau
    ever = block.any(axis=0)
    return {
        "fpr_node": float(ever.sum() / n_h),
        "fpr_sample": float(block.mean()),
        "n_false_isolations": int(ever.sum()),
    }


# --------------------------------------------------------------------------- #
# the whole report
# --------------------------------------------------------------------------- #
def evaluate(t: np.ndarray, s_bar: np.ndarray, compromised: np.ndarray,
             t_compromise: np.ndarray, taus=DEFAULT_TAUS,
             with_curve: bool = False) -> dict:
    """Full report for one score series. ``s_bar`` has shape (T, |V|)."""
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return {"auc": float("nan"), "n_samples": 0}
    lab = labels_for(t, compromised, t_compromise)
    out: dict = {
        "auc": roc_auc(s_bar, lab),
        "n_samples": int(t.size),
        "n_positive": int(lab.sum()),
        "mean_S_bar_compromised": float(s_bar[:, np.asarray(compromised, bool)].mean())
        if np.any(compromised) else float("nan"),
        "mean_S_bar_healthy": float(s_bar[:, ~np.asarray(compromised, bool)].mean())
        if np.any(~np.asarray(compromised, bool)) else float("nan"),
    }
    for tau in taus:
        key = f"{tau:g}".replace(".", "")
        d = detection_delay(t, s_bar, compromised, t_compromise, tau)
        f = false_positives(t, s_bar, compromised, tau)
        out[f"det_rate_tau{key}"] = d["detection_rate"]
        out[f"median_delay_tau{key}"] = d["median_delay_s"]
        out[f"n_censored_tau{key}"] = d["n_censored"]
        out[f"fpr_node_tau{key}"] = f["fpr_node"]
        out[f"fpr_sample_tau{key}"] = f["fpr_sample"]
        out[f"false_isolations_tau{key}"] = f["n_false_isolations"]
    if with_curve:
        out["roc"] = roc_curve(s_bar, lab)
    return out


def evaluate_detector(detector, state, taus=DEFAULT_TAUS) -> dict:
    """In process evaluation at full T_sample resolution.

    Returns one flat dict: the main feature set unprefixed, each ablation subset
    under ``abl_x1x3_auc`` and friends.
    """
    t, s_all = detector.score_history()
    if t.size == 0:
        return {"auc": float("nan"), "n_samples": 0}
    comp = np.asarray(state.compromised, dtype=bool)
    tc = np.asarray(state.t_compromise, dtype=float)

    out = evaluate(t, s_all[:, detector.main_index, :], comp, tc, taus)
    out["detector_warm_at"] = detector.warm_at
    for k, subset in enumerate(detector.subsets):
        name = "".join(subset)
        out[f"abl_{name}_auc"] = roc_auc(s_all[:, k, :],
                                         labels_for(t, comp, tc))
    return out


def ablation_table(detector, state, taus=DEFAULT_TAUS) -> dict[str, dict]:
    """Per feature subset report, for the diagonal table of section 4.5."""
    t, s_all = detector.score_history()
    comp = np.asarray(state.compromised, dtype=bool)
    tc = np.asarray(state.t_compromise, dtype=float)
    return {"".join(sub): evaluate(t, s_all[:, k, :], comp, tc, taus)
            for k, sub in enumerate(detector.subsets)}


# --------------------------------------------------------------------------- #
# offline CLI over events.parquet
# --------------------------------------------------------------------------- #
def from_events(path: str, score_column: str = "S_bar",
                taus=DEFAULT_TAUS, with_curve: bool = False) -> dict:
    """Reconstruct the score matrix from an events table and evaluate it.

    The event log is sampled at ``telemetry_period``, coarser than the detector's
    own ``T_sample``, so detection delay is quantised to that period.  Use the
    summary row for the authoritative figures.
    """
    import pyarrow.parquet as pq

    tab = pq.read_table(path)
    cols = {name: np.asarray(tab.column(name).to_numpy(zero_copy_only=False))
            for name in ("t", "node_id", score_column, "compromised", "t_compromise")}
    times = np.unique(cols["t"])
    nodes = np.unique(cols["node_id"])
    idx_t = np.searchsorted(times, cols["t"])
    idx_n = np.searchsorted(nodes, cols["node_id"])

    s_bar = np.full((times.size, nodes.size), np.nan, dtype=float)
    s_bar[idx_t, idx_n] = cols[score_column]

    comp = np.zeros(nodes.size, dtype=bool)
    tc = np.full(nodes.size, INF, dtype=float)
    comp[idx_n] = cols["compromised"].astype(bool)
    tc[idx_n] = cols["t_compromise"]

    keep = ~np.isnan(s_bar).any(axis=1)
    return evaluate(times[keep], s_bar[keep], comp, tc, taus, with_curve)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sim.evaluate",
        description="Offline detector evaluation over an events parquet.")
    ap.add_argument("events", help="path to {run_id}_events.parquet")
    ap.add_argument("--score", default="S_bar", help="score column (default S_bar)")
    ap.add_argument("--taus", default="0.5,0.7,0.9")
    ap.add_argument("--curve", action="store_true", help="include the ROC curve")
    ap.add_argument("--json", action="store_true", help="dump raw JSON")
    args = ap.parse_args(argv)

    taus = tuple(float(x) for x in args.taus.split(","))
    report = from_events(args.events, args.score, taus, args.curve)
    if args.json:
        print(json.dumps(report, indent=2, default=float))
        return 0
    for key, value in report.items():
        if key == "roc":
            continue
        print(f"{key:28s}{value:.4f}" if isinstance(value, float)
              else f"{key:28s}{value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
