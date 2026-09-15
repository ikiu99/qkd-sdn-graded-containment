"""Posterior risk estimate S_i(t).

S is not a detector output, it is a posterior risk estimate: evidence from three
orthogonal channels, mapped onto a common scale, combined with a configuration
prior, and smoothed.

Chain (section 4.3):

    1. robust standardisation on a dual baseline
           z = max(z_temporal, z_peer)          median / MAD, never mean / std,
                                                because the attack contaminates
                                                the mean and the deviation
    2. evidence      e_k = clip(z / z_max, 0, 1)
    3. score         S   = sigmoid(lam_s * (sum_k w_k e_k - theta_0) + logit(prior))
    4. smoothing     S_bar = alpha S + (1 - alpha) S_bar_prev,  S_bar(0) = prior

The detector reads the state through :mod:`telemetry` only, and that module may
not touch a single ``*_true`` field.

Ablation: all seven feature subsets are scored from the *same* evidence matrix
in the same pass. Under NullPolicy nothing the detector produces feeds back
into the simulation, so the seven variants are exact - not an approximation of
seven separate runs, but identical to them.
"""
from __future__ import annotations

import numpy as np

from .config import DetectorConfig
from .interfaces import Detector
from .state import NetworkState
from .telemetry import TelemetryCollector
from .topology import Topology

EPS = 1e-9
MAD_SCALE = 1.4826
# Relative floor on the MAD. A feature that is constant across the whole base
# window has MAD = 0 and would send z to infinity; eps = 1e-9 is not enough,
# the floor has to scale with the median (section 4.3, "numerical traps").
MAD_FLOOR_REL = 0.01

FEATURE_SUBSETS: tuple[tuple[str, ...], ...] = (
    ("x1",), ("x2",), ("x3",),
    ("x1", "x2"), ("x1", "x3"), ("x2", "x3"),
    ("x1", "x2", "x3"),
)
FEATURE_INDEX = {"x1": 0, "x2": 1, "x3": 2}


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def build_prior(topo: Topology, cfg: DetectorConfig,
                key_flow: np.ndarray | None, seed_topology: int) -> np.ndarray:
    """pi_i = sigmoid(c0 + c1*Exposure + c2*Phi + c3*Age), phase 1 section 7.

    Phi is the baseline key *flow* through the node, deliberately not degree
    centrality: key rate, not node degree, is the right measure of how much a
    relay is worth compromising.

    Bias warning that belongs in the analysis: under
    ``attack.selection='top_keyflow'`` the compromised nodes are by construction
    the high-Phi ones, so the prior guesses them right for free and the AUC is
    inflated. Always report random and top_keyflow separately, never averaged.
    """
    n = topo.n_nodes
    exposure = np.asarray(topo.exposure, dtype=float)
    phi = (minmax(key_flow) if key_flow is not None else np.zeros(n, dtype=float))
    # own substream: adding Age must not move the exposure frozen in the topology
    age = np.random.default_rng([int(seed_topology), 777]).uniform(0.0, 1.0, n)
    logit = cfg.c0 + cfg.c1 * exposure + cfg.c2 * phi + cfg.c3 * age
    return sigmoid(logit)


class SuspicionDetector(Detector):
    def __init__(self, cfg: DetectorConfig, n_nodes: int, topo: Topology,
                 key_flow: np.ndarray | None = None, seed_topology: int = 1,
                 **_: object):
        self.cfg = cfg
        self.topo = topo
        self.n_nodes = int(n_nodes)
        self.collector = TelemetryCollector(topo, cfg)

        self.prior = build_prior(topo, cfg, key_flow, seed_topology)
        self._prior_logit = np.log(self.prior / (1.0 - self.prior))

        self.n_base = int(round(cfg.W_base / cfg.T_sample))
        self._hist = np.zeros((self.n_base, self.n_nodes, 3), dtype=float)
        self._pos = 0
        self._filled = 0

        # latest sample, read by the telemetry logger
        self.x = np.zeros((self.n_nodes, 3), dtype=float)
        self.e = np.zeros((self.n_nodes, 3), dtype=float)
        self.S = self.prior.copy()

        # one S_bar per ablation subset; index -1 is the full feature set
        self._subsets = (FEATURE_SUBSETS if cfg.ablation
                         else (tuple(cfg.features),))
        self._weights = [self._subset_weights(s) for s in self._subsets]
        self.S_bar_variants = np.repeat(self.prior[None, :], len(self._subsets), axis=0)
        self.main_index = self._subsets.index(tuple(cfg.features)) \
            if tuple(cfg.features) in self._subsets else len(self._subsets) - 1

        # score history, for the offline evaluation harness
        self.hist_t: list[float] = []
        self.hist_S_bar: list[np.ndarray] = []
        self.warm_at: float | None = None
        # evidence history, only for the calibration pilot run - the operating
        # point has to be set against the MEASURED noise floor of a healthy node,
        # not against an idealised e = 0
        self.keep_evidence = bool(getattr(cfg, "keep_evidence", False))
        self.hist_e: list[np.ndarray] = []

    def _subset_weights(self, subset: tuple[str, ...]) -> np.ndarray:
        """Renormalise the configured weights onto the active subset.

        The constraint of phase 1 is sum(w) = 1; an ablation that simply drops a
        term would also shift the effective operating point theta_0, and the
        ablation table would then confound 'this feature is blind here' with
        'the threshold moved'.
        """
        w = np.zeros(3, dtype=float)
        for name in subset:
            w[FEATURE_INDEX[name]] = self.cfg.weights[FEATURE_INDEX[name]]
        total = w.sum()
        return w / total if total > 0 else w

    def initial_S_bar(self, n_nodes: int) -> np.ndarray:
        return self.prior.copy()

    @property
    def warm(self) -> bool:
        return self._filled >= self.n_base

    def update(self, state: NetworkState, topo: Topology, t: float) -> np.ndarray:
        x = self.collector.collect(state, topo, t)
        self.x = x

        self._hist[self._pos] = x
        self._pos = (self._pos + 1) % self.n_base
        self._filled += 1

        if not self.warm:
            # Detector warm-up: until the base window is full there is no robust
            # baseline, so the score is the prior alone and nothing is evaluated.
            self.e = np.zeros_like(x)
            self.S = self.prior.copy()
            self.S_bar_variants[:] = self.prior[None, :]
            return self.S
        if self.warm_at is None:
            self.warm_at = float(t)

        self.e = self._evidence(x)
        self.S = self._score(self.e, self._weights[self.main_index])

        a = self.cfg.alpha
        for k, w in enumerate(self._weights):
            s_k = self.S if k == self.main_index else self._score(self.e, w)
            self.S_bar_variants[k] = a * s_k + (1.0 - a) * self.S_bar_variants[k]

        self.hist_t.append(float(t))
        self.hist_S_bar.append(self.S_bar_variants.copy())
        if self.keep_evidence:
            self.hist_e.append(self.e.copy())
        return self.S

    def _evidence(self, x: np.ndarray) -> np.ndarray:
        """Dual baseline robust standardisation, then the map onto [0,1]."""
        # temporal: against the node's own history
        med_t = np.median(self._hist, axis=0)                       # (|V|,3)
        mad_t = np.median(np.abs(self._hist - med_t), axis=0)
        mad_t = np.maximum(mad_t, MAD_FLOOR_REL * np.abs(med_t) + EPS)
        z_temp = (x - med_t) / (MAD_SCALE * mad_t)

        # peer: against the cohort of all nodes at this instant
        med_p = np.median(x, axis=0)                                # (3,)
        mad_p = np.median(np.abs(x - med_p), axis=0)
        mad_p = np.maximum(mad_p, MAD_FLOOR_REL * np.abs(med_p) + EPS)
        z_peer = (x - med_p) / (MAD_SCALE * mad_p)

        # the first catches a node whose own behaviour changed, the second a node
        # that differs from everyone else; a persistent attack eventually leaks
        # into the temporal baseline, and the peer branch is what keeps it visible
        z = np.maximum(z_temp, z_peer)
        # only an upward excursion is suspicious
        np.clip(z, 0.0, self.cfg.z_max, out=z)
        return z / self.cfg.z_max

    def _score(self, e: np.ndarray, w: np.ndarray) -> np.ndarray:
        logit = self.cfg.lam_s * (e @ w - self.cfg.theta_0)
        if self.cfg.use_prior:
            logit = logit + self._prior_logit
        return sigmoid(logit)

    def score_history(self) -> tuple[np.ndarray, np.ndarray]:
        """(t of shape (T,), S_bar of shape (T, n_subsets, |V|))."""
        if not self.hist_t:
            return (np.zeros(0, dtype=float),
                    np.zeros((0, len(self._subsets), self.n_nodes), dtype=float))
        return np.array(self.hist_t, dtype=float), np.stack(self.hist_S_bar)

    @property
    def subsets(self) -> tuple[tuple[str, ...], ...]:
        return self._subsets
