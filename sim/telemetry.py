"""Raw detection features x1, x2, x3.

Phase 4 reference: section 4.2 of phase3-4-build-spec.

ABSOLUTE ACCESS CONSTRAINT.  Everything in this module may read only what the
SDN controller is actually served over ETSI GS QKD 015/018:

    state.qber_obs, state.R_obs, state.buffer_reported, state.ledger_consumed
    and the topology.

Reading ``state.buffers``, any ``*_true`` field, or ``compromised`` is a bug.
``test_detector_isolation`` poisons exactly those fields with NaN and asserts
the features come out unchanged.

Three periods, not one (section 4.1): features are computed every ``T_sample``
(30 s) over an overlapping window ``W_feat`` (300 s), and the robust baseline in
``detector.py`` runs over ``W_base`` (3600 s).  Sampling the features at 300 s
instead would leave the baseline with 6 samples, and a median/MAD over 6 points
is not a statistic.
"""
from __future__ import annotations

import numpy as np

from .config import DetectorConfig
from .state import NetworkState
from .topology import Topology

EPS = 1e-12
SQRT2 = 1.4142135623730951
MAD_SCALE = 1.4826
MAD_FLOOR_REL = 0.01


def _peer_z(v: np.ndarray) -> np.ndarray:
    """Robust z of each link against the cohort of links at this instant."""
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    mad = max(float(mad), MAD_FLOOR_REL * abs(float(med)) + EPS)
    return (v - med) / (MAD_SCALE * mad)


class TelemetryCollector:
    """Computes the three raw features once per detector sample."""

    def __init__(self, topo: Topology, cfg: DetectorConfig):
        self.topo = topo
        self.cfg = cfg
        self.n_nodes = topo.n_nodes
        self.n_edges = topo.n_edges
        self.W_feat = float(cfg.W_feat)
        self.B_max = float(topo.B_max)
        self._x1_s: np.ndarray | None = None      # x1_ema state, see _x1
        # x1_mode='cusum' state: the frozen (mu, sigma) reference, the
        # samples still being collected for it, and the running statistic
        self._cusum_ref: tuple[np.ndarray, np.ndarray] | None = None
        self._cusum_buf: list[np.ndarray] = []
        self._cusum_c: np.ndarray | None = None

        # Flattened node -> incident (edge, side) map, so every feature is a
        # couple of vector ops plus one reduceat instead of a Python loop.
        edges, sides, offsets = [], [], []
        cursor = 0
        ends = topo.edge_ends
        for i in range(topo.n_nodes):
            inc = topo.node_edges[i]
            offsets.append(cursor)
            for e in inc:
                edges.append(int(e))
                sides.append(0 if ends[e, 0] == i else 1)
            cursor += int(inc.size)
        self.flat_edges = np.array(edges, dtype=np.int64)
        self.flat_sides = np.array(sides, dtype=np.int64)
        self.offsets = np.array(offsets, dtype=np.int64)
        self.deg = topo.node_deg.astype(float)

        # ring buffer holding the last W_feat/T_sample samples
        self.depth = int(round(cfg.W_feat / cfg.T_sample)) + 1
        self._rep = np.zeros((self.depth, topo.n_edges, 2), dtype=float)
        self._ledger = np.zeros((self.depth, topo.n_edges), dtype=float)
        self._pos = 0
        self._n = 0

        # long run per edge EMA behind x3, separate from the feature baseline
        self._ema_qber: np.ndarray | None = None
        self._ema_skr: np.ndarray | None = None     # on R_obs / R_nominal
        self._dev_qber: np.ndarray | None = None    # EMA of |deviation|, the scale
        self._dev_skr: np.ndarray | None = None
        self._alpha_ema = float(cfg.T_sample) / float(cfg.W_ema)

    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        """True once the W_feat window is fully populated."""
        return self._n >= self.depth

    def _push(self, state: NetworkState) -> None:
        self._rep[self._pos] = state.buffer_reported
        self._ledger[self._pos] = state.ledger_consumed
        self._pos = (self._pos + 1) % self.depth
        self._n += 1

    def _oldest(self) -> int:
        return self._pos if self._n >= self.depth else 0

    # ------------------------------------------------------------------ #
    def collect(self, state: NetworkState, topo: Topology, t: float) -> np.ndarray:
        """Return the raw (|V|, 3) feature matrix and advance the window."""
        self._push(state)
        old = self._oldest()
        rep_now = state.buffer_reported
        rep_past = self._rep[old]
        ledger_delta = state.ledger_consumed - self._ledger[old]

        out = np.zeros((self.n_nodes, 3), dtype=float)
        if self.ready:
            out[:, 0] = self._x1(state, rep_now, rep_past, ledger_delta)
            out[:, 1] = self._x2(rep_now)
        out[:, 2] = self._x3(state)
        return out

    # ------------------------------------------------------------------ #
    def _x1(self, state: NetworkState, rep_now: np.ndarray, rep_past: np.ndarray,
            ledger_delta: np.ndarray) -> np.ndarray:
        """Key accounting residual.

        K_obs is *inferred* from generation minus the change in the reported
        buffer, never read from a consumption report: a malicious node can lie
        about what it consumed, but the buffer level and the key rate reach the
        controller from two independent sources.

        Buffer saturation has to be accounted for, and this is not a detail.  A
        link sitting at B_max discards the key it generates, so the naive
        ``R*W - dB`` reads the whole discarded amount as consumption and every
        idle-but-full link becomes a permanent false positive.  The controller
        knows B_max and knows what it authorised, so it can bound the key the
        link could actually have *stored*:

            G_hat = min(R_obs*W, headroom_at_window_start + K_exp)
            K_obs = G_hat - (B_now - B_past)

        On an unsaturated link the cap is inactive and this is the plain
        estimator.  On a saturated link it collapses to K_obs == K_exp, i.e. a
        residual of zero - which is the honest answer: key stolen from a link
        that refills to the brim leaves no trace in the buffer level.  F1 is
        therefore observable only where links run near capacity, and that is a
        real property of the mechanism, worth stating in the paper rather than
        hiding behind an estimator that produces confident nonsense.
        """
        e, side = self.flat_edges, self.flat_sides
        rep_p = rep_past[e, side]
        d_rep = rep_now[e, side] - rep_p

        gen_raw = state.R_obs[e] * self.W_feat
        headroom = np.maximum(self.B_max - rep_p, 0.0)
        gen_hat = np.minimum(gen_raw, headroom + ledger_delta[e])

        k_obs = np.add.reduceat(np.maximum(gen_hat - d_rep, 0.0), self.offsets)
        k_exp = np.add.reduceat(ledger_delta[e], self.offsets)
        # Standardised, not relative.  The spec writes x1 = |K_obs-K_exp| /
        # (K_exp + eps); that form diverges on an idle node (K_exp -> 0, so pure
        # report noise is divided by eps) and shrinks on a busy one (large
        # denominator), which inverts the ranking - measured here at AUC 0.22
        # for profile G on T3 before the change.  Dividing by the residual the
        # controller should *expect* instead makes x1 a signal to noise ratio:
        # ~1 wherever nothing is wrong, regardless of how much traffic the node
        # carries, and >1 only on genuine excess consumption.
        #
        # Two independent report draws per edge give sqrt(2) sigma on dB, and the
        # unmanaged background stream contributes a share of K_exp.
        sigma_rep = self.cfg.report_sigma * rep_now[e, side] * SQRT2
        sigma_node = np.sqrt(np.add.reduceat(sigma_rep ** 2, self.offsets)
                             + (self.cfg.unmanaged_assumed * k_exp) ** 2)
        # Signed for the same reason x2 is.  ``abs`` is the literal phase 1
        # formula, and it makes the feature two-sided: a node that consumed
        # LESS than the controller authorised - an idle node, or one whose
        # buffer report happened to drift up - scores exactly as high as one
        # that drained extra key, though only the second is an attack.  Every
        # attack that moves x1 at all moves it upwards (G draws more than the
        # ledger records), so the sign carries the whole of the discrimination
        # and the magnitude carries half of it plus a symmetric noise channel.
        # As in x2, it is left unclipped here: the detector clips z at zero, and
        # clipping twice would compress the MAD that the clip is measured against.
        r = k_obs - k_exp
        if self.cfg.x1_mode == "abs":
            r = np.abs(r)
        r = r / (sigma_node + EPS)

        # Pre-standardisation EMA.  Default OFF, and the default is the result:
        # this is a measured dead end kept as a config field so that the negative
        # result is reproducible rather than folklore.
        #
        # The idea was sound on paper.  The two sources x1 cannot separate differ
        # in their time signature - background traffic is episodic and zero mean,
        # a greedy relay drains on every sample from t_c - so averaging the
        # residual should shrink the episodic part as 1/sqrt(n) and leave the
        # persistent part alone.  Measured on profile G it does almost nothing
        # (x1 AUC 0.657 -> 0.666 at a 1500 s window) while the false-positive
        # rate on profile P goes 0.078 -> 0.329.
        #
        # The reason is that the baseline is computed from the same series.
        # Smoothing makes the series autocorrelated, so the median/MAD taken over
        # W_base underestimates its spread by very nearly the same factor the
        # noise shrank by; the numerator and the denominator both fall and the z
        # barely moves, except that every node now looks more extreme against its
        # own deflated MAD.  Aggregating in time cannot help a statistic whose
        # scale is estimated from its own history.  A method that works would
        # need to hold the baseline fixed - a CUSUM on the evidence, which is a
        # different detector, not a tuning of this one.
        if self.cfg.x1_ema > 0.0:
            a = self.cfg.x1_ema
            self._x1_s = r if self._x1_s is None else a * r + (1.0 - a) * self._x1_s
            r = self._x1_s

        if self.cfg.x1_mode == "cusum":
            r = self._cusum(r)
        return r

    def _cusum(self, r: np.ndarray) -> np.ndarray:
        """Accumulate the residual against a reference frozen at warm-up.

        This is the method the EMA above could not be: the failure there was
        that the baseline is re-estimated from the same smoothed series, so
        numerator and denominator shrink together.  A CUSUM holds the reference
        FIXED while evidence accumulates, which is the textbook answer to a
        small persistent shift buried in episodic noise -- exactly the shape of
        a greedy relay against honest background traffic.

            C_t = max(0, C_{t-1} + (r_t - mu_0)/sigma_0 - k)

        ``mu_0`` and ``sigma_0`` come from the first ``n_ref`` samples, which
        under the experiment design lie inside the warm-up window and therefore
        before t_c.  **That is an assumption and the paper states it**: it takes
        the network to have been clean when it was commissioned.  It is the
        weaker of the two options only in that a relay compromised from the
        first second would be folded into its own reference; against that
        adversary the feature degrades to no worse than the plain residual,
        because a constant offset is exactly what the reference absorbs.

        ``k`` is the slack that keeps an honest node's statistic at zero instead
        of letting rounding walk it upward.  One-sided by construction, which is
        the same argument the signed residual rests on.

        The output is a ramp, not a level, so its own temporal median/MAD drifts
        with it -- but the detector takes the maximum of the temporal and peer
        baselines, and the peer branch is exactly where a ramp shows: a
        compromised node climbs while its peers stay at zero.
        """
        k = float(self.cfg.x1_cusum_k)
        if self._cusum_ref is None:
            self._cusum_buf.append(r.copy())
            if len(self._cusum_buf) < self.cfg.x1_cusum_ref:
                return np.zeros_like(r)
            hist = np.stack(self._cusum_buf)                 # (n_ref, n_nodes)
            mu = np.median(hist, axis=0)
            mad = np.median(np.abs(hist - mu), axis=0) * MAD_SCALE
            self._cusum_ref = (mu, np.maximum(mad, 1e-3))
            self._cusum_buf = []
            self._cusum_c = np.zeros_like(r)
        mu, sigma = self._cusum_ref
        step = (r - mu) / sigma - k
        lam = float(self.cfg.x1_cusum_decay)
        self._cusum_c = np.maximum(0.0, lam * self._cusum_c + step)
        return self._cusum_c.copy()

    def _x2(self, rep_now: np.ndarray) -> np.ndarray:
        """Neighbour discrepancy: the two ends of a link report the same buffer.

        ``abs`` is the literal formula of phase 1 section 5: the mean absolute
        gap over the adjacent links.  It has a structural false positive built
        in - the gap belongs to the *link*, so the honest neighbour of a liar
        sees exactly the same magnitude and lights up just as brightly.  Measured
        on profile L that put the node-level false positive rate at 0.29 (f=0.05)
        rising to 0.71 (f=0.30).

        ``signed`` disambiguates.  A liar inflates *every* link it touches, so
        from its own side the gap is positive on all of them; its honest
        neighbour sees one negative gap on the shared link and nothing on the
        rest.  Taking the positive part of the signed gap from each node's own
        side therefore attributes the discrepancy to the end that is inflating,
        and it needs no new information - only the side index the controller
        already has.  Consistent with the rest of the chain, where only an upward
        excursion counts as evidence.
        """
        e, side = self.flat_edges, self.flat_sides
        if self.cfg.x2_mode == "abs":
            d = np.abs(rep_now[e, 0] - rep_now[e, 1]) / self.B_max
        else:
            # Signed, and deliberately NOT clipped here.  Clipping at the edge
            # level zeroes half of a symmetric noise sample, which compresses the
            # MAD of the downstream robust normalisation and inflates the
            # positive tail into false evidence - measured as FPR rising from
            # 0.11 to 0.16 on profiles with no liar at all.  The detector already
            # clips z at zero, so one-sidedness is applied once, in the right
            # place, on a statistic whose spread is still meaningful.
            d = (rep_now[e, side] - rep_now[e, 1 - side]) / self.B_max
        return np.add.reduceat(d, self.offsets) / self.deg

    def _x3(self, state: NetworkState) -> np.ndarray:
        """Quantum layer deviation, worst adjacent link.

        max over the incident links, not mean: a tap sits on a single link and
        averaging over the degree dilutes it away.

        Dual baseline at the *link* level, and this is a necessary addition
        rather than a flourish.  Section 4.2 defines the deviation against a
        per link moving average alone; with a persistent tap that average
        absorbs the attack within ~W_ema and the signal disappears for the rest
        of the run.  Measured: AUC 0.55 for profile E, because the evidence only
        survived about a tenth of the post-compromise window.  The peer branch -
        the same max(temporal, peer) construction section 6 already prescribes
        one level up - compares each link against the cohort of links at the
        same instant, so a tap that is permanent stays permanently visible.

        The SKR peer comparison runs on R_obs / R_nominal, not on R_obs: link
        rates differ by an order of magnitude across lengths, and the nominal
        rate is commissioning data the operator has anyway.  QBER is compared
        raw, so the natural spread of qber_base across links stays in as noise.
        """
        q, r = state.qber_obs, state.R_obs
        ratio = r / (self.topo.R + EPS)
        if self._ema_qber is None:
            self._ema_qber = q.copy()
            self._ema_skr = ratio.copy()
            self._dev_qber = np.full_like(q, 0.05) * q
            self._dev_skr = np.full_like(ratio, 0.05)
            return np.zeros(self.n_nodes, dtype=float)

        # Both branches are standardised, so b1 and b2 combine like quantities.
        # Deviations are expressed in units of the link's own measurement spread
        # (temporal) or of the cohort spread (peer), never as a bare relative
        # change: link rates differ by an order of magnitude across lengths and
        # the base QBER spans a factor of two, so a raw relative deviation would
        # weight the long links and the noisy links differently.
        z_q_t = (q - self._ema_qber) / (MAD_SCALE * self._dev_qber + EPS)
        z_r_t = (self._ema_skr - ratio) / (MAD_SCALE * self._dev_skr + EPS)

        z_q_p = _peer_z(q)
        z_r_p = -_peer_z(ratio)          # a *drop* in delivered rate is suspicious

        z_q = np.maximum(z_q_t, z_q_p)
        z_r = np.maximum(z_r_t, z_r_p)
        val = self.cfg.b1 * z_q + self.cfg.b2 * z_r

        # update after taking the deviation, so a step change is not instantly
        # absorbed into its own baseline
        a = self._alpha_ema
        self._dev_qber += a * (np.abs(q - self._ema_qber) - self._dev_qber)
        self._dev_skr += a * (np.abs(ratio - self._ema_skr) - self._dev_skr)
        self._ema_qber += a * (q - self._ema_qber)
        self._ema_skr += a * (ratio - self._ema_skr)

        return np.maximum.reduceat(val[self.flat_edges], self.offsets)


def collect(state: NetworkState, topo: Topology, cfg: DetectorConfig,
            t: float) -> np.ndarray:
    """Stateless convenience wrapper, for one off inspection in a notebook.

    The real path is :class:`TelemetryCollector`, which has to carry the window
    and the EMA across samples.
    """
    return TelemetryCollector(topo, cfg).collect(state, topo, t)
