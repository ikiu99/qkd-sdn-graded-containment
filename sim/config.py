"""Configuration dataclasses, YAML loading and deterministic run identifiers.

"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from types import UnionType
from typing import Any, Union, get_origin, get_type_hints

import yaml

KM_MODES = ("OTP", "AES")
ADMISSION_MODES = ("optimistic", "reserve_full")
ATTACK_PROFILES = ("P", "G", "L", "E")
POLICY_TYPES = ("B0", "B1", "B2", "B3", "B4", "BT", "B8",
                # reimplementations of published methods, see sim/policy.py
                "B5", "B6", "B7",
                "none", "null")
RHO_MODES = ("product", "min")
SELECTIONS = ("random", "top_keyflow")
FEATURE_NAMES = ("x1", "x2", "x3")
X1_MODES = ("signed", "abs", "cusum")
ROUTE_MODES = ("exp", "logrisk")
HYBRID_AGGS = ("product", "max", "mean")
X2_MODES = ("signed", "abs")


@dataclass(frozen=True)
class TopologyConfig:
    name: str = "nsfnet"                 # "nsfnet" | "usnet" | "net50"
    path: str = "data/topologies/nsfnet.json"
    R_max: float = 20000.0               # bit/s
    L_0: float = 21.7147                 # km, = 10/(alpha ln10) at alpha=0.2 dB/km
    B_max: float = 10e6                  # bit


@dataclass(frozen=True)
class DemandConfig:
    lam: float = 0.05                    # Poisson arrival rate (demands per second)
    km_mode: str = "OTP"                 # "OTP" | "AES"
    # OTP
    rate_min: float = 1e3                # bit/s session data rate
    rate_max: float = 1e5
    # AES
    T_rk: float = 60.0                   # s re-keying period
    key_size: int = 256                  # bit
    # shared
    T_s_min: float = 60.0                # s session duration
    T_s_max: float = 600.0
    admission: str = "optimistic"        # "optimistic" | "reserve_full"
    T_adm: float = 60.0                  # s look-ahead horizon for optimistic admission


@dataclass(frozen=True)
class RoutingConfig:
    K: int = 4                           # cached paths per pair
    T_route: float = 300.0               # s recomputation period
    disjoint_K: int = 3                  # for B2 in later phases


@dataclass(frozen=True)
class AttackConfig:
    """Phase 3, section 3.4 of phase3-4-build-spec."""
    type: str = "compromise"        # registry key in interfaces.ATTACKS
    enabled: bool = False
    profile: str = "P"              # "P" passive | "G" greedy | "L" liar | "E" eavesdropper
    f: float = 0.10                 # fraction of compromised nodes
    selection: str = "random"       # "random" | "top_keyflow"
    t_c: float | None = None        # None -> 0.25 * horizon
    gamma: float = 0.30             # G intensity
    delta: float = 0.50             # L intensity
    q: float = 0.03                 # E: QBER increase, absolute
    s: float = 0.25                 # E: relative SKR drop
    collude: bool = False           # L: paired ends report one agreed number


@dataclass(frozen=True)
class NoiseConfig:
    """Observation noise, phase 3 section 3.3.

    A first class component, not a refinement: with a noiseless observer every
    healthy node has K_obs == K_exp exactly, every threshold gives AUC = 1.0 and
    the whole of phase 4 becomes meaningless.
    """
    sigma_qber_rel: float = 0.10    # N1, relative QBER measurement noise
    sigma_skr_rel: float = 0.05     # N2, key rate jitter
    sigma_report_rel: float = 0.02  # N3, buffer report noise, per edge side
    unmanaged_fraction: float = 0.10  # N4, background traffic invisible to the ledger
    scale: float = 1.0              # scales all four together (the noise sweep axis)
    qber_base_min: float = 0.015    # per edge base QBER, drawn from rng_topology
    qber_base_max: float = 0.030


@dataclass(frozen=True)
class DetectorConfig:
    """Phase 4, section 4.3.

    Three distinct periods (section 4.1): sampling every T_sample, features
    accumulated over the overlapping window W_feat, robust baseline over W_base.
    W_base / T_sample = 120 samples, which is a statistically sound base for
    median and MAD; the 6 samples of a naive 300 s sampling are not.
    """
    type: str = "suspicion"         # registry key in interfaces.DETECTORS
    enabled: bool = False
    T_sample: float = 30.0          # s, how often x1,x2,x3 are computed
    W_feat: float = 300.0           # s, feature accumulation window (overlapping)
    W_base: float = 3600.0          # s, robust baseline window
    W_ema: float = 1800.0           # s, per edge EMA horizon behind x3
    z_max: float = 5.0
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    lam_s: float = 6.0              # score slope
    theta_0: float = 0.35           # score midpoint
    alpha: float = 0.05             # EWMA smoothing
    x1_mode: str = "signed"         # "signed" | "abs" | "cusum", telemetry._x1
    # x1_mode="cusum" only: samples used for the frozen reference, and the
    # per-sample slack that holds an honest node at zero. The reference
    # window sits inside the warm-up, i.e. before t_c - an assumption the
    # paper states rather than hides.
    x1_cusum_ref: int = 12
    x1_cusum_k: float = 0.5
    # Forgetting factor. 1.0 is the textbook unbounded CUSUM, which the
    # rolling baseline downstream cannot cope with; below 1 the statistic
    # is bounded and stationary under the null while still integrating
    # over about 1/(1-decay) samples.
    x1_cusum_decay: float = 0.9
    x1_ema: float = 0.0             # EMA on the x1 residual before
                                    # standardisation; 0 disables it
    x2_mode: str = "signed"         # "signed" | "abs" - see telemetry._x2
    b1: float = 0.5                 # x3: QBER term
    b2: float = 0.5                 # x3: SKR term
    # Operator calibrated constants used to standardise the x1 residual. These
    # are the controller's *assumptions* about its own measurement quality, not
    # the simulator's true noise settings: keeping them separate is what allows
    # a later sweep over "what if the operator's assumption is wrong".
    report_sigma: float = 0.02      # assumed relative buffer report uncertainty
    unmanaged_assumed: float = 0.10 # assumed share of consumption off ledger
    features: tuple[str, ...] = ("x1", "x2", "x3")   # active subset, for ablation
    use_prior: bool = True
    ablation: bool = True           # also track the other 6 feature subsets
    keep_evidence: bool = False     # calibration pilot only: retain e_k samples
    # prior, phase 1 section 7
    c0: float = -2.0
    c1: float = 1.0                 # Exposure_i
    c2: float = 1.5                 # Phi_i, baseline key flow
    c3: float = 0.5                 # Age_i
    phi_path: str = ""              # "" -> config/calibration_{topology}_{km_mode}.yaml


@dataclass(frozen=True)
class PolicyConfig:
    """Policy configuration: the graded response and the baselines it is measured against.

    Five policies share one config so that a sweep can move between them on a
    single axis and every run in a scenario hashes identically outside
    ``policy.*`` - which is what makes the DRR baseline join exact.

        B0  no defence, the reference
        B1  binary: isolate at a fixed threshold on the same S_bar
        B2  static multipath: XOR key shares over m node-disjoint paths,
            no detection at all
        B3  graded continuous response (the proposal)
        B4  oracle: perfect knowledge of V_c, the achievable upper bound
    """
    type: str = "B0"
    enabled: bool = True            # B0 is a real policy, not an "off" state
    # --- B1 ---
    tau: float = 0.5                # binary isolation threshold
    # --- B2 ---
    m_paths: int = 2                # XOR shares / node-disjoint legs
    # --- B3 ---
    kappa: float = 3.0              # routing weight penalty, w = exp(kappa*S_bar)
    S_iso: float = 0.5              # rho hits 0 exactly here, and isolation starts
    rho_start: float = 0.0          # score at which throttling begins
    rho_blind: float = 1.0          # BT only: uniform quota, no detection
    rho_mode: str = "product"       # "product" | "min" over the intermediate relays
    # --- shared response shaping ---
    hysteresis: float = 0.1         # release at (threshold - hysteresis)
    dwell: float = 600.0            # s, minimum time in a state before it may flip
    rho_min: float = 0.05           # floor applied when isolation would partition
    tear_down_on_isolate: bool = False
    # B5, Luo & Li 2025: sharpness of the trust derating exp(-beta * d^2).
    # Calibrated on the full horizon at 20 seeds, not guessed, and reported at
    # the baseline's OWN best point rather than at ours: beta = 50 reaches
    # DRR 0.995 on the liar for dRR +0.365, while beta = 200 buys nothing more
    # on L (0.995) and costs +0.478. Above 1000 the policy stops detecting and
    # starts throttling everything - at 5000 it reaches DRR 0.867 on the
    # UNDETECTABLE profile P, which is volume, not detection, and matches the
    # blind control. Swept in config/sweep_baselines.yaml.
    beta_trust: float = 50.0
    # B7, Bi et al. 2023: weight on remaining key vs link availability,
    # their alpha with beta = 1 - alpha
    alpha_key: float = 0.5
    # How the routing weight turns a risk score into a path cost.
    #   "exp"     w_i = exp(kappa * S_bar_i), the phase 1 formula, measured inert
    #   "logrisk" w_i = 1 + kappa * (-log(1 - S_bar_i))
    # The second is the one the damage model implies. Routing cost sums
    # 0.5*(w_u + w_v) over the edges of a path, so each interior node
    # contributes w_i exactly once and the path cost becomes
    # hops + kappa * sum(-log(1 - S_bar)) - i.e. hop count plus the cumulative
    # log-probability that the path contains a compromised relay. Minimising it
    # maximises the chance the whole path is clean, which is what exposure
    # actually depends on; exp(kappa*S) merely makes a suspect node a constant
    # factor dearer and is dominated by the quota lever acting on the same node.
    route_mode: str = "exp"
    # B8: a session gets m_paths XOR legs only when the cumulative probability
    # that its cheapest single path carries a compromised relay exceeds this.
    # 1.0 disables the trigger (never redundant), 0.0 makes B8 into B2.
    hybrid_tau: float = 0.35  # also cut sessions already in flight
    # how the per-node risks along a path combine into one trigger value.
    # "product" is 1 - prod(1 - S_i), "max" the worst single relay, "mean"
    # the product form normalised by the interior length.
    hybrid_agg: str = "product"


@dataclass(frozen=True)
class SimConfig:
    run_id: str = ""
    horizon: int = 86400                 # s
    dt: float = 1.0                      # s
    warmup: int = 1800                   # s - nothing is recorded into metrics here
    telemetry_period: float = 300.0      # s - node level logging period
    seed_topology: int = 1
    seed_demand: int = 2
    seed_attack: int = 3
    seed_policy: int = 4
    seed_noise: int = 5                  # observation noise + background traffic
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    output_dir: str = "results/raw"

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon / self.dt))

    @property
    def t_compromise(self) -> float:
        """Resolved attack start time."""
        return (0.25 * self.horizon if self.attack.t_c is None
                else float(self.attack.t_c))


_SECTIONS = {
    "topology": TopologyConfig,
    "demand": DemandConfig,
    "routing": RoutingConfig,
    "attack": AttackConfig,
    "noise": NoiseConfig,
    "detector": DetectorConfig,
    "policy": PolicyConfig,
}


# loading
def _deep_merge(base: dict, extra: dict) -> dict:
    """Recursive dict merge, extra wins. base is not mutated."""
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_dotted(overrides: dict) -> dict:
    """Turn {"demand.lam": 0.1} into {"demand": {"lam": 0.1}}."""
    out: dict = {}
    for key, value in overrides.items():
        parts = str(key).split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value
    return out


def _coerce(cls, raw: dict) -> dict:
    """Cast YAML scalars to the annotated field types.

    PyYAML follows YAML 1.1, where ``10.0e6`` (no sign in the exponent) parses as
    a string, so a plain ``cls(**raw)`` would silently carry strings into the
    arithmetic.
    """
    hints = get_type_hints(cls)
    out = {}
    for key, value in raw.items():
        target = hints.get(key)
        origin = get_origin(target)
        if origin is tuple:
            # YAML gives a list; the dataclasses are frozen, so store a tuple
            out[key] = tuple(value)
        elif origin in (Union, UnionType):
            # only ``float | None`` occurs so far (attack.t_c)
            out[key] = None if value is None else float(value)
        elif target is float and not isinstance(value, bool):
            out[key] = float(value)
        elif target is int and not isinstance(value, bool):
            out[key] = int(value)
        elif target is str:
            out[key] = str(value)
        elif target is bool:
            out[key] = bool(value)
        else:
            out[key] = value
    return out


def config_from_dict(raw: dict) -> SimConfig:
    raw = dict(raw or {})
    kwargs: dict[str, Any] = {}
    for f in fields(SimConfig):
        if f.name in _SECTIONS:
            section_cls = _SECTIONS[f.name]
            section_raw = raw.get(f.name, {}) or {}
            unknown = set(section_raw) - {sf.name for sf in fields(section_cls)}
            if unknown:
                raise ValueError(f"unknown keys in section {f.name}: {sorted(unknown)}")
            kwargs[f.name] = section_cls(**_coerce(section_cls, section_raw))
        elif f.name in raw:
            kwargs[f.name] = raw[f.name]
    kwargs.update(_coerce(SimConfig, {k: v for k, v in kwargs.items()
                                      if k not in _SECTIONS}))
    unknown_top = set(raw) - {f.name for f in fields(SimConfig)}
    if unknown_top:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown_top)}")
    cfg = SimConfig(**kwargs)
    validate(cfg)
    if not cfg.run_id:
        cfg = dataclasses.replace(cfg, run_id=make_run_id(cfg))
    return cfg


def load_config(path: str, overrides: dict | None = None) -> SimConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = _deep_merge(raw, _expand_dotted(overrides))
    return config_from_dict(raw)


def validate(cfg: SimConfig) -> None:
    if cfg.demand.km_mode not in KM_MODES:
        raise ValueError(f"km_mode must be one of {KM_MODES}")
    if cfg.demand.admission not in ADMISSION_MODES:
        raise ValueError(f"admission must be one of {ADMISSION_MODES}")
    if cfg.dt <= 0:
        raise ValueError("dt must be > 0")
    if cfg.horizon <= 0:
        raise ValueError("horizon must be > 0")
    if cfg.warmup < 0 or cfg.warmup >= cfg.horizon:
        raise ValueError("warmup must satisfy 0 <= warmup < horizon")
    if cfg.telemetry_period <= 0:
        raise ValueError("telemetry_period must be > 0")
    if cfg.routing.K < 1:
        raise ValueError("routing.K must be >= 1")
    if cfg.demand.lam < 0:
        raise ValueError("demand.lam must be >= 0")
    if cfg.demand.T_s_min <= 0 or cfg.demand.T_s_max < cfg.demand.T_s_min:
        raise ValueError("require 0 < T_s_min <= T_s_max")
    if cfg.demand.km_mode == "OTP" and (
        cfg.demand.rate_min <= 0 or cfg.demand.rate_max < cfg.demand.rate_min
    ):
        raise ValueError("require 0 < rate_min <= rate_max")
    if cfg.demand.km_mode == "AES" and (cfg.demand.T_rk <= 0 or cfg.demand.key_size <= 0):
        raise ValueError("require T_rk > 0 and key_size > 0")
    if cfg.topology.L_0 <= 0 or cfg.topology.R_max <= 0 or cfg.topology.B_max <= 0:
        raise ValueError("topology R_max, L_0, B_max must be > 0")

    # --- phase 3: attack -------------------------------------------------- #
    a = cfg.attack
    if a.profile not in ATTACK_PROFILES:
        raise ValueError(f"attack.profile must be one of {ATTACK_PROFILES}")
    if a.selection not in SELECTIONS:
        raise ValueError(f"attack.selection must be one of {SELECTIONS}")
    if not 0.0 <= a.f <= 1.0:
        raise ValueError("attack.f must lie in [0,1]")
    if a.enabled and cfg.t_compromise >= cfg.horizon:
        raise ValueError("attack t_c must fall inside the horizon")
    if a.enabled and cfg.t_compromise < cfg.warmup:
        raise ValueError("attack t_c must not fall inside the metric warm-up window")

    # --- phase 3: noise --------------------------------------------------- #
    n = cfg.noise
    if min(n.sigma_qber_rel, n.sigma_skr_rel, n.sigma_report_rel,
           n.unmanaged_fraction, n.scale) < 0:
        raise ValueError("noise parameters must be >= 0")
    if not 0 < n.qber_base_min <= n.qber_base_max < 0.5:
        raise ValueError("require 0 < qber_base_min <= qber_base_max < 0.5")

    # --- phase 4: detector ------------------------------------------------ #
    d = cfg.detector
    if d.T_sample <= 0 or d.W_feat < d.T_sample or d.W_base < d.W_feat:
        raise ValueError("require 0 < T_sample <= W_feat <= W_base")
    for name, period in (("T_sample", d.T_sample), ("W_feat", d.W_feat),
                         ("W_base", d.W_base)):
        if abs(period / cfg.dt - round(period / cfg.dt)) > 1e-9:
            raise ValueError(f"detector.{name} must be a whole number of dt steps")
    if abs(d.W_feat / d.T_sample - round(d.W_feat / d.T_sample)) > 1e-9:
        raise ValueError("detector.W_feat must be a whole multiple of T_sample")
    if not 0.0 < d.alpha <= 1.0:
        raise ValueError("detector.alpha must lie in (0,1]")
    if len(d.weights) != 3 or min(d.weights) < 0 or sum(d.weights) <= 0:
        raise ValueError("detector.weights must be three non negative numbers")
    if not 0.0 <= d.x1_ema < 1.0:
        raise ValueError("detector.x1_ema must be in [0, 1)")
    if d.x1_mode not in X1_MODES:
        raise ValueError(f"detector.x1_mode must be one of {X1_MODES}")
    if d.x2_mode not in X2_MODES:
        raise ValueError(f"detector.x2_mode must be one of {X2_MODES}")
    bad = set(d.features) - set(FEATURE_NAMES)
    if bad or not d.features:
        raise ValueError(f"detector.features must be a non empty subset of {FEATURE_NAMES}")
    # --- phase 5: policy ---------------------------------------------------- #
    p = cfg.policy
    if p.route_mode not in ROUTE_MODES:
        raise ValueError(f"policy.route_mode must be one of {ROUTE_MODES}")
    if p.hybrid_agg not in HYBRID_AGGS:
        raise ValueError(f"policy.hybrid_agg must be one of {HYBRID_AGGS}")
    if not 0.0 <= p.hybrid_tau <= 1.0:
        raise ValueError("policy.hybrid_tau must be in [0, 1]")
    if not 0.0 <= p.alpha_key <= 1.0:
        raise ValueError("policy.alpha_key must be in [0, 1]")
    if p.beta_trust < 0.0:
        raise ValueError("policy.beta_trust must be >= 0")
    if p.type not in POLICY_TYPES:
        raise ValueError(f"policy.type must be one of {POLICY_TYPES}")
    if p.rho_mode not in RHO_MODES:
        raise ValueError(f"policy.rho_mode must be one of {RHO_MODES}")
    if not 0.0 < p.tau <= 1.0 or not 0.0 < p.S_iso <= 1.0:
        raise ValueError("policy.tau and policy.S_iso must lie in (0,1]")
    if not 0.0 <= p.hysteresis < min(p.tau, p.S_iso):
        raise ValueError("policy.hysteresis must be >= 0 and below the threshold")
    if not 0.0 < p.rho_blind <= 1.0:
        raise ValueError("policy.rho_blind must lie in (0,1]")
    if not 0.0 <= p.rho_start < p.S_iso:
        raise ValueError("policy.rho_start must satisfy 0 <= rho_start < S_iso")
    if p.dwell < 0 or p.kappa < 0 or not 0.0 <= p.rho_min <= 1.0:
        raise ValueError("require dwell >= 0, kappa >= 0, rho_min in [0,1]")
    if p.m_paths < 1:
        raise ValueError("policy.m_paths must be >= 1")
    if p.type in ("B1", "B3") and not cfg.detector.enabled:
        raise ValueError(f"policy {p.type} needs a detector; set detector.enabled")

    if d.enabled:
        n_base = int(round(d.W_base / d.T_sample))
        if n_base < 30:
            raise ValueError(
                f"W_base/T_sample = {n_base} samples is too few for a robust "
                "median/MAD baseline; phase 4 section 4.1 calls for ~120")
        if cfg.warmup + d.W_base >= cfg.horizon:
            raise ValueError("no time left after warmup + W_base; raise the horizon")


# serialisation / identity
def to_dict(cfg: SimConfig) -> dict:
    """Plain nested dict of the whole config."""
    return asdict(cfg)


def flatten(obj: Any, prefix: str = "") -> dict:
    """Flatten nested dataclasses/dicts into a.b.c keys (for the parquet summary row)."""
    out: dict = {}
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if isinstance(obj, dict):
        for k in sorted(obj):
            out.update(flatten(obj[k], f"{prefix}{k}."))
        return out
    key = prefix[:-1] if prefix.endswith(".") else prefix
    if isinstance(obj, (list, tuple)):
        out[key] = json.dumps(list(obj), sort_keys=True)
    else:
        out[key] = obj
    return out


def _hash_config(cfg: SimConfig, drop: tuple[str, ...]) -> str:
    payload = to_dict(cfg)
    for key in drop:
        payload.pop(key, None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def scenario_id(cfg: SimConfig) -> str:
    """Identity of everything except the policy - the DRR baseline join key.

    DRR compares a policy run against the B0 run of the *same* scenario. Joining
    on "every column except policy.*" over a flattened parquet is fragile: adding
    one config field later silently changes the key set and invalidates every
    previously computed DRR. One hash column is stable under schema growth and
    is checkable (each scenario_id group must hold exactly one B0 row per seed).
    """
    return _hash_config(cfg, ("policy", "output_dir", "run_id"))


def base_scenario_id(cfg: SimConfig) -> str:
    """Identity of everything except policy AND attack - the PSI join key.

    PSI is path stretch relative to B0 *without* an attack, so its baseline has
    to be matched across attack settings too.
    """
    return _hash_config(cfg, ("policy", "attack", "output_dir", "run_id"))


def make_run_id(cfg: SimConfig) -> str:
    """Deterministic hash over every config field except output_dir and run_id.

    Stable across processes and key orderings: sha1 over canonical sorted JSON,
    never the (salted) builtin hash().
    """
    payload = to_dict(cfg)
    payload.pop("output_dir", None)
    payload.pop("run_id", None)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    return f"{cfg.topology.name}_{cfg.demand.km_mode}_{digest}"
