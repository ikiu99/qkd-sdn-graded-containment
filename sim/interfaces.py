"""Extension points for phases 3-5.

This module is the whole extensibility story: phases 3, 4 and 5 add new
implementations of these ABCs and register them in the factories below, and
``runner.py`` never changes. Phase 2 ships only the Null implementations, which
together are exactly baseline B0.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import is_dataclass

import numpy as np

from .state import NetworkState
from .topology import Topology


# attack (phase 3)
class Attack(ABC):
    @abstractmethod
    def compromised_nodes(self, t: float) -> np.ndarray:
        """Boolean mask over nodes, True where the node is compromised at t."""

    @abstractmethod
    def apply(self, state: NetworkState, t: float) -> None:
        """Mutate the state to reflect the attacker behaviour at time t.

        Called every step. This is where an attacker touches the *true* state:
        draining buffers, raising QBER, lowering the key rate.
        """

    def observe(self, state: NetworkState, t: float) -> None:
        """Tamper with the reported observables, after the honest noise model ran.

        Called once per detector sample, never per step. Default: no tampering.
        Split from :meth:`apply` because lying about a measurement is not the
        same act as changing the thing being measured, and only profile L does it.
        """
        return None


class NullAttack(Attack):
    """No compromised node, no effect."""

    def __init__(self, n_nodes: int = 0, **_: object):
        self._mask = np.zeros(int(n_nodes), dtype=bool)

    def compromised_nodes(self, t: float) -> np.ndarray:
        return self._mask

    def apply(self, state: NetworkState, t: float) -> None:
        return None


# detector (phase 4)
class Detector(ABC):
    @abstractmethod
    def update(self, state: NetworkState, topo: Topology, t: float) -> np.ndarray:
        """Raw per node score S in [0,1].

        A detector may read only what the SDN controller can see: the ``*_obs``
        observables, ``buffer_reported``, ``ledger_consumed`` and the topology.
        Reading ``buffers``, any ``*_true`` field or ``compromised`` is a bug;
        ``test_detector_isolation`` enforces it.
        """

    def initial_S_bar(self, n_nodes: int) -> np.ndarray:
        """Starting value of the smoothed score.

        Phase 4, section 4.3: start from the prior, not from zero, otherwise the
        measured detection delay is inflated by the climb out of zero.
        """
        return np.zeros(int(n_nodes), dtype=float)


class NullDetector(Detector):
    """Always zero."""

    def __init__(self, n_nodes: int = 0, **_: object):
        self._zeros = np.zeros(int(n_nodes), dtype=float)

    def update(self, state: NetworkState, topo: Topology, t: float) -> np.ndarray:
        return self._zeros


# policy (phase 5)
class Policy(ABC):
    @abstractmethod
    def apply(self, S_bar: np.ndarray, t: float = 0.0, state=None):
        """Map the smoothed score onto the response levers.

        Returns a ``policy.PolicyDecision``. The arguments are the whole
        interface: a policy sees the score and the clock and nothing else.
        Phase 4 fixed an information boundary for the detector, and the policy
        is on the same side of it - B4's ground truth is injected explicitly at
        construction so the one implementation that cheats is visible.
        Enforced by ``test_policy_isolation``.

        Implementations must return FRESH arrays. Handing back internal ones
        makes ``state.rho`` an alias of policy-owned memory and defeats the
        staleness check in the route table.
        """


class NullPolicy(Policy):
    """(ones, ones, zeros_bool) - baseline B0: plain shortest path, no throttling."""

    def __init__(self, n_nodes: int = 0, **_: object):
        self.n = int(n_nodes)

    def apply(self, S_bar: np.ndarray, t: float = 0.0, state=None):
        from .policy import PolicyDecision
        return PolicyDecision(
            node_weight=np.ones(self.n, dtype=float),
            rho=np.ones(self.n, dtype=float),
            isolated=np.zeros(self.n, dtype=bool),
        )


# factories - phases 3-5 register their classes here, nothing else moves
ATTACKS: dict[str, type[Attack]] = {"none": NullAttack, "null": NullAttack}
DETECTORS: dict[str, type[Detector]] = {"none": NullDetector, "null": NullDetector}
POLICIES: dict[str, type[Policy]] = {"none": NullPolicy, "null": NullPolicy}

# Deferred registrations. attack.py and detector.py import this module, so they
# cannot be imported at module scope here; they are pulled in on first use and
# cached into the registry above. Adding a phase 5 policy follows the same shape.
_LAZY: dict[tuple[str, str], tuple[str, str]] = {
    ("attack", "compromise"): (".attack", "CompromiseAttack"),
    ("detector", "suspicion"): (".detector", "SuspicionDetector"),
    ("policy", "B0"): (".policy", "NoDefencePolicy"),
    ("policy", "B1"): (".policy", "BinaryPolicy"),
    ("policy", "B2"): (".policy", "MultipathPolicy"),
    ("policy", "B3"): (".policy", "GradedPolicy"),
    ("policy", "B4"): (".policy", "OraclePolicy"),
    ("policy", "BT"): (".policy", "BlindThrottlePolicy"),
    ("policy", "B5"): (".policy", "TrustDeratePolicy"),
    ("policy", "B6"): (".policy", "NonOverlappingMultipathPolicy"),
    ("policy", "B7"): (".policy", "KeyAwareRoutingPolicy"),
    ("policy", "B8"): (".policy", "HybridPolicy"),
}


def _resolve(registry: dict, name: str, kind: str):
    if name in registry:
        return registry[name]
    target = _LAZY.get((kind, name))
    if target is not None:
        import importlib
        module = importlib.import_module(target[0], package=__package__)
        registry[name] = getattr(module, target[1])
        return registry[name]
    known = sorted(set(registry) | {n for k, n in _LAZY if k == kind})
    raise ValueError(f"unknown {kind} type {name!r}; available: {known}")


def _build(registry: dict, spec, kind: str, **kwargs):
    """Instantiate from a typed config dataclass or a plain ``{"type": ...}`` dict.

    A section carrying ``enabled=False`` always yields the Null implementation,
    so switching a phase off never depends on also blanking its ``type``.
    """
    if is_dataclass(spec) and not isinstance(spec, type):
        if not getattr(spec, "enabled", True):
            return registry["none"](**kwargs)
        return _resolve(registry, str(getattr(spec, "type", "none")), kind)(
            cfg=spec, **kwargs)

    spec = dict(spec or {})
    name = str(spec.pop("type", "none"))
    return _resolve(registry, name, kind)(**kwargs, **spec)


def build_attack(spec, n_nodes: int = 0, **kwargs) -> Attack:
    return _build(ATTACKS, spec, "attack", n_nodes=n_nodes, **kwargs)


def build_detector(spec, n_nodes: int = 0, **kwargs) -> Detector:
    return _build(DETECTORS, spec, "detector", n_nodes=n_nodes, **kwargs)


def build_policy(spec, n_nodes: int = 0, **kwargs) -> Policy:
    return _build(POLICIES, spec, "policy", n_nodes=n_nodes, **kwargs)


def ewma(S: np.ndarray, S_bar: np.ndarray, alpha: float) -> np.ndarray:
    """S_bar(t) = alpha*S(t) + (1-alpha)*S_bar(t-dt), phase 1 section 8.2."""
    return alpha * S + (1.0 - alpha) * S_bar
