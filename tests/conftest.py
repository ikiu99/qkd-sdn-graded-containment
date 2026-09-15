"""Shared fixtures. Every test runs with the runtime invariants enabled."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.config import config_from_dict  # noqa: E402

T1 = os.path.join(ROOT, "data", "topologies", "nsfnet.json")
T3 = os.path.join(ROOT, "data", "topologies", "net50.json")


def make_cfg(tmp_path=None, **over) -> "object":
    """Small, fast config: T1, 1 h horizon, 5 min warm-up."""
    raw = {
        "horizon": 3600,
        "dt": 1.0,
        "warmup": 300,
        "telemetry_period": 300.0,
        "seed_topology": 1,
        "seed_demand": 2,
        "seed_attack": 3,
        "seed_policy": 4,
        "topology": {"name": "nsfnet", "path": T1},
        "demand": {"lam": 0.2, "km_mode": "OTP",
                   "rate_min": 10.0, "rate_max": 1000.0},
        "routing": {"K": 4},
        "output_dir": str(tmp_path) if tmp_path is not None else "results/raw",
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return config_from_dict(raw)


@pytest.fixture
def cfg(tmp_path):
    return make_cfg(tmp_path)


@pytest.fixture
def topo():
    from sim.config import TopologyConfig
    from sim.topology import load_topology
    return load_topology(TopologyConfig(name="nsfnet", path=T1))
