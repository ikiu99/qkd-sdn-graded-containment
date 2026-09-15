"""Reproducibility: same seed -> same numbers -> byte identical parquet.

Also guards the seed isolation contract of section 5: the demand stream and the
attack stream must be independent, otherwise changing lambda silently moves the
compromised node selection and every cross scenario comparison is contaminated.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import make_cfg
from sim.config import config_from_dict, make_run_id, to_dict
from sim.interfaces import ATTACKS, NullAttack
from sim.runner import run, simulate

VOLATILE = {"wall_time_s", "summary_path"}


def _stable(summary: dict) -> dict:
    return {k: v for k, v in summary.items() if k not in VOLATILE}


def test_same_seed_same_output(tmp_path):
    cfg = make_cfg(tmp_path)
    a = _stable(run(cfg, write=False, asserts=False))
    b = _stable(run(cfg, write=False, asserts=False))
    assert a == b
    assert a["n_offered"] > 0


def test_byte_identical_parquet(tmp_path):
    """Definition of done 3: rerunning with the same seed must reproduce the
    output file byte for byte."""
    cfg = make_cfg(tmp_path)
    run(cfg, write=True, asserts=False)
    first = {suffix: (tmp_path / f"{cfg.run_id}{suffix}.parquet").read_bytes()
             for suffix in ("", "_events")}
    run(cfg, write=True, asserts=False)
    for suffix, blob in first.items():
        again = (tmp_path / f"{cfg.run_id}{suffix}.parquet").read_bytes()
        assert hashlib.sha256(blob).hexdigest() == hashlib.sha256(again).hexdigest(), (
            f"{suffix or 'summary'} parquet is not reproducible")
        assert len(blob) > 0


def test_different_demand_seed_changes_results(tmp_path):
    a = run(make_cfg(tmp_path, seed_demand=2), write=False, asserts=False)
    b = run(make_cfg(tmp_path, seed_demand=99), write=False, asserts=False)
    assert a["n_offered"] != b["n_offered"] or a["KPD"] != b["KPD"]


class ProbeAttack(NullAttack):
    """Records what it draws from the attack stream, so the test can prove the
    demand stream cannot reach into it."""
    draws: list[tuple[int, list[float]]] = []

    def __init__(self, n_nodes=0, rng=None, **kw):
        super().__init__(n_nodes=n_nodes)
        ProbeAttack.draws.append((len(ProbeAttack.draws), rng.random(8).tolist()))


def test_seed_isolation(tmp_path):
    """Changing seed_demand must not move a single number drawn from rng_attack."""
    ATTACKS["probe"] = ProbeAttack
    try:
        ProbeAttack.draws.clear()
        for tag, seed_demand in enumerate((2, 99, 12345)):
            cfg = make_cfg(tmp_path, seed_demand=seed_demand,
                           horizon=600, warmup=60,
                           attack={"type": "probe", "enabled": True})
            run(cfg, write=False, asserts=False)
        assert len(ProbeAttack.draws) == 3
        first = ProbeAttack.draws[0][1]
        for _, drawn in ProbeAttack.draws[1:]:
            assert drawn == first
    finally:
        ATTACKS.pop("probe", None)
        ProbeAttack.draws.clear()


def test_run_id_is_order_independent():
    a = config_from_dict({"topology": {"name": "nsfnet"}, "demand": {"lam": 0.1}})
    b = config_from_dict({"demand": {"lam": 0.1}, "topology": {"name": "nsfnet"}})
    assert a.run_id == b.run_id == make_run_id(a)
    c = config_from_dict({"demand": {"lam": 0.2}, "topology": {"name": "nsfnet"}})
    assert c.run_id != a.run_id


def test_run_id_ignores_output_dir():
    a = config_from_dict({"output_dir": "results/raw"})
    b = config_from_dict({"output_dir": "somewhere/else"})
    assert a.run_id == b.run_id


def test_state_is_identical_not_just_summary(tmp_path):
    cfg = make_cfg(tmp_path)
    ra = simulate(cfg, write=False, asserts=False)
    rb = simulate(cfg, write=False, asserts=False)
    assert np.array_equal(ra.state.buffers, rb.state.buffers)
    assert np.array_equal(ra.state.actual_consumed, rb.state.actual_consumed)
    assert sorted(ra.state.sessions) == sorted(rb.state.sessions)
