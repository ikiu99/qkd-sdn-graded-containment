"""Parquet output.

Two tables per run:
  summary  - one row: every (flattened) config field plus every metric, the
             damage accounting of phase 3 and the detector report of phase 4
  events   - node telemetry sampled every telemetry_period, never per step
             (86400 steps x 50 nodes would be 4.3M rows per run), carrying the
             raw features x1..x3, the evidence e1..e3, the score, and the
             ground truth columns the offline evaluator needs

The ground truth columns (compromised, t_compromise) exist for the *evaluator*,
which is allowed to know the answer. The detector is not: see telemetry.py.

Files are written with a fixed schema and column order so that two runs with the
same seed produce byte identical parquet files.
"""
from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

from .config import SimConfig, base_scenario_id, flatten, scenario_id

COMPRESSION = "snappy"

# Wall clock timings are the only non deterministic numbers a run produces;
# they stay in the returned summary dict (and in the sweep log) but never
# enter the parquet, so two runs with the same seed are byte identical.
VOLATILE_KEYS = ("wall_time_s", "summary_path")

_F64 = pa.float64()
EVENT_SCHEMA = pa.schema(
    [("t", _F64), ("node_id", pa.int32())]
    + [(name, _F64) for name in ("x1", "x2", "x3", "e1", "e2", "e3",
                                 "S", "S_bar", "prior", "rho")]
    + [("isolated", pa.bool_())]
    + [(name, _F64) for name in ("node_key_flow", "adjacent_buffer_mean",
                                 "qber_obs_mean")]
    + [("exposed", pa.bool_()), ("compromised", pa.bool_()),
       ("t_compromise", _F64)]
)


def summary_row(cfg: SimConfig, metrics_summary: dict) -> dict:
    # The two baseline join keys of phase 6. DRR compares a policy run to the B0
    # run of the same scenario, PSI to the B0-without-attack run of the same base
    # scenario; carrying them as single columns keeps those joins exact and
    # stable when the config schema grows.
    row = {"run_id": cfg.run_id,
           "scenario_id": scenario_id(cfg),
           "base_scenario_id": base_scenario_id(cfg)}
    row.update(flatten(cfg))
    row.pop("run_id.", None)
    row.update({k: v for k, v in metrics_summary.items()
                if k not in VOLATILE_KEYS})
    return row


def _summary_table(row: dict) -> pa.Table:
    ordered = {k: [row[k]] for k in sorted(row)}
    return pa.table(ordered)


def _events_table(tel: dict[str, list]) -> pa.Table:
    cols = {name: tel.get(name, []) for name in EVENT_SCHEMA.names}
    arrays = [pa.array(cols[f.name], type=f.type) for f in EVENT_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=EVENT_SCHEMA)


def write_run(cfg: SimConfig, metrics_summary: dict,
              telemetry: dict[str, list] | None = None) -> dict[str, str]:
    """Write results/raw/{run_id}.parquet (+ _events.parquet). Returns the paths."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    summary_path = os.path.join(cfg.output_dir, f"{cfg.run_id}.parquet")
    pq.write_table(_summary_table(summary_row(cfg, metrics_summary)),
                   summary_path, compression=COMPRESSION)
    out = {"summary": summary_path}

    if telemetry is not None:
        events_path = os.path.join(cfg.output_dir, f"{cfg.run_id}_events.parquet")
        pq.write_table(_events_table(telemetry), events_path, compression=COMPRESSION)
        out["events"] = events_path
    return out


def read_summaries(pattern: str = "results/raw/*.parquet"):
    """Concatenate every summary parquet into one table (never write into a
    shared output file from the workers - aggregate afterwards)."""
    import glob
    files = [f for f in sorted(glob.glob(pattern)) if not f.endswith("_events.parquet")]
    if not files:
        raise FileNotFoundError(f"no summary parquet matched {pattern}")
    tables = [pq.read_table(f) for f in files]
    return pa.concat_tables(tables, promote_options="default")
