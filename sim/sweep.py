"""Cartesian sweep expansion and parallel, resumable execution.

Every worker is independent - no shared state, each run writes its own parquet.
Never write into one shared output file; aggregate afterwards with
``logging_io.read_summaries``.

Sweep file format::

    base: config/base_t3_otp.yaml
    output_dir: results/raw
    axes:
      demand.lam: [0.1, 0.2, 0.3]
      seed_demand: [1, 2, 3, 4, 5]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from .config import SimConfig, _deep_merge, _expand_dotted, config_from_dict
from .runner import run


def expand(sweep_yaml: str) -> list[SimConfig]:
    """Cartesian product of the declared axes -> configs with unique run_ids."""
    with open(sweep_yaml, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    base_path = doc.get("base")
    if not base_path:
        raise ValueError("sweep file needs a 'base' key pointing at a config YAML")
    with open(base_path, "r", encoding="utf-8") as fh:
        base_raw = yaml.safe_load(fh) or {}

    common = {k: v for k, v in doc.items()
              if k not in ("base", "axes", "variants")}
    axes: dict = doc.get("axes") or {}
    keys = sorted(axes)
    combos = list(itertools.product(*(axes[k] for k in keys))) if keys else [()]
    # ``variants`` holds whole override dicts, for fields that only make sense
    # when they move together - swapping topology.name and topology.path, say.
    variants: list = doc.get("variants") or [{}]

    configs, seen = [], set()
    for variant, combo in itertools.product(variants, combos):
        raw = _deep_merge(base_raw, _expand_dotted(common))
        raw = _deep_merge(raw, _expand_dotted(dict(variant)))
        raw = _deep_merge(raw, _expand_dotted(dict(zip(keys, combo))))
        raw.pop("run_id", None)                     # always re-derive the identity
        cfg = config_from_dict(raw)
        if cfg.run_id in seen:
            continue                                # identical parameter point
        seen.add(cfg.run_id)
        configs.append(cfg)
    return configs


def output_path(cfg: SimConfig) -> str:
    return os.path.join(cfg.output_dir, f"{cfg.run_id}.parquet")


def _run_one(payload: tuple[dict, bool, bool]) -> dict:
    """Worker entry point. Must stay module level so it can be pickled."""
    raw, asserts, events = payload
    cfg = config_from_dict(raw)
    try:
        summary = run(cfg, write=True, asserts=asserts, progress=False,
                      events=events)
        return {"run_id": cfg.run_id, "ok": True,
                "wall_time_s": summary.get("wall_time_s"), "RR": summary.get("RR")}
    except Exception:
        return {"run_id": cfg.run_id, "ok": False, "error": traceback.format_exc()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.sweep",
                                 description="Expand and run a sweep in parallel.")
    ap.add_argument("--sweep", required=True, help="path to the sweep YAML")
    ap.add_argument("--workers", type=int, default=None,
                    help="default: cpu_count() - 1, so the laptop stays usable")
    ap.add_argument("--no-assert", action="store_true")
    ap.add_argument("--events", action="store_true",
                    help="also write the per-node event log. OFF by default: at "
                         "~10 MB per run a phase 6 matrix would be tens of GB, "
                         "and every figure it feeds is already summarised into "
                         "the summary row")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the output parquet already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="only list the runs that would be executed")
    args = ap.parse_args(argv)

    configs = expand(args.sweep)
    pending = configs if args.force else [
        c for c in configs if not os.path.exists(output_path(c))]
    done = len(configs) - len(pending)
    print(f"{len(configs)} runs in the sweep, {done} already on disk, "
          f"{len(pending)} to execute")

    if args.dry_run:
        for c in pending:
            print(f"  {c.run_id}  {c.topology.name}/{c.demand.km_mode}  "
                  f"lam={c.demand.lam}  profile={c.attack.profile}  "
                  f"f={c.attack.f}  sel={c.attack.selection}  "
                  f"prior={c.detector.use_prior}  noise={c.noise.scale}  "
                  f"alpha={c.detector.alpha}  seed_attack={c.seed_attack}")
        return 0
    if not pending:
        return 0

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    payloads = [({k: v for k, v in _as_raw(c).items()},
                 not args.no_assert, args.events) for c in pending]

    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, p): p[0].get("run_id") for p in payloads}
        iterator = as_completed(futures)
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(futures), desc="sweep", unit="run")
        except ImportError:
            pass
        for fut in iterator:
            res = fut.result()
            if not res["ok"]:
                # one bad run must never take the sweep down
                failures.append(res)
                print(f"\nFAILED {res['run_id']}\n{res['error']}", file=sys.stderr)

    print(f"finished: {len(pending) - len(failures)} ok, {len(failures)} failed")
    return 1 if failures else 0


def _as_raw(cfg: SimConfig) -> dict:
    """Config -> plain dict, so workers rebuild it instead of unpickling numpy."""
    from .config import to_dict
    return to_dict(cfg)


if __name__ == "__main__":
    sys.exit(main())
