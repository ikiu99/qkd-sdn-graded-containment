# qkd-sdn-sim

Simulator and analysis code for *Graded detection and containment of compromised
relays in software-defined quantum key distribution networks*.

A trusted relay in a QKD network sees every key that crosses it, and a
compromised one leaves the quantum channel untouched. This code models that
threat and compares ways of responding to it.

## What it does

- Discrete fixed-step simulation of a QKD network: per-link key generation,
  Poisson session demand, k-shortest-path routing with admission control, and
  key consumption in both one-time-pad and AES re-keying modes.
- Four adversary profiles for a compromised relay, plus an observation-noise
  model.
- A risk detector over three key-management telemetry features, and the
  response policies built on it, including reimplementations of three published
  methods used as baselines.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
.venv/Scripts/activate       # source .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"
pip install matplotlib
```

## Quick start

```bash
# one run
python -m sim.runner --config config/attack_t3_otp.yaml

# regenerate the paper figures from the results in this repository
python scripts/figures.py --pattern results/analysis.parquet

# tests
pytest
```

[REPRODUCE.md](REPRODUCE.md) explains how to rebuild everything, including
re-running the sweeps from scratch.

## Layout

| | |
|---|---|
| `sim/` | the simulator: topology, key generation, demand, routing, attack, detector, policies |
| `scripts/` | sweeps, analysis and figure generation |
| `config/` | run and sweep configurations |
| `tests/` | 137 tests, including validation against closed-form models |
| `results/analysis.parquet` | aggregated results; every figure and table is built from this |
| `docs/related_work.md` | sources behind the model parameters |

## Licence

Code under MIT ([`LICENSE`](LICENSE)); data and figures under CC BY 4.0
([`LICENSE-DATA.md`](LICENSE-DATA.md)).
