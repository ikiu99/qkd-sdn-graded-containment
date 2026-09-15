# Reproducing the paper

Everything in the paper is built from one file, `results/analysis.parquet`,
which is tracked in this repository (4.8 MB). You do not have to re-run the
simulator to check a number or regenerate a figure.

Three levels of reproduction, cheapest first.

---

## Level 1 — regenerate every figure from the shipped results (seconds)

```bash
python scripts/figures.py --pattern results/analysis.parquet
```

Writes PNG and PDF for each figure into `results/figures/`. Set `QKD_PAPER=1`
to strip the question titles from the PDFs, which is how the manuscript copies
were produced:

```bash
QKD_PAPER=1 python scripts/figures.py --pattern results/analysis.parquet
```

## Level 2 — regenerate every table and reported statistic (seconds)

```bash
python scripts/analyze.py --pattern results/analysis.parquet --out /tmp/check.parquet --all
```

Do **not** write the output back over `results/analysis.parquet` unless you are
rebuilding from raw runs; `--pattern` and `--out` pointing at the same file
overwrites the input with a re-derived copy.

## Level 3 — re-run the simulator from scratch (hours)

`results/analysis.parquet` is an aggregation of about 19,500 individual runs,
each a separate parquet file of roughly 50 kB. Those raw files are not tracked:
they total about 1 GB, which is more than a git repository should carry, and
they are fully determined by the configuration files that generated them.

To rebuild them:

```bash
# each sweep writes one parquet per run into results/raw/
python -m sim.sweep --sweep config/sweep_matrix.yaml       --no-assert
python -m sim.sweep --sweep config/sweep_ablation.yaml     --no-assert
python -m sim.sweep --sweep config/sweep_intensity.yaml    --no-assert
python -m sim.sweep --sweep config/sweep_p6_frontier.yaml  --no-assert
python -m sim.sweep --sweep config/sweep_p6_ofat.yaml      --no-assert
python -m sim.sweep --sweep config/sweep_ofat_prior_off.yaml --no-assert
python -m sim.sweep --sweep config/sweep_minimax.yaml      --no-assert
python -m sim.sweep --sweep config/sweep_family.yaml       --no-assert
python -m sim.sweep --sweep config/sweep_baselines.yaml    --no-assert
python -m sim.sweep --sweep config/sweep_struct.yaml       --no-assert
python -m sim.sweep --sweep config/sweep_control.yaml      --no-assert
python -m sim.sweep --sweep config/sweep_session_len.yaml  --no-assert
python -m sim.sweep --sweep config/sweep_p8_weights.yaml   --no-assert
python -m sim.sweep --sweep config/sweep_noise.yaml        --no-assert
python -m sim.sweep --sweep config/sweep_alpha.yaml        --no-assert

# then aggregate
python scripts/analyze.py --pattern "results/raw/*.parquet" --out results/analysis.parquet
```

A sweep skips runs whose output already exists, so an interrupted sweep can be
restarted with the same command. `--workers N` sets parallelism; the default is
one less than the core count. `--dry-run` lists what would run without running
it.

Every run is named by a hash of its full resolved configuration, so two runs
with identical settings are the same file and the sweeps above can be issued in
any order.

### Runs that are deliberately not part of the pool

`config/sweep_busiest_why.yaml` writes to `results/diagnostics/`, not
`results/raw/`. It answers one "why" question in the text (which feature carries
the AUC rise when the adversary selects the busiest relays) and must not join
the pool the figures average over.

---

## Tests

```bash
pytest
```

129 tests. They cover key conservation, the routing and admission logic against
closed-form models, the damage accounting identity, the XOR multipath truth
table, and the calibration solver.

---

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -e ".[dev]"
pip install matplotlib        # figures only
```

Exact versions the paper was produced with are pinned in
`requirements-lock.txt`.

---

## Where each figure and table comes from

| In the paper | Function in `scripts/figures.py` | Sweep |
|---|---|---|
| Fig. 1, minimax | `fig_minimax` | `sweep_minimax` |
| Fig. 2, targeting vs volume | `fig_targeting` | `sweep_control` |
| Fig. 3, published baselines | `fig_baselines` | `sweep_baselines` |
| Fig. 4, hybrid policy | `fig_hybrid` | `sweep_struct` |
| Fig. 5, one factor at a time | `fig_ofat` | `sweep_p6_ofat`, `sweep_ofat_prior_off` |
| Table 1, parameters | — | — |
| Table 2, worst case | `scripts/analyze.py --minimax` | `sweep_minimax` |
| Table 3, baselines | `scripts/analyze.py --baselines` | `sweep_baselines` |
| Table 4, hybrid | `scripts/report_struct.py` | `sweep_struct` |

`scripts/figures.py` also generates several figures that are not in the paper
(the feature ablation heat map, the score separation, the topology family, the
frontier grid and others). They are kept because they are how the results were
checked, and any of them can be regenerated from the same shipped parquet.

---

## A note on what the oracle policy (B4) is

B4 knows exactly which relays are compromised and isolates the ones the
partition guard allows. It throttles nothing. It is therefore **not** an upper
bound on damage prevented: a policy that also throttles can prevent more damage
than B4 by admitting less traffic, and ours does on the tapping profile. B4
bounds what perfect *detection* buys when spent on isolation alone. The only
real ceiling on the damage reduction ratio is 1.

---

## Licence

Code is under the MIT Licence (`LICENSE`). Data and figures are under
CC BY 4.0 (`LICENSE-DATA.md`), because they are a research record and their
reuse should carry attribution.
