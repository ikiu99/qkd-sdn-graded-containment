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

`results/analysis.parquet` is an aggregation of 28,662 individual runs, each a
separate parquet file of roughly 50 kB. Those raw files are not tracked: they
total about 1.4 GB, which is more than a git repository should carry, and they
are fully determined by the configuration files that generated them.

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

# added with the reviewer response
python -m sim.sweep --sweep config/sweep_collusion.yaml     --no-assert
python -m sim.sweep --sweep config/sweep_hybrid_agg.yaml    --no-assert
python -m sim.sweep --sweep config/sweep_joint_minimax.yaml --no-assert

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

### Selecting a cell

`cell()` in `scripts/figures.py` pins every categorical mode at the value the
paper uses: the coordinated-liar flag, the session-teardown choice, the
path-risk aggregation and the routing objective. `op()` does the same for every
policy knob a caller does not name. Both exist because each of those axes was
added by a study that selects on it, and each one, left unpinned, turned some
unrelated figure's operating point into an average over a variant that study
measured and rejected. Pass `None` to relax a pin deliberately.

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

137 tests. They cover key conservation, the routing and admission logic against
closed-form models, the damage accounting identity, the XOR multipath truth
table, the calibration solver, the coordinated-liar path and the three
path-risk aggregations.

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
| Table 2, published baselines | `scripts/analyze.py --baselines` | `sweep_baselines` |
| Table 3, hybrid policy | `scripts/report_struct.py` | `sweep_struct` |

The minimax worst cases had a table of their own until the manuscript was cut
back; Figure 1 carries them now, with each policy's worst case marked on its
curve. `scripts/analyze.py --minimax` still prints them.

`scripts/figures.py` also produces several figures that are not in the paper;
they were how the results were checked and regenerate from the same file.

Note that the oracle policy B4 isolates the compromised set and never throttles,
so it is not an upper bound on damage prevented, only on what perfect detection
buys when spent on isolation.

---

## Licence

Code is under the MIT Licence (`LICENSE`). Data and figures are under
CC BY 4.0 (`LICENSE-DATA.md`), because they are a research record and their
reuse should carry attribution.
