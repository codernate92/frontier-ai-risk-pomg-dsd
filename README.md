# Frontier AI Risk POMG-DSD

Working paper codebase and artifact bundle for a systems-dynamics and POMG-DSD framework for frontier AI risk.

## Status

This repository is a research artifact bundle, not a production system. It includes the current manuscript, generated figures, experiment outputs, and the Python code used to construct the structural graphs and differentiable control environment.

## What This Repo Contains

- `paper.tex`: main manuscript source
- `paper.pdf`: compiled manuscript
- `pomg_dsd.py`: differentiable 97-node POMG-DSD environment
- `run_pomg_dsd.py`: main calibration, training, and evaluation pipeline
- `run_pomg_adaptation.py`: frozen-defender adversarial adaptation run
- `run_pomg_boundary_eval.py`: boundary-condition evaluation entrypoint
- `run_pomg_cbf_ablation.py`: mid-rollout CBF shutdown stress test
- `graphs.py`, `simulations.py`, `run_all.py`: structural and legacy experiment code
- `paper_figures/`: generated paper figures used by the manuscript
- `csv/`: experiment outputs used in the manuscript tables and plots
- `figures/`: legacy figure assets from earlier experiment passes

## Quick Start

Compile the paper:

```bash
tectonic paper.tex
```

Run the main POMG-DSD pipeline:

```bash
python run_pomg_dsd.py
```

Generate figure artifacts:

```bash
python generate_pomg_figures.py
python generate_paper_figures.py
```

## Repository Layout

- `paper.tex` is the authoritative manuscript source.
- `paper_figures/` and `csv/` contain the artifact outputs currently referenced by the paper.
- `pomg_results.json` is the consolidated result bundle from the main differentiable-control pipeline.
- `known_exploited_vulnerabilities.json` is the local KEV anchor used by the calibration step.

## Caveats

- This is a working-paper snapshot, so some extensions are implemented in code before all derived artifacts are rerun.
- The boundary-condition evaluation entrypoints are included, but some manuscript claims remain intentionally conservative unless refreshed outputs are regenerated.
- The repository is designed for transparency and auditability rather than minimal size.

## License

MIT. See `LICENSE`.
