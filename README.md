# Frontier AI Risk POMG-DSD

Working paper codebase and artifact bundle for a systems-dynamics and POMG-DSD framework for frontier AI risk.

## Contents

- `paper.tex`: main manuscript source
- `paper.pdf`: compiled manuscript
- `pomg_dsd.py`: differentiable 97-node POMG-DSD environment
- `run_pomg_dsd.py`: main calibration, training, and evaluation pipeline
- `run_pomg_adaptation.py`: frozen-defender adversarial adaptation run
- `run_pomg_boundary_eval.py`: boundary-condition evaluation entrypoint
- `run_pomg_cbf_ablation.py`: CBF shutdown stress test entrypoint
- `graphs.py`, `simulations.py`, `run_all.py`: structural and legacy experiment code
- `paper_figures/`: generated paper figures
- `csv/`: experiment outputs used in the manuscript
- `figures/`: legacy figure assets

## Build

Compile the manuscript with `tectonic paper.tex`.

## Notes

This repository is an artifact bundle for the current working paper state. Some boundary-condition extensions are implemented in code but may require rerunning the corresponding scripts to refresh all derived artifacts.
