# Dual-mode CO2/N2 hollow-fibre module analysis (Matrimid 5218)

Code and results for: *Can pure-gas permeability data design a mixed-gas module? A correctability map for dual-mode CO2/N2 transport in Matrimid 5218 hollow fibres* (M.V. Ajoku, E.D. Ezeokolie).

## Contents
- `matrimid_dualmode_module_analysis.py`: complete model, calibration, analyses and figures
- `results/`: numerical results (JSON), one file per analysis section
- `figures/`: all manuscript figures (600 dpi PNG and PDF)

## Requirements
Python >= 3.9, numpy, scipy, matplotlib

## Usage
    python matrimid_dualmode_module_analysis.py

Sections can be switched on or off in the `RUN` dictionary at the top. A full run takes about 30 min on one CPU. The section-to-manuscript map is in the script header.

## Licence
MIT
