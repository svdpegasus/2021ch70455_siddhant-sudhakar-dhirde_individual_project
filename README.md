# Self-Organized Criticality in Forest Fire Systems

## Overview
This repository contains all materials for the individual complexity science project
(April 2026) demonstrating **Self-Organized Criticality (SOC)** using the
**Drossel–Schwabl Forest Fire Cellular Automaton Model**.

## Files
| File | Description |
|------|-------------|
| `code.py` | Python simulation (main code) |
| `2021ch70455_siddhant sudhakar dhirde_individual_project.tex` | LaTeX source for the journal-style report |
| `2021ch70455_siddhant sudhakar dhirde_individual_project.pdf` | Compiled PDF report |
| `figures/` | All generated plots (auto-created by simulation) |

## Requirements
```bash
pip install numpy matplotlib scipy
```

## How to Run
```bash
python code.py
```
This generates all figures in the `figures/` directory.

## To Compile the LaTeX Report
```bash
pdflatex 2021ch70455_siddhant sudhakar dhirde_individual_project.tex
pdflatex 2021ch70455_siddhant sudhakar dhirde_individual_project.tex   # run twice for cross-references
```

## Model Description
The Drossel–Schwabl model (1992) is a stochastic cellular automaton on an L×L grid.

**Rules (applied synchronously each step):**
1. BURNING → EMPTY (ash)
2. TREE with burning neighbour → BURNING (fire spreads)
3. TREE (no burning neighbour) → BURNING with probability *f* (lightning)
4. EMPTY → TREE with probability *p* (growth)

**Parameters:**
- L = 128, p = 0.05, f = 0.0005 (p/f = 100)
- 8,000 steps total; 2,000 step burn-in
- Both 4-neighbour (von Neumann) and 8-neighbour (Moore) topologies tested

## Key Results
| Connectivity | Power-law exponent τ | Max avalanche |
|:---:|:---:|:---:|
| 4 (von Neumann) | 0.40 | 2,376 |
| 8 (Moore) | 0.34 | 2,750 |

