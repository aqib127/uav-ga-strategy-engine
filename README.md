# UAV GA Strategy Engine

A genetic-algorithm planner for a fleet of 4 UAVs surveying a 50x50 hazard grid.
Balances **coverage & target detection** against **energy, time, and hazard exposure**
under hard constraints (flight range, no-fly zones).

## Requirements
- Python 3.10+ (tested on 3.14)
- numpy >= 1.24
- matplotlib >= 3.7

## Install

    python -m venv venv
    source venv/bin/activate          # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## Run

### Single GA run (~40 s)

    python main.py --seed 1

Produces:
- results/fig2_convergence.png  - best vs average fitness curve
- results/fig3_paths.png        - evolved flight paths over the map
- results/fig4_coverage.png     - coverage heat-map
- results/summary.json          - final fitness + objectives

### GA vs Random search (~3 min)

    python run_comparison.py --seeds 3

Produces results/fig5_comparison.png + results/comparison.json.

### Hyper-parameter sensitivity (~2 min)

    python run_sensitivity.py --seeds 2 --gens 30

Produces results/fig6_sensitivity.png + results/sensitivity.json.

## Project structure

    uav-ga-strategy-engine/
    |-- uav_ga/
    |   |-- config.py          # all parameters (grid, GA, weights, limits)
    |   |-- environment.py     # priority zones, hazards, no-fly zones, targets
    |   |-- chromosome.py      # hybrid real/binary encoding (21 genes x 4 UAVs)
    |   |-- fitness.py         # multi-objective fitness + penalties
    |   |-- operators.py       # tournament, crossover, mutation, repair
    |   |-- ga.py              # main evolution loop
    |   `-- baseline.py        # random search
    |-- main.py                # single GA run
    |-- run_comparison.py      # GA vs random
    |-- run_sensitivity.py     # hyper-parameter sweep
    `-- results/               # generated figures + JSON summaries

## Method summary

- **Chromosome** (84 genes): per UAV = deployment (r,c) + 8 waypoints (r,c each)
  + speed + sensor power + sensor ON/OFF (binary). Hybrid real/binary encoding
  so spatial coordinates and continuous controls coexist with a binary switch.

- **Fitness** (multi-objective, weighted sum):

        F = 0.35*Coverage + 0.25*Detection
          - 0.15*Energy - 0.10*Time - 0.15*Hazard - Penalty

  All objectives normalised to [0,1]. Dynamic penalty for flight-range
  violations and no-fly-zone intrusions.

- **Operators**: tournament selection (k=3), segment-swap + geometric-blend
  crossover, Gaussian spatial mutation (sigma=2), and a repair operator that
  pushes waypoints off no-fly cells to the nearest free cell.

## Results (seed = 1)
- Best fitness: **0.1752**
- Coverage: 0.630 | Detection: 0.238 | Energy: 0.444 | Time: 0.380
- Hazard: 0.000 | Penalty: 0.000
- Runtime: ~38 s

## Team
- **Aqib Ali** (Roll: <03-134261-007>) - Code + GitHub
