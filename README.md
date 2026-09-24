# Physics-Informed E-SINDy-MPC

MATLAB implementation accompanying the paper **Physics-Informed E-SINDy-MPC for Coordinated Optimal Dispatch of Hydro-Wind-Solar-Pumped Storage Multi-Energy Systems**

The code implements physics-informed ensemble sparse identification, risk-aware active learning, centralized MPC, periodic distributed MPC, and event-triggered distributed MPC with ADMM coordination.

## Requirements

- MATLAB R2024b or later
- Optimization Toolbox (`quadprog`)

## Data layout

Place the English CSV datasets under the following paths:

```text
data/
|-- identification/
|   |-- low_data_high_noise/
|   |   |-- train.csv
|   |   `-- test.csv
|   |-- typical_day_transfer/
|   |   |-- train.csv
|   |   `-- test.csv
|   `-- extreme_extrapolation/
|       |-- train.csv
|       |-- active_learning_pool.csv
|       `-- test.csv
`-- mpc/
    |-- low_data_high_noise.csv
    |-- typical_day_transfer.csv
    `-- extreme_15min.csv
```

The repository already includes the ten English CSV datasets used by `run_all.m`. Their headers match `load_dataset.m`, including `Time index`, `Window index`, `Scenario index`, `Time resolution_min`, `Load power_MW`, `Wind power_MW`, `PV power_MW`, `Hydropower output_MW`, `Pumped-storage generation_MW`, `Pumped-storage pumping_MW`, `Stored energy_MWh`, and `Net load_MW`.

## Running the experiments

Set the repository root as the current MATLAB folder and run:

```matlab
run_all
```

This executes the full experiment suite, using 20 random seeds for active learning. For a shorter functional check, run:

```matlab
run_all(true)
```

Quick mode uses fewer random seeds and must not be used to reproduce the final numerical results reported in the paper. The main pseudorandom seed is fixed at `20260810`.


## Main files

- `run_all.m`: complete reproduction entry point
- `run_exp1.m`: SINDy, W-SINDy, and PI-E-SINDy comparison
- `run_recovery.m`: sparse-equation recovery benchmark
- `run_exp2_al.m`: active-learning comparison
- `run_exp3_mpc.m`: MPC comparison, ablation, and sensitivity studies
- `simulate_mpc.m`: closed-loop MPC simulation
- `admm_coord.m`: distributed ADMM coordination
- `make_figures.m`: generation of publication figures

## Reproducibility notes

- Run all scripts from the repository root so relative paths resolve correctly.
- Solver wall-clock times vary across computers even when optimization results agree.
- The public data should be described as synthetic datasets constructed from publicly available engineering parameters, not as operational SCADA data.

## Citation

Please cite the associated paper when using this software or its datasets. Machine-readable citation metadata are provided in `CITATION.cff`.

## License

The source code is released under the MIT License. See `LICENSE`.
