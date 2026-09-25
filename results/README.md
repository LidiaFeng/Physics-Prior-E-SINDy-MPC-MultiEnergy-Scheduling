# Reference results

The CSV files in this directory were generated with the formal `run_all` configuration and correspond to the paper as follows:

| File | Paper item |
|---|---|
| `table_5_2_surrogate_identification.csv` | Table 5-2, surrogate identification performance |
| `table_5_3_parameter_recovery.csv` | Table 5-3, benchmark parameter recovery |
| `table_5_4_active_learning.csv` | Table 5-4, active-learning test results |
| `table_5_5_mpc_performance.csv` | Table 5-5, MPC dispatch performance |
| `table_5_6_residual_feedback_ablation.csv` | Table 5-6, residual-feedback ablation |
| `table_5_7_trigger_threshold_sensitivity.csv` | Table 5-7, trigger-threshold sensitivity |

Values in the manuscript are rounded versions of these outputs. Solver wall-clock times are hardware-dependent and are therefore not expected to match across computers. The dispatch-cost column is expressed in million USD because the implementation divides the accumulated dollar-valued objective by `1e6`.

The files `complete_results.mat`, `experiment_results.xlsx`, and `mpc_step_log.csv` are regenerable and excluded by `.gitignore` from the public repository by default.
