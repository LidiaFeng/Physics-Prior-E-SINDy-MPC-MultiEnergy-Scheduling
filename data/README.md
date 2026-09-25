# Synthetic datasets

This directory contains all CSV datasets read by `run_all.m`. The files are synthetic research data constructed from publicly available engineering parameters for a representative hydro-wind-solar-pumped-storage system. 

## Dataset organization

```text
data/
|-- identification/
|   |-- low_data_high_noise/
|   |   |-- train.csv                         1,400 samples
|   |   `-- test.csv                            300 samples
|   |-- typical_day_transfer/
|   |   |-- train.csv                         8,664 samples
|   |   `-- test.csv                             96 samples
|   `-- extreme_extrapolation/
|       |-- train.csv                         8,760 samples
|       |-- active_learning_pool.csv            192 samples
|       `-- test.csv                            192 samples
`-- mpc/
    |-- low_data_high_noise.csv                 300 samples
    |-- typical_day_transfer.csv                 96 samples
    `-- extreme_15min.csv                       384 samples
```

The low-data, high-noise scenario uses 20 non-overlapping 100-hour windows with a 14/3/3 train-validation-test split. Gaussian noise is scaled independently for each nonzero continuous physical field to an SNR of 10 dB. The random seed used to prepare this scenario is `20260610` for identification data and `20260710` for MPC-transfer data.

The typical-day transfer training set excludes four representative seasonal days. Its test set contains 96 hourly samples from the four excluded days.

The extreme-extrapolation identification data contain a 192-sample active-learning candidate pool and an independent 192-sample test set. The MPC dataset contains all 384 samples at 15-minute resolution.

## Installed capacities used in the paper

| Generation type | Installed capacity MW | Share |
|---|---:|---:|
| Hydropower | 10,800 | 66.67% |
| Wind | 1,200 | 7.41% |
| Photovoltaics | 3,000 | 18.52% |
| Pumped storage | 1,200 | 7.41% |
| Total | 16,200 | 100.00% |

These values correspond to Table 5-1 of the paper.

## Units and conventions

- Power variables are expressed in MW.
- Stored energy is expressed in MWh.
- Time resolution is expressed in minutes.
- Electricity tariffs are stored in CNY/MWh in the CSV files and converted to USD/MWh in load_dataset.m using an exchange rate of 7.2 CNY/USD. Accordingly,   the total dispatch costs reported by the program are expressed in million USD.
- Pumped-storage generation and pumping are stored as separate nonnegative variables.
- `Scenario index` and `Window index` identify independent sequences and prevent prediction horizons from crossing scenario boundaries.

All filenames and column headers are ASCII English to support reproducible execution across operating systems and MATLAB installations.
