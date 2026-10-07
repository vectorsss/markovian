# Datasets

`benchmark/data.py` reads four CSV files from this directory. Crime and MIC are
included (CC BY 4.0). Housing and H1N1 are not included; the scripts below
rebuild them, byte-identical to the files used in the paper.

| File | Dataset | Target | Included |
|---|---|---|---|
| `crimedata_processing.csv` | Communities and Crime Unnormalized (UCI) | `ViolentCrimesPerPop` (regression) | yes |
| `MIC_processing.csv` | Myocardial infarction complications (UCI) | `LET_IS` (classification) | yes |
| `housing_price.csv` | Boston housing (StatLib) | `MEDV` (regression) | no: `python data/prepare_housing.py` |
| `process_data.csv` | Flu Shot Learning, H1N1 (DrivenData) | `h1n1_vaccine` (classification) | no: see below |

## Crime

Source: M. Redmond (2009), *Communities and Crime Unnormalized* [Dataset], UCI
Machine Learning Repository, https://doi.org/10.24432/C5PC8X. License: CC BY 4.0.

Changes made to the original file (2215 rows, all kept):
- dropped the 5 identifier columns (`communityname`, `state`, `countyCode`,
  `communityCode`, `fold`) and the 22 police (LEMAS) columns, which have more
  than 84% missing values;
- columns named as in the original attribute list;
- missing values replaced by the column mean;
- added `communityName_freq` (share of rows with the same community name) and
  `state_encoded` (smoothed target encoding of the state on
  `ViolentCrimesPerPop`, fitted on all rows; min_samples_leaf = 20,
  smoothing = 10).

The other crime counts and rates (`murders`, `murdPerPop`, `rapes`, ...) are kept
as features; `benchmark/data.py` drops only `ViolentCrimesPerPop` and
`nonViolPerPop` from the features.

## MIC

Source: S.E. Golovenkin, V.A. Shulman, D.A. Rossiev, P.A. Shesternya,
S.Yu. Nikulina, Yu.V. Orlova, V.F. Voino-Yasenetsky (2020), *Myocardial
infarction complications* [Dataset], UCI Machine Learning Repository,
https://doi.org/10.24432/C53P5M. License: CC BY 4.0.

Changes made to the original file (1700 rows, all kept):
- dropped `ID` and the 11 columns with more than 20% missing values;
- missing values replaced by the column mean;
- `LET_IS` made binary: 0 = alive, 1 = death (any cause).

The 11 complication columns (`FIBR_PREDS` ... `P_IM_STEN`, listed as outputs in
the original description) are kept as features.

## Housing

Source: D. Harrison and D.L. Rubinfeld (1978), "Hedonic prices and the demand for
clean air", *Journal of Environmental Economics and Management* 5, 81-102;
data from StatLib, https://lib.stat.cmu.edu/datasets/boston.

```bash
python data/prepare_housing.py    # downloads from StatLib, writes housing_price.csv
```

No changes: the 506 records and 14 columns are written as CSV.

## H1N1

Source: DrivenData, *Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines*,
https://www.drivendata.org/competitions/66/flu-shot-learning/ (data from the
CDC National 2009 H1N1 Flu Survey). The competition rules do not allow
redistribution, so the data are not included.

1. Join the competition and download `training_set_features.csv` and
   `training_set_labels.csv` into `data/raw/`.
2. Run `python data/prepare_h1n1.py` to write `process_data.csv`.

Changes made (26707 rows, all kept): features and labels merged on
`respondent_id`; numeric columns: missing values replaced by the column mean,
truncated to an integer; text columns: codes 0..k-1 in alphabetical order,
missing values coded as k.
