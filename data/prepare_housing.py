"""Build data/housing_price.csv (Boston housing) from the original StatLib file.

Source: Harrison, D. and Rubinfeld, D.L. (1978), "Hedonic prices and the demand
for clean air", J. Environ. Economics & Management 5, 81-102; distributed by
StatLib at https://lib.stat.cmu.edu/datasets/boston

    python data/prepare_housing.py

No preprocessing: the 506 records are written as CSV with the original column
names (CHAS and RAD as integers). The output is identical to the file used in
the paper.
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
URL = "https://lib.stat.cmu.edu/datasets/boston"
COLUMNS = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
           "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
HEADER_LINES = 22  # description lines before the data in the StatLib file


def prepare(text: str) -> pd.DataFrame:
    lines = text.splitlines()[HEADER_LINES:]
    values = np.array([float(x) for line in lines for x in line.split()])
    df = pd.DataFrame(values.reshape(-1, len(COLUMNS)), columns=COLUMNS)
    for col in ("CHAS", "RAD"):
        df[col] = df[col].astype(int)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(HERE / "housing_price.csv"))
    args = p.parse_args()
    with urllib.request.urlopen(URL) as r:
        text = r.read().decode("utf-8")
    df = prepare(text)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  {df.shape}")


if __name__ == "__main__":
    main()
