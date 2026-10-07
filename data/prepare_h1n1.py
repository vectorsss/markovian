"""Build data/process_data.csv (H1N1) from the DrivenData Flu Shot Learning files.

The raw data cannot be redistributed. Download ``training_set_features.csv`` and
``training_set_labels.csv`` from
https://www.drivendata.org/competitions/66/flu-shot-learning/data/
into ``data/raw/`` and run:

    python data/prepare_h1n1.py

Processing (same as for the paper):
- merge features and labels on ``respondent_id`` (row order kept);
- numeric columns: missing values -> column mean, then truncated to int;
- text columns: codes 0..k-1 in alphabetical order, missing values -> k.
"""

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def prepare(raw_dir: Path) -> pd.DataFrame:
    features = pd.read_csv(raw_dir / "training_set_features.csv")
    labels = pd.read_csv(raw_dir / "training_set_labels.csv")
    df = features.merge(labels, on="respondent_id")
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            df[col] = s.fillna(s.mean()).astype(int)
        else:
            cats = sorted(s.dropna().unique())
            df[col] = s.map({c: i for i, c in enumerate(cats)}).fillna(len(cats)).astype(int)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw-dir", default=str(HERE / "raw"))
    p.add_argument("--out", default=str(HERE / "process_data.csv"))
    args = p.parse_args()
    df = prepare(Path(args.raw_dir))
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  {df.shape}")


if __name__ == "__main__":
    main()
