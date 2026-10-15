"""Convert the 'Disease Symptom Prediction' (itachi9604) dataset.csv into a binary symptom matrix.

Input : Disease, Symptom_1 ... Symptom_17  (symptom names as text, many blanks)
Output: one 0/1 column per symptom + 'disease' label column

Usage: python src/prepare_dataset.py --input data/raw/dataset.csv --output data/processed/symptoms_binary.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def clean(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/processed/symptoms_binary.csv")
    ap.add_argument("--dedupe", action="store_true", help="Drop duplicate rows from the output")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    label_col = df.columns[0]                      # 'Disease'
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]

    records = []
    for _, row in df.iterrows():
        symptoms = {clean(v) for v in row[symptom_cols].dropna() if str(v).strip()}
        records.append((clean(row[label_col]), symptoms))

    all_symptoms = sorted({s for _, syms in records for s in syms})
    out = pd.DataFrame(0, index=range(len(records)), columns=all_symptoms)
    for i, (_, syms) in enumerate(records):
        out.loc[i, list(syms)] = 1
    out["disease"] = [d for d, _ in records]
    if args.dedupe:
        out = out.drop_duplicates().reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {args.output}: {len(out)} rows, {len(all_symptoms)} symptoms, {out['disease'].nunique()} diseases")
    print(f"Duplicate rows remaining: {out.duplicated().sum()}")


if __name__ == "__main__":
    main()
