import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stampa la top-k dei valori per ogni feature: conteggio e percentuale."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "train_processed.csv"),
        help="Percorso CSV input (default: data/train_processed.csv)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Numero di valori top da mostrare per feature (default: 5)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="(Opzionale) Percorso CSV per salvare la tabella top-k combinata.",
    )
    return parser.parse_args()


def safe_value_label(value) -> str:
    if pd.isna(value):
        return "<NULL>"
    return str(value)


def build_topk(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    total_rows = len(df)
    rows = []

    for col in df.columns:
        vc = df[col].value_counts(dropna=False).head(k)
        rank = 1
        for value, count in vc.items():
            rows.append(
                {
                    "feature": col,
                    "rank": int(rank),
                    "value": safe_value_label(value),
                    "count": int(count),
                    "percentage": round((count / total_rows) * 100, 4),
                }
            )
            rank += 1

    return pd.DataFrame(rows).sort_values(["feature", "rank"]).reset_index(drop=True)


def print_topk(topk_df: pd.DataFrame) -> None:
    for feature in topk_df["feature"].unique():
        print(f"\n=== {feature} ===")
        feature_rows = topk_df[topk_df["feature"] == feature]
        for _, row in feature_rows.iterrows():
            print(
                f"{int(row['rank'])}. Valore: {row['value']} | Conteggio: {row['count']} | Percentuale: {row['percentage']:.2f}%"
            )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    topk = int(args.topk)
    output_csv = Path(args.output_csv) if args.output_csv else None

    if not input_path.exists():
        raise SystemExit(f"File non trovato: {input_path}")

    df = pd.read_csv(input_path)

    topk_df = build_topk(df, k=topk)

    print_topk(topk_df)

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        topk_df.to_csv(output_csv, index=False)
        print(f"\nTop-{topk} salvata in: {output_csv}")


if __name__ == "__main__":
    main()
