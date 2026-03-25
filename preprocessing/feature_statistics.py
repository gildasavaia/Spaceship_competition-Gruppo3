import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcola le statistiche di distribuzione per ogni feature: "
            "conteggi e percentuali per ogni valore."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "train_processed.csv"),
        help="Percorso del dataset CSV in input.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "report_missing" / "06_feature_distribution_stats.csv"),
        help="Percorso CSV di output con la distribuzione completa.",
    )
    parser.add_argument(
        "--output-overview",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "report_missing" / "07_feature_overview_stats.csv"),
        help="Percorso CSV di output con overview per feature.",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "report_missing" / "06_feature_distribution_stats.txt"),
        help="Percorso TXT di output leggibile per umani.",
    )
    return parser.parse_args()


def safe_value_label(value) -> str:
    if pd.isna(value):
        return "<NULL>"
    return str(value)


def build_feature_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    rows = []

    for col in df.columns:
        series = df[col]
        value_counts = series.value_counts(dropna=False)

        for value, count in value_counts.items():
            rows.append(
                {
                    "feature": col,
                    "value": safe_value_label(value),
                    "count": int(count),
                    "percentage": round((count / total_rows) * 100, 4),
                }
            )

    dist_df = pd.DataFrame(rows)
    dist_df = dist_df.sort_values(["feature", "count"], ascending=[True, False]).reset_index(drop=True)
    return dist_df


def build_feature_overview(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    rows = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        rows.append(
            {
                "feature": col,
                "dtype": str(df[col].dtype),
                "total_rows": total_rows,
                "missing_count": missing_count,
                "missing_percentage": round((missing_count / total_rows) * 100, 4),
                "unique_non_null_values": int(df[col].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def write_human_report(dist_df: pd.DataFrame, output_txt: Path) -> None:
    with output_txt.open("w", encoding="utf-8") as f:
        current_feature = None

        for _, row in dist_df.iterrows():
            feature = row["feature"]
            if feature != current_feature:
                if current_feature is not None:
                    f.write("\n")
                f.write(f"=== {feature} ===\n")
                current_feature = feature

            f.write(
                f"- Valore: {row['value']} | Conteggio: {row['count']} | Percentuale: {row['percentage']:.2f}%\n"
            )


def print_console_preview(dist_df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("DISTRIBUZIONE PERCENTUALE PER FEATURE (ANTEPRIMA)")
    print("=" * 90)

    for feature in dist_df["feature"].unique():
        feature_rows = dist_df[dist_df["feature"] == feature]
        print(f"\n[{feature}] Tutti i valori:")
        for _, row in feature_rows.iterrows():
            print(f"  - {row['value']}: {row['count']} ({row['percentage']:.2f}%)")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_overview = Path(args.output_overview)
    output_txt = Path(args.output_txt)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    dist_df = build_feature_distribution(df)
    overview_df = build_feature_overview(df)

    dist_df.to_csv(output_csv, index=False)
    overview_df.to_csv(output_overview, index=False)
    write_human_report(dist_df, output_txt)

    print_console_preview(dist_df)

    print("\n" + "=" * 90)
    print(f"Dataset letto da: {input_path}")
    print(f"Distribuzione completa salvata in: {output_csv}")
    print(f"Overview feature salvata in: {output_overview}")
    print(f"Report testuale salvato in: {output_txt}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
