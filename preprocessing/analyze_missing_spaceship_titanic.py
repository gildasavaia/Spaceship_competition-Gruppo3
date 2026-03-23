#!/usr/bin/env python3
"""
Analisi dei valori mancanti per il dataset Spaceship Titanic.

Cosa fa:
1. Legge il file CSV di train.
2. Calcola i valori mancanti per colonna.
3. Calcola quanti valori mancanti ha ogni passeggero.
4. Calcola le combinazioni di colonne mancanti più frequenti.
5. Salva dei CSV di riepilogo utili per decidere una strategia di imputazione.

Esempio:
    python analyze_missing_spaceship_titanic.py --input train.csv --output-dir report_missing
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analizza i valori mancanti nel dataset Spaceship Titanic"
    )
    parser.add_argument(
        "--input",
        default="train.csv",
        help="Percorso del file CSV di input (default: train.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="report_missing",
        help="Cartella in cui salvare i file di output (default: report_missing)",
    )
    parser.add_argument(
        "--top-combinations",
        type=int,
        default=20,
        help="Numero di combinazioni di colonne mancanti da mostrare a video (default: 20)",
    )
    return parser.parse_args()


def percentage(series: pd.Series, total: int) -> pd.Series:
    return (series / total * 100).round(2)


def missing_columns_for_row(row: pd.Series) -> str:
    missing_cols = row.index[row.isna()].tolist()
    if not missing_cols:
        return "NESSUNA"
    return " | ".join(missing_cols)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Se il path fornito non esiste, proviamo una serie di percorsi comuni
    if not input_path.exists():
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path(args.input),
            project_root / "data" / Path(args.input).name,
            project_root / Path(args.input).name,
            Path.cwd() / "data" / Path(args.input).name,
            Path(__file__).resolve().parent / Path(args.input).name,
        ]

        found = None
        tried = []
        for p in candidates:
            tried.append(str(p))
            if p.exists():
                found = p
                break

        if found is None:
            raise FileNotFoundError(
                "File non trovato. Percorsi provati:\n" + "\n".join(tried)
                + "\n\nMetti il file in data/train.csv o passa il percorso con --input"
            )

        input_path = found

    df = pd.read_csv(input_path)
    n_rows, n_cols = df.shape

    # Maschera dei missing
    missing_mask = df.isna()

    # 1) Missing per colonna
    missing_per_column = missing_mask.sum().sort_values(ascending=False)
    missing_per_column_df = pd.DataFrame(
        {
            "colonna": missing_per_column.index,
            "n_missing": missing_per_column.values,
            "percentuale_missing": percentage(missing_per_column, n_rows).values,
            "n_non_missing": (n_rows - missing_per_column).values,
        }
    )

    # 2) Missing per riga/passeggero
    missing_per_row = missing_mask.sum(axis=1)
    missing_per_row_df = pd.DataFrame(
        {
            "PassengerId": df["PassengerId"] if "PassengerId" in df.columns else df.index,
            "n_missing": missing_per_row,
            "colonne_mancanti": df.apply(missing_columns_for_row, axis=1),
        }
    ).sort_values(by=["n_missing", "PassengerId"], ascending=[False, True])

    # 3) Distribuzione del numero di missing per passeggero
    row_missing_distribution = missing_per_row.value_counts().sort_index()
    row_missing_distribution_df = pd.DataFrame(
        {
            "n_missing_nel_passeggero": row_missing_distribution.index,
            "n_passeggeri": row_missing_distribution.values,
            "percentuale_passeggeri": percentage(row_missing_distribution, n_rows).values,
        }
    )

    # 4) Combinazioni di colonne mancanti
    missing_combination_series = df.apply(missing_columns_for_row, axis=1).value_counts()
    missing_combination_df = pd.DataFrame(
        {
            "combinazione_colonne_mancanti": missing_combination_series.index,
            "n_passeggeri": missing_combination_series.values,
            "percentuale_passeggeri": percentage(missing_combination_series, n_rows).values,
        }
    )

    # 5) Matrice utile a capire quali colonne mancano insieme
    missing_together = missing_mask.T.dot(missing_mask)
    missing_together.index.name = "colonna"
    missing_together.reset_index(inplace=True)

    # Salvataggi
    save_dataframe(missing_per_column_df, output_dir / "01_missing_per_colonna.csv")
    save_dataframe(missing_per_row_df, output_dir / "02_missing_per_passeggero.csv")
    save_dataframe(row_missing_distribution_df, output_dir / "03_distribuzione_missing_per_passeggero.csv")
    save_dataframe(missing_combination_df, output_dir / "04_combinazioni_colonne_mancanti.csv")
    save_dataframe(missing_together, output_dir / "05_missing_in_comune_tra_colonne.csv")

    # Report testuale
    with open(output_dir / "report_riassuntivo.txt", "w", encoding="utf-8") as f:
        f.write("ANALISI VALORI MANCANTI - SPACESHIP TITANIC\n")
        f.write("=" * 60 + "\n")
        f.write(f"Righe: {n_rows}\n")
        f.write(f"Colonne: {n_cols}\n\n")
        f.write("Missing per colonna:\n")
        f.write(missing_per_column_df.to_string(index=False))
        f.write("\n\nDistribuzione missing per passeggero:\n")
        f.write(row_missing_distribution_df.to_string(index=False))
        f.write("\n\nTop combinazioni di colonne mancanti:\n")
        f.write(missing_combination_df.head(args.top_combinations).to_string(index=False))

    # Stampa sintetica a video
    print_section("INFO GENERALI")
    print(f"File letto: {input_path}")
    print(f"Numero righe: {n_rows}")
    print(f"Numero colonne: {n_cols}")

    print_section("MISSING PER COLONNA")
    print(missing_per_column_df.to_string(index=False))

    print_section("DISTRIBUZIONE DEL NUMERO DI MISSING PER PASSEGGERO")
    print(row_missing_distribution_df.to_string(index=False))

    print_section(f"TOP {args.top_combinations} COMBINAZIONI DI COLONNE MANCANTI")
    print(missing_combination_df.head(args.top_combinations).to_string(index=False))

    print_section("FILE SALVATI")
    for file_path in sorted(output_dir.iterdir()):
        print(f"- {file_path}")

    print_section("SUGGERIMENTO OPERATIVO")
    print(
        "Usa questi report per definire regole diverse a seconda del numero di missing per passeggero.\n"
        "Per esempio:\n"
        "- 0 missing: nessun intervento\n"
        "- 1 missing: imputazione mirata in base alla colonna\n"
        "- 2 o 3 missing: imputazione usando gruppi simili (HomePlanet, Deck, Group, CryoSleep, VIP...)\n"
        "- troppi missing nello stesso record: valuta se quel passeggero è poco affidabile o richiede regole speciali"
    )


if __name__ == "__main__":
    main()