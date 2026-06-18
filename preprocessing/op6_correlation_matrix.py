import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Op6Result:
    """Risultato dell'operazione OP6: matrice di correlazione."""
    corr_matrix: pd.DataFrame
    corr_with_target: Optional[pd.Series] = field(default=None)


def run_op6(df: pd.DataFrame, output_dir: str = "outputs/op6") -> Op6Result:
    """
    Calcola la matrice di correlazione sul DataFrame già preprocessato.

    Gestisce automaticamente le colonne categoriche (object) tramite
    one-hot encoding temporaneo, così anche le feature come HomePlanet,
    Destination, Deck e Side vengono incluse nella matrice di correlazione.

    Args:
        df: DataFrame già preprocessato (post encoding/scaling).
        output_dir: Cartella in cui salvare l'heatmap (default: 'outputs/op6').

    Returns:
        Op6Result con la matrice di correlazione e le correlazioni con il target.
    """
    # Creiamo la cartella di output se non esiste
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- PREPARAZIONE: encoding temporaneo delle categoriche ---
    df_analysis = df.copy()

    # Convertiamo le colonne booleane in int per la correlazione
    bool_cols = df_analysis.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df_analysis[col] = df_analysis[col].astype(int)

    # Identifichiamo le colonne categoriche (object)
    cat_cols = df_analysis.select_dtypes(include=["object"]).columns.tolist()

    if cat_cols:
        print(f"   -> Colonne categoriche trovate: {cat_cols}")
        print("   -> Applico one-hot encoding temporaneo per la correlazione...")
        # One-hot encoding senza drop_first per esplorare tutte le categorie
        df_analysis = pd.get_dummies(df_analysis, columns=cat_cols, drop_first=False, dtype=int)

    # Selezioniamo solo le colonne numeriche
    df_numerico = df_analysis.select_dtypes(include=["number"])

    print(f"   -> Totale colonne nella matrice: {len(df_numerico.columns)}")

    # --- CALCOLO DELLA MATRICE DI CORRELAZIONE ---
    corr_matrix = df_numerico.corr()

    # --- VISUALIZZAZIONE HEATMAP COMPLETA ---
    n_cols = len(corr_matrix.columns)
    fig_size = max(20, n_cols * 0.6)  # Scala la figura in base al numero di colonne
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        corr_matrix,
        annot=n_cols <= 30,  # Annota solo se le colonne non sono troppe
        cmap="coolwarm",
        fmt=".2f" if n_cols <= 30 else "",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    plt.title("Matrice di Correlazione (Completa con Categoriche)", fontsize=24)
    plt.xticks(rotation=45, ha="right", fontsize=max(8, 14 - n_cols // 5))
    plt.yticks(fontsize=max(8, 14 - n_cols // 5))
    plt.tight_layout()

    save_path = out_path / "correlation_matrix.png"
    plt.savefig(str(save_path), dpi=300)
    plt.show()
    plt.close()
    print(f"   -> Heatmap completa salvata in: {save_path}")

    # --- ESTRAZIONE CORRELAZIONE RISPETTO AL TARGET ---
    corr_target = None
    if "Transported" in corr_matrix.columns:
        corr_target = corr_matrix["Transported"].drop("Transported").abs().sort_values(ascending=False)

        # --- GRAFICO A BARRE DELLE CORRELAZIONI CON IL TARGET ---
        plt.figure(figsize=(14, max(6, len(corr_target) * 0.35)))
        colors = ["#e74c3c" if v >= 0.15 else "#3498db" if v >= 0.05 else "#bdc3c7"
                   for v in corr_target.values]
        corr_target.plot(kind="barh", color=colors, edgecolor="black", linewidth=0.5)
        plt.xlabel("Correlazione (valore assoluto)", fontsize=14)
        plt.ylabel("")
        plt.title("Correlazione con Transported (tutte le feature)", fontsize=18)
        plt.axvline(x=0.15, color="red", linestyle="--", alpha=0.5, label="Soglia 0.15")
        plt.axvline(x=0.05, color="blue", linestyle="--", alpha=0.5, label="Soglia 0.05")
        plt.legend(fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        bar_path = out_path / "correlation_with_target.png"
        plt.savefig(str(bar_path), dpi=300)
        plt.show()
        plt.close()
        print(f"   -> Grafico correlazioni con target salvato in: {bar_path}")

        print("\n   -> Top 15 correlazioni con Transported:")
        print(corr_target.head(15).to_string())

    return Op6Result(corr_matrix=corr_matrix, corr_with_target=corr_target)
