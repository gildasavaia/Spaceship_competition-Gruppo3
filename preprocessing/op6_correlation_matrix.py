import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def compute_correlation_matrix(df: pd.DataFrame, output_dir: str = None):
    """
    Calcola:
    - matrice di correlazione
    - correlazione rispetto al target (Transported)
    - salva CSV e immagini (heatmap)

    Args:
        df (pd.DataFrame)
        output_dir (str, optional)

    Returns:
        dict con:
            - corr_matrix
            - corr_with_target
    """

    # --- Copia per sicurezza ---
    df = df.copy()

    # --- Conversione boolean ---
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    # --- Selezione numeriche ---
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    # --- Rimozione colonne inutili ---
    numeric_df = numeric_df.drop(columns=["PassengerId"], errors="ignore")

    # --- Correlation matrix ---
    corr_matrix = numeric_df.corr()

    # --- Correlazione con target ---
    corr_with_target = None
    if "Transported" in corr_matrix.columns:
        corr_with_target = corr_matrix["Transported"].sort_values(ascending=False)

    # --- Salvataggi ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # CSV
        corr_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

        if corr_with_target is not None:
            corr_with_target.to_csv(
                os.path.join(output_dir, "correlation_with_target.csv")
            )

        # --- HEATMAP COMPLETA ---
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=True)
        plt.title("Correlation Matrix")
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
        plt.close()

        # --- HEATMAP TOP FEATURE (quella utile) ---
        if corr_with_target is not None:
            top_features = (
                corr_with_target.abs()
                .sort_values(ascending=False)
                .head(10)
                .index
            )

            filtered_corr = corr_matrix.loc[top_features, top_features]

            plt.figure(figsize=(8, 6))
            sns.heatmap(filtered_corr, cmap="coolwarm", annot=True)
            plt.title("Top Correlated Features")
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, "top_correlation_matrix.png"))
            plt.close()

        print(f"[OP6] File salvati in: {output_dir}")

    return {
        "corr_matrix": corr_matrix,
        "corr_with_target": corr_with_target
    }


def run_op6(df: pd.DataFrame, config: dict = None):
    """
    Entry point pipeline
    """

    output_dir = None
    if config and "output_dir" in config:
        output_dir = config["output_dir"]

    results = compute_correlation_matrix(df, output_dir)

    return results