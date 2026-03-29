"""# Importa le funzioni dai file op1, op2, op3 e dal nuovo op4
from op1_read_file import run_load_and_convert_to_csv
from op2_data_evaluation import run_evaluation
from op3_split_dataset import run_split_dataset  
from op4_handler_nullvalue import run_handle_null_values # <-- NUOVO IMPORT
from pathlib import Path

# Percorso al dataset
data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"

# OP1: Lettura
risultato = run_load_and_convert_to_csv(str(data_path))
mio_dataframe = risultato.df_output

# OP3: Feature Engineering (Split)
risultato_split = run_split_dataset(mio_dataframe)
mio_dataframe_modificato = risultato_split.df_output

# OP2: Valutazione finale sui dati puliti e imputati
valutazione = run_evaluation(mio_dataframe_imputato)
print("Valutazione completata.")

# --- NUOVO STEP OP4 ---
# OP4: Gestione dei valori nulli (DEVE avvenire dopo lo split, perché usa la colonna 'Group')
risultato_imputazione = run_handle_null_values(mio_dataframe_modificato)
mio_dataframe_imputato = risultato_imputazione.df_output
# ----------------------

# Salva il DataFrame imputato
output_path = data_path.with_name("train_processed.csv")
mio_dataframe_imputato.to_csv(output_path, index=False)
print(f"Dataset finale salvato in: {output_path}")

pipeline iniziale di dario
"""


""" pipeline gilda 
from pathlib import Path
import json

from op1_read_file import run_load_and_convert_to_csv
from op2_data_evaluation import run_evaluation
from op3_split_dataset import run_split_dataset
from op4_handler_nullvalue import run_handle_null_values
from op5_sumcosts_names import run_op5
from op6_correlation_matrix import run_op6


data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"

# OP1: Lettura
risultato = run_load_and_convert_to_csv(str(data_path))
mio_dataframe = risultato.df_output

# OP3: Split / feature engineering
risultato_split = run_split_dataset(mio_dataframe)
mio_dataframe_modificato = risultato_split.df_output

# OP4: Imputazione valori nulli solo per gruppi di cardinalità 1
risultato_imputazione = run_handle_null_values(mio_dataframe_modificato)
mio_dataframe_imputato = risultato_imputazione.df_output

# OP5: Costi + rimozione nomi
risultato_op5 = run_op5(mio_dataframe_imputato)
mio_dataframe_finale = risultato_op5.df_output

TotalSpending è fondamentale → spesso è una delle feature più predittive nei dataset tipo Spaceship Titanic.
Se lasci le singole colonne + il totale → il modello può imparare meglio (non rimuoverle subito).
Se invece vuoi ridurre dimensionalità → puoi eliminare le singole colonne dopo aver creato il totale.


# --- NUOVO STEP OP6 ---
risultato_op6 = run_op6(
    mio_dataframe_finale,
    config={"output_dir": "outputs/op6"}
)

print("OP6 completato.")

if risultato_op6["corr_with_target"] is not None:
    print("\nTop correlazioni con Transported:")
    print(risultato_op6["corr_with_target"].head(10))
# ----------------------


# Salvataggio dizionari
dict_output_path = data_path.with_name("probability_dictionaries.json")
with open(dict_output_path, "w", encoding="utf-8") as f:
    json.dump(risultato_imputazione.probability_dictionaries, f, indent=4, ensure_ascii=False)

print(f"Dizionari salvati in: {dict_output_path}")

# OP2: Valutazione finale
valutazione = run_evaluation(mio_dataframe_finale)
print("Valutazione completata.")


# CORRETTO (usa mio_dataframe_finale)
output_path = data_path.with_name("train_processed.csv")
mio_dataframe_finale.to_csv(output_path, index=False)

print(f"Dataset finale salvato in: {output_path}")
"""


#nuova pipeline 
from pathlib import Path
import json

from op1_read_file import run_load_and_convert_to_csv
from op2_data_evaluation import run_evaluation
from op3_split_dataset import run_split_dataset
from op4_handler_nullvalue import run_handle_null_values
from op5_sumcosts_names import run_op5
from op6_correlation_matrix import run_op6

def main():
    print("--- INIZIO PIPELINE ---")
    
    # Percorso al dataset originale
    data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"

    # OP1: Lettura
    print("\n[Esecuzione OP1] Lettura dataset...")
    risultato = run_load_and_convert_to_csv(str(data_path))
    mio_dataframe = risultato.df_output

    # OP3: Split / feature engineering (es. estrazione Deck, Num, Side da Cabin)
    print("\n[Esecuzione OP3] Split dataset e feature engineering...")
    risultato_split = run_split_dataset(mio_dataframe)
    mio_dataframe_modificato = risultato_split.df_output

    # OP4: Pulizia iniziale (CryoSleep) e Imputazione valori nulli (gruppi multipli e singoli)
    print("\n[Esecuzione OP4] Gestione e imputazione valori nulli...")
    risultato_imputazione = run_handle_null_values(mio_dataframe_modificato)
    mio_dataframe_imputato = risultato_imputazione.df_output

    # Salvataggio dizionari delle probabilità (basati su OP4)
    dict_output_path = data_path.with_name("probability_dictionaries.json")
    with open(dict_output_path, "w", encoding="utf-8") as f:
        json.dump(risultato_imputazione.probability_dictionaries, f, indent=4, ensure_ascii=False)
    print(f"Dizionari delle probabilità salvati in: {dict_output_path}")

    # OP5: Somma dei costi (TotalSpending) + rimozione nomi
    # Nota: TotalSpending è fondamentale e molto predittivo. 
    # Manteniamo per ora le singole colonne e creiamo il totale per ridurre/aumentare dimensionalità a scelta.
    print("\n[Esecuzione OP5] Calcolo costi totali e gestione nomi...")
    risultato_op5 = run_op5(mio_dataframe_imputato)
    mio_dataframe_finale = risultato_op5.df_output

    # OP6: Matrice di correlazione
    print("\n[Esecuzione OP6] Calcolo matrice di correlazione...")
    risultato_op6 = run_op6(
        mio_dataframe_finale,
        config={"output_dir": "outputs/op6"}
    )
    print("OP6 completato.")

    if risultato_op6["corr_with_target"] is not None:
        print("\nTop correlazioni con Transported:")
        print(risultato_op6["corr_with_target"].head(10))

    # OP2: Valutazione finale (sui dati completi, puliti e arricchiti)
    print("\n[Esecuzione OP2] Valutazione finale dataset...")
    valutazione = run_evaluation(mio_dataframe_finale)
    print("Valutazione completata.")

    # Salvataggio del dataset finale elaborato
    output_path = data_path.with_name("train_processed.csv")
    mio_dataframe_finale.to_csv(output_path, index=False)
    print(f"\nn[FINE] Dataset finale salvato in: {output_path}")

if __name__ == "__main__":
    main()

















