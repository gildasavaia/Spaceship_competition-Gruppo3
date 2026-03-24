# Importa le funzioni dai file op1_read_file, op2_data_evaluation e dal nuovo op3_split_passenger
from op1_read_file import run_load_and_convert_to_csv
from op2_data_evaluation import run_evaluation
from op3_split_dataset import run_split_dataset  
from pathlib import Path

# Percorso al dataset `data/train.csv` relativo alla radice del progetto
data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"

# Richiama la funzione passandole il percorso del file
risultato = run_load_and_convert_to_csv(str(data_path))

# Accedi al DataFrame
mio_dataframe = risultato.df_output

# --- NUOVO STEP ---
# Dividi PassengerId in Group e GroupSize
risultato_split = run_split_dataset(mio_dataframe)
mio_dataframe_modificato = risultato_split.df_output
# ------------------

# Esegui l'analisi esplorativa con op2 passando il DataFrame aggiornato
valutazione = run_evaluation(mio_dataframe_modificato)

print("Valutazione completata.")

# Salva il DataFrame modificato su file nella cartella data
output_path = data_path.with_name("train_processed.csv")
mio_dataframe_modificato.to_csv(output_path, index=False)
print(f"Dataset modificato salvato in: {output_path}")