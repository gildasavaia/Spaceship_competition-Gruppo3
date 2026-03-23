# Importa le funzioni dai file op1_read_file e op2_data_evaluation
from op1_read_file import run_load_and_convert_to_csv
from op2_data_evaluation import run_evaluation
from pathlib import Path

# Percorso al dataset `data/train.csv` relativo alla radice del progetto
data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"

# Richiama la funzione passandole il percorso del file
risultato = run_load_and_convert_to_csv(str(data_path))

# Accedi al DataFrame
mio_dataframe = risultato.df_output

# Esegui l'analisi esplorativa con op2
valutazione = run_evaluation(mio_dataframe)

print("Valutazione completata.")