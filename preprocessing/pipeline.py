# Importa la funzione dal file op1_read_file.py
from op1_read_file import run_load_and_convert_to_csv
from pathlib import Path

# Percorso al dataset `data/train.csv` relativo alla radice del progetto
data_path = Path(__file__).resolve().parents[1] / "data" / "train.csv"

# Richiama la funzione passandole il percorso del file
risultato = run_load_and_convert_to_csv(str(data_path))

# Accedi al DataFrame
mio_dataframe = risultato.df_output

print(mio_dataframe)