import pandas as pd
import os

# Percorso del dataset
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_processed.csv')

# Carica il dataset
df = pd.read_csv(dataset_path)

# Conta i null per ogni colonna
null_counts = df.isnull().sum()
null_percentages = (df.isnull().sum() / len(df)) * 100

# Crea un DataFrame con i risultati
null_report = pd.DataFrame({
    'Colonna': null_counts.index,
    'Null Count': null_counts.values,
    'Null Percentage (%)': null_percentages.values,
    'Non-Null Count': len(df) - null_counts.values
})

# Ordina per numero di null decrescente
null_report = null_report.sort_values('Null Count', ascending=False)

# Salva il report
output_path = os.path.join(os.path.dirname(__file__), '..', 'report_missing', '00_null_counts_per_colonna.csv')
null_report.to_csv(output_path, index=False)

# Mostra il report
print("\n" + "="*80)
print("REPORT VALORI NULL PER COLONNA")
print("="*80)
print(null_report.to_string(index=False))

# Calcola il totale
total_null = null_counts.sum()
print(f"\nTotale valori null nel dataset: {total_null}")
print(f"Percentuale globale: {(total_null / (len(df) * len(df.columns)) * 100):.2f}%")
print(f"\nReport salvato in: {output_path}")
print("="*80 + "\n")
