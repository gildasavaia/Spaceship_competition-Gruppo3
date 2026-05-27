import os
import sys
import subprocess
import json
import pandas as pd
from pathlib import Path


def print_header(title):
    print("\n" + "=" * 65)
    print(f" {title.upper()} ".center(65, '='))
    print("=" * 65 + "\n")


def run_script(script_path, working_dir):
    """Esegue uno script Python come sottoprocesso nel suo ambiente isolato."""
    if not script_path.exists():
        print(f"❌ Errore critico: Il file {script_path.name} non è stato trovato!")
        return False
    try:
        subprocess.run([sys.executable, str(script_path)], cwd=str(working_dir), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n Il processo del modello è terminato con un errore (Codice: {e.returncode}).")
        return False
    except KeyboardInterrupt:
        print("\n Esecuzione interrotta manualmente dall'utente.")
        return False


def salva_e_esci(session_metrics, outputs_dir):
    """Funzione dedicata al salvataggio della Leaderboard prima di chiudere tutto."""
    print_header("USCITA DAL SISTEMA. GENERAZIONE REPORT...")
    if session_metrics:
        # Creiamo il DataFrame dalla lista delle metriche salvate
        df_leaderboard = pd.DataFrame(session_metrics)

        # Riordiniamo le colonne per chiarezza
        cols = ['Model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cols = [c for c in cols if c in df_leaderboard.columns]
        df_leaderboard = df_leaderboard[cols]

        # Salviamo il file riassuntivo
        report_path = outputs_dir / "leaderboard_sessione_corrente.csv"
        df_leaderboard.to_csv(report_path, index=False)

        print(f"🏆 Classifica Modelli Salvata con Successo!\n Trovi il file in: {report_path}")
        print("\nEcco l'anteprima dei tuoi risultati:")
        print(df_leaderboard.to_string(index=False))
    else:
        print("Nessuna nuova metrica generata in questa sessione.")
    sys.exit(0)  # Chiude brutalmente e in modo sicuro il programma


def main():
    print_header("SPACESHIP TITANIC - MISSION CONTROL ROOM")
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # === LA NOSTRA LISTA GLOBALE DELLE METRICHE ===
    # È posizionata fuori dai cicli, così se torni al preprocessing NON la perdi!
    session_metrics = []

    # =========================================================
    # CICLO ESTERNO: GESTISCE IL RITORNO AL PREPROCESSING
    # =========================================================
    while True:
        print_header("Fase 1: Preprocessing dei Dati")
        print("Hai bisogno di generare i file CSV per l'addestramento?")
        print("Premi 'S' per avviare pipeline.py, oppure 'N' se i dati sono già pronti.")
        scelta_pre = input("Scelta (S/N): ").strip().lower()

        if scelta_pre == 's':
            prep_script = base_dir / "preprocessing" / "pipeline.py"
            prep_dir = base_dir / "preprocessing"
            print("\n Avvio della pipeline di preprocessing in corso...")
            success = run_script(prep_script, prep_dir)
            if not success:
                print(" Preprocessing fallito o interrotto.")
                continue  # Rifa la domanda di preprocessing se qualcosa va storto
            print("\n Preprocessing completato con successo!")
        else:
            print("\n Preprocessing saltato. Uso i dati CSV esistenti.")

        # =========================================================
        # CICLO INTERNO: GESTISCE IL MENU DEI MODELLI
        # =========================================================
        while True:
            print_header("Fase 2: Selezione Modello Machine Learning")
            print("Scegli quale algoritmo vuoi addestrare o testare:")
            print("  [1] LightGBM                  (Il più veloce)")
            print("  [2] Random Forest             (Robusto, basato su Alberi)")
            print("  [3] Support Vector Classifier (Matematico spaziale)")
            print("  [4] XGBoost                   (Kaggle Standard)")
            print("  [5] CatBoost                  (Re delle variabili categoriche)")
            print("  [6] Rete Neurale (PyTorch)    (Deep Learning)")
            print("-" * 65)
            print("  [7] Torna indietro alla Fase 1 (Preprocessing)")
            print("  [0] Esci e Salva Report della Sessione")

            scelta_modello = input("\n Inserisci il numero (0-7): ").strip()

            script_to_run = None
            work_dir = None

            if scelta_modello == "1":
                script_to_run = base_dir / "LightGBM" / "LightGBM_model_main.py"
                work_dir = base_dir / "LightGBM"
            elif scelta_modello == "2":
                script_to_run = base_dir / "Random Forest Classifier" / "Random_Forest_Classifier_model_main.py"
                work_dir = base_dir / "Random Forest Classifier"
            elif scelta_modello == "3":
                script_to_run = base_dir / "Support_Vector_Classifier" / "Support_Vector_Classifier_model_main.py"
                work_dir = base_dir / "Support_Vector_Classifier"
            elif scelta_modello == "4":
                script_to_run = base_dir / "XGBoost" / "Main_XGBoost.py"
                work_dir = base_dir / "XGBoost"
            elif scelta_modello == "5":
                script_to_run = base_dir / "CATBoost" / "Main_catboost.py"
                work_dir = base_dir / "CATBoost"
            elif scelta_modello == "6":
                script_to_run = base_dir / "NN_Pytorch" / "main_NN_pytch.py"
                work_dir = base_dir / "Rete_Neurale"
            elif scelta_modello == "7":
                # Rompe il ciclo interno (Fase 2) e ricomincia dal ciclo esterno (Fase 1)
                break
            elif scelta_modello == "0":
                # Salva i risultati ed esce
                salva_e_esci(session_metrics, outputs_dir)
            else:
                print(" Scelta non valida. Assicurati di inserire un numero tra 0 e 7.")
                continue

            # Esecuzione del modello selezionato
            if script_to_run:
                print(f"\n Collegamento al modulo... Avvio in corso...")
                run_script(script_to_run, work_dir)

                # --- IL TRUCCO: RECUPERO DELLE METRICHE ---
                temp_file = outputs_dir / "temp_metrics.json"
                if temp_file.exists():
                    with open(temp_file, "r") as f:
                        new_metrics = json.load(f)
                    session_metrics.append(new_metrics)
                    temp_file.unlink()  # Cancelliamo il file per la prossima run
                    print(f"\n [Orchestratore] Acquisite metriche per: {new_metrics['Model']}")

                # --- MENU POST-ADDESTRAMENTO ---
                print("\n" + "-" * 50)
                scelta_dopo = input(
                    "Premi INVIO per il Menu Modelli, 'P' per il Preprocessing, o '0' per Salvare e Uscire: ").strip().lower()

                if scelta_dopo == 'p':
                    break  # Rompe il ciclo modelli e torna alla Fase 1
                elif scelta_dopo == '0':
                    salva_e_esci(session_metrics, outputs_dir)  # Salva e chiude
                # Se preme solo INVIO, il ciclo while True dei modelli riparte da capo!


if __name__ == "__main__":
    main()