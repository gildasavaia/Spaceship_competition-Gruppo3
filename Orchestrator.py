import sys
import subprocess
import json
import pandas as pd
from pathlib import Path

""" Questo script è il cervello dell'intero progetto, esso gestisce l'esecuzione isolata dei vari moduli (Preprocessing
e Modelli predittivi) fornendo un'interfaccia utente interattiva a riga di comando.

1) Utilizza il modulo 'subprocess' per far girare ogni algoritmo nel proprio ambiente di memoria. Se un modello va in crash,
   l'Orchestrator non fa la stessa fine.
2) Dialoga con la classe MetricsEvaluator raccogliendo i risultati dei modelli tramite un file JSON temporaneo, creando una
   cronologia della sessione.
3. Genera una Leaderboard, ovvero alla fine consolida i risultati in un report Pandas (.csv) per permettere un confronto
   immediato tra i vari modelli"""


def print_header(title):
    """Utility grafica per stampare titoli centrati e ben visibili nel terminale."""

    print("\n" + "-" * 65)
    print(f" {title.upper()} ".center(65, '-'))
    print("-" * 65 + "\n")


def run_script(script_path, working_dir):
    """ Esegue uno script Python come sottoprocesso nel suo ambiente isolato. Questo pattern assicura che le variabili e
    le librerie caricate da un modello vengano dimenticate non appena il modello ha finito, prevenendo i memory leak."""

    if not script_path.exists():
        print(f"Errore critico: Il file {script_path.name} non è stato trovato!")
        return False
    try:
        # sys.executable richiama esattamente l'interprete Python che stiamo usando ora.
        # cwd imposta la cartella di partenza corretta per lo script.
        subprocess.run([sys.executable, str(script_path)], cwd=str(working_dir), check=True)
        return True
    # Cattura gli errori di codice del sottomodulo.
    except subprocess.CalledProcessError as e:
        print(f"\n Il processo del modello è terminato con un errore (Codice: {e.returncode}).")
        return False
    # Cattura la chiusura manuale dell'utente.
    except KeyboardInterrupt:
        print("\n Esecuzione interrotta manualmente dall'utente.")
        return False


def salva_e_esci(session_metrics, outputs_dir):
    """ Funzione dedicata al salvataggio della Leaderboard prima di chiudere tutto. Prende la cronologia dei risultati
    in memoria e la esporta in CSV."""

    print_header("USCITA DAL SISTEMA. GENERAZIONE REPORT...")
    if session_metrics:
        # Creiamo un DataFrame Pandas partendo dalla lista di dizionari JSON raccolti.
        df_leaderboard = pd.DataFrame(session_metrics)

        # Riordiniamo le colonne mettendo prima l'Accuracy (la metrica di Kaggle) e poi le altre.
        cols = ['Model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        # Filtriamo solo le colonne che esistono effettivamente per evitare errori.
        cols = [c for c in cols if c in df_leaderboard.columns]
        df_leaderboard = df_leaderboard[cols]

        # Salviamo il file riassuntivo finale.
        report_path = outputs_dir / "leaderboard_sessione_corrente.csv"
        df_leaderboard.to_csv(report_path, index=False)

        print(f"Classifica Modelli Salvata con Successo!\n Trovi il file in: {report_path}")
        print("\nEcco l'anteprima dei tuoi risultati:")
        print(df_leaderboard.to_string(index=False)) # index=False rende la stampa più pulita.
    else:
        print("Nessuna nuova metrica generata in questa sessione.")
    sys.exit(0)  # Chiude brutalmente e in modo sicuro il programma


def main():
    print_header("SPACESHIP TITANIC - MISSION CONTROL ROOM")

    # Risoluzione dinamica dei percorsi: capisce automaticamente in quale cartella del PC si trova.
    base_dir = Path(__file__).resolve().parent
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Lista globale delle metriche. È posizionata fuori dai cicl in modo tale che l'utente possa fare avanti e indietro
    # tra i menù senza perdere la memoria dei modelli già addestrati.
    session_metrics = []

    # CICLO ESTERNO (Fase 1): Gestisce il PreProcessing ed il loop principale.
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
                continue # Se fallisce, l'istruzione 'continue' riavvia il ciclo rifacendo la domanda.
            print("\n Preprocessing completato con successo!")
        else:
            print("\n Preprocessing saltato. Uso i dati CSV esistenti.")

        # CICLO INTERNO (Fase 2): Gestisce il menù dei modelli di Machine Learning.
        while True:
            print_header("Fase 2: Selezione Modello Machine Learning")
            print("Scegli quale algoritmo vuoi addestrare o testare:")
            print("  [1] LightGBM (Tree)                 (Il più veloce)")
            print("  [2] Random Forest (Tree)            (Robusto, basato su Alberi)")
            print("  [3] Support Vector Classifier (nn) (Matematico spaziale)")
            print("  [4] XGBoost (Tree)                   (Kaggle Standard)")
            print("  [5] CatBoost (Tree)                  (Re delle variabili categoriche)")
            print("  [6] Rete Neurale (nn)              (PyTorch)")
            print("-" * 65)
            print("  [7] Torna indietro alla Fase 1 (Preprocessing)")
            print("  [0] Esci e Salva Report della Sessione")

            scelta_modello = input("\n Inserisci il numero (0-7): ").strip()

            script_to_run = None
            work_dir = None

            # Assegnazione dinamica del percorso dello script in base alla scelta.
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
                work_dir = base_dir / "NN_Pytorch"
            elif scelta_modello == "7":
                break # L'istruzione 'break' distrugge il ciclo interno (Fase 2) tornando nuovamente nel ciclo esterno (Fase 1).
            elif scelta_modello == "0":
                # Salva i risultati ed esce.
                salva_e_esci(session_metrics, outputs_dir)
            else:
                print(" Scelta non valida. Assicurati di inserire un numero tra 0 e 7.")
                continue

            # Esecuzione del modello selezionato.
            if script_to_run:
                print(f"\n Collegamento al modulo... Avvio in corso...")
                run_script(script_to_run, work_dir)

                '''Recupero delle Metriche. Siccome i modelli girano in un processo isolato, le loro variabili muoiono con
                loro. Per recuperare i punteggi, i modelli scrivono un file JSON. Qui l'Orchestrator lo cerca, lo legge,
                salva i dati nella lista globale 'session_metrics' e poi distrugge il file per fare pulizia.'''

                temp_file = outputs_dir / "temp_metrics.json"
                if temp_file.exists():
                    with open(temp_file, "r") as f:
                        new_metrics = json.load(f)
                    session_metrics.append(new_metrics) # Aggiunge il dizionario letto alla nostra cronologia.
                    temp_file.unlink() # Cancella il file per evitare che modelli successivi leggano dati vecchi.
                    print(f"\n [Orchestratore] Acquisite metriche per: {new_metrics['Model']}")

                # Terminata un'esecuzione, chiediamo all'utente cosa fare dopo.
                print("\n" + "-" * 50)
                scelta_dopo = input(
                    "Premi INVIO per il Menu Modelli, 'P' per il Preprocessing, o '0' per Salvare e Uscire: ").strip().lower()

                if scelta_dopo == 'p':
                    break # Rompe il ciclo modelli e torna alla Fase 1.
                elif scelta_dopo == '0':
                    salva_e_esci(session_metrics, outputs_dir) # Salva e chiude.
                # Se l'utente preme solo INVIO, il codice scorre ed il 'while True' interno riparte in automatico mostrando di nuovo la lista dei modelli.


if __name__ == "__main__":
    main()