# Spaceship Titanic Challenge

Il contest "Spaceship Titanic" ci sfida a risolvere un mistero cosmico. Sfruttando le nostre competenze di data science, puntiamo a prevedere il destino dei passeggeri che hanno incontrato un'anomalia spazio-temporale durante il loro viaggio.

[Challenge Link](https://[miosito.com](https://www.kaggle.com/competitions/spaceship-titanic))

# Project Files
I file chiavi  relativi a questo progetto sono i seguenti: 
- [train.csv](https://github.com/gildasavaia/Spaceship_competition-Gruppo3/blob/main/data/train.csv) : Set di dati di addestramento contenente informazioni personali ed etichette di riferimento.
- [test.csv](https://github.com/gildasavaia/Spaceship_competition-Gruppo3/blob/main/data/test.csv) : Set di dati per la predizione
- [sample_submission.csv](https://github.com/gildasavaia/Spaceship_competition-Gruppo3/blob/main/data/sample_submission.csv) : Formato di esempio per la sottomissione dei risultati.

# Struttura del progetto
Il progetto Titanic Spaceship si articola in tre macro moduli principali, che rappresentano le fasi fondamentali del workflow:

- [Pre-processing]()
- [Development]()
- [Evaluation]()

## Librerie necessarie 
 **Pandas:** Manipolazione dei dati, caricamento file e operazioni di preprocessing avanzate.
 
 **NumPy:** Calcoli numerici, trasformazioni logaritmiche e gestione `NaN`.
 
 **Scikit-Learn:** Core framework per le trasformazioni ML (`train_test_split`, `KFold`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`).
 
 **Seaborn & Matplotlib:** Generazione delle visualizzazioni e delle matrici di correlazione.

## Installazione
Per replicare l'ambiente di sviluppo:

1. **Clona la repository:**
   ```bash
   git clone [https://github.com/gildasavaia/Spaceship_competition-Gruppo3](https://github.com/gildasavaia/Spaceship_competition-Gruppo3)
   cd Spaceship_competition-Gruppo3
   ```
2. **Crea un ambiente virtuale (consigliato):**
  ```bash
  python -m venv venv
  source venv/bin/activate  # Su Linux/Mac
  venv\Scripts\activate     # Su Windows 
  ```

## Pre-processing
Il pre-processing è la fase in cui i dati grezzi vengono ripuliti e trasformati in un formato ottimizzato per l’analisi e il training dei modelli. Poiché la qualità dei dati influisce direttamente sulle performance predittive, questa fase è stata curata nei minimi dettagli.
Nel nostro approccio, la preparazione dei dati non è gestita tramite script frammentati, ma è completamente orchestrata da una pipeline strutturata e centralizzata tramite codice di controllo centrale.
Questa pipeline coordina 9 moduli distinti, garantendo modularità, scalabilità e, soprattutto, la totale prevenzione del Data Leakage applicando calcoli statistici solo sui dati di Train.

### Pipeline di Pre-processing
Per avviare la pipeline di pre-processing, eseguire il seguente comando da terminale:

*python pipeline.py*

Durante l’esecuzione, un'interfaccia interattiva da terminale richiederà i parametri fondamentali per adattare il preprocessing alle esigenze del modello.
#### Metodo di split:
1 → *Hold-out*: l metodo più veloce. Divide il dataset in due blocchi (es. 80% Train, 20% Test). Ottimo per test rapidi o se il dataset è molto grande.

2 → *K-Fold*: Divide i dati in $K$ "fette" (folds) e addestra il modello $K$ volte, garantendo che ogni singola riga venga usata per il test esattamente una volta. È lo standard aureo per evitare overfitting e valutare le vere capacità del modello su Kaggle.
#### Tipo di modello:
1 → *Alberi decisionali*: Selezionando questa opzione, la pipeline applicherà l'Ordinal Encoding (trasforma le categorie in numeri progressivi) ed eviterà di scalare i dati, poiché i modelli tree-based non ne hanno bisogno e lavorano meglio con i dati originali.

2 → *Reti neurali / modelli lineari*: Questa opzione attiva tecniche specifiche per i modelli basati sui pesi: applicherà la Standardizzazione (per portare tutte le feature sulla stessa scala), trasformazioni logaritmiche sui costi, e userà il One-Hot Encoding rimuovendo le colonne collineari per prevenire la Dummy Variable Trap.

#### Gestione Automatica degli Output
La pipeline è progettata per non "sporcare" la root del progetto. Genera e organizza automaticamente i file di output nelle seguenti directory:
- data/preprocessed_folds/: Contiene i dataset pronti per il training (.csv). I nomi dei file vengono generati dinamicamente in base alle scelte dell'utente (es. holdout_tree_train.csv oppure kfold_1_nn_test.csv).
- data/probability_dictionaries/: Salva i dizionari in formato .json calcolati durante la fase di imputazione dei nulli (OP4). Memorizzare queste probabilità è fondamentale per mantenere la coerenza in fase di deployment su nuovi dati.
- outputs/op6/: Salva l'immagine correlation_matrix.png generata dall'analisi post-processing.


### Workflow moduli 
Di seguito il dettaglio tecnico delle 9 operazioni (OP) eseguite in sequenza:

OP1 – Lettura Dati: Il modulo rileva dinamicamente il formato del file di input (.xlsx, .json, .parquet, .csv) e, in caso non sia CSV, lo converte e lo carica automaticamente tramite Pandas.

OP2 – Data Evaluation: Esegue un'Analisi Esplorativa (EDA) automatizzata. Conta i valori nulli, analizza la struttura del dataset e mappa le colonne di tipo Object restituendone i valori unici.

OP3 – Feature Engineering (Split Colonne): Estrae variabili nascoste suddividendo stringhe complesse.
- Estrae Group e calcola GroupSize dalla colonna PassengerId.
- Scompone Cabin in Deck, Num e Side.
- Divide Name in Names e Surnames.

OP4 – Gestione Valori Nulli: Il cuore della pipeline. Applica imputazioni intelligenti e regole di business:
- Regola Logica: Identifica i passeggeri con CryoSleep == True e impone matematicamente a 0 le loro spese di bordo.
- Imputazione Gruppi: Sfrutta la colonna Group per riempire i NaN di chi viaggia in compagnia utilizzando la moda del gruppo.
- Imputazione Probabilistica: Per i passeggeri singoli, i NaN vengono riempiti campionando casualmente dai dizionari di probabilità calcolati.
- Zero Data Leakage: Le medie (es. Age) e i dizionari di probabilità vengono fittati rigorosamente sul Train set e poi applicati al Test set.

OP5 – Feature Engineering (Spending & Pulizia): Aggrega i costi (RoomService, FoodCourt, ecc.) creando la macro-feature TotalSpending. Successivamente, trasforma le colonne di costo originali in variabili binarie (1 se ha speso, 0 altrimenti).

OP6 – Correlation Matrix: Calcola e salva in output la matrice di correlazione estesa (correlation_matrix.png) includendo anche le feature intermedie, utile per individuare le dinamiche tra le variabili prima degli scarti finali.

OP7 – Dataset Splitting: Divide il dataset in variabili predittive e target. Offre due vie gestite tramite interazione utente.
- Hold-out: Classico split (es. 80/20) tramite train_test_split.
- K-Fold CV: Divide i dati in un numero specificato di folds mescolati preventivamente (shuffle).

OP8 – Scaling: Ramo di ottimizzazione numerica diviso per tipologia di modello target.
- Ramo Modelli ad Albero: Nessuna scalatura.
- Ramo Modelli Lineari / Reti Neurali: Applica trasformazione logaritmica (log1p) su TotalSpending per ridurre l'asimmetria, e utilizza lo StandardScaler sulle altre features numeriche.

OP9 – Encoding: Ramo di ottimizzazione categorica per evitare la Dummy Variable Trap:
- Ramo Modelli ad Albero: Applica OrdinalEncoder.
- Ramo Modelli Lineari / Reti Neurali: Applica OneHotEncoder ed esegue il drop preventivo di colonne altamente collineari come CryoSleep_False, VIP_False e Side_S.




### Output:
dataset ottimizzati per ciascun tipo di modello



## Development 

## Evaluation

