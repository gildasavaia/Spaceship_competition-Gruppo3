# Spaceship Titanic — Documentazione Completa del Progetto

Competizione Kaggle **Spaceship Titanic**: predire quali passeggeri sono stati teletrasportati in una dimensione alternativa durante la collisione con un'anomalia spaziotemporale.

---

## Indice

1. [Struttura del Progetto](#1-struttura-del-progetto)
2. [Avvio Rapido](#2-avvio-rapido)
3. [Fase 1 — Pre-Processing](#3-fase-1--pre-processing)
   - [Panoramica della Pipeline](#panoramica-della-pipeline)
   - [OP1 — Lettura File](#op1--op1_read_filepy)
   - [OP2 — Analisi Esplorativa](#op2--op2_data_evaluationpy)
   - [OP3 — Feature Engineering](#op3--op3_split_datasetpy)
   - [OP4 — Gestione Valori Nulli](#op4--op4_handler_nullvaluepy)
   - [OP5 — Aggregazione Costi](#op5--op5_sumcosts_namespy)
   - [OP6 — Matrice di Correlazione](#op6--op6_correlation_matrixpy)
   - [OP7 — Divisione del Dataset](#op7--divisione-del-dataset)
   - [OP8 — Scaling](#op8--op8_scalingpy)
   - [OP9 — Encoding](#op9--op9_encodingpy)
   - [relations.py — Analisi Regole di Dominio](#relationspy--analisi-regole-di-dominio)
   - [pipeline.py — Pipeline Principale](#pipelinepy--pipeline-principale)
   - [Output del Pre-Processing](#output-del-pre-processing)
4. [Fase 2 — Development & Evaluation](#4-fase-2--development--evaluation)
   - [Orchestrator](#orchestrator--orchestratorpy)
   - [MetricsEvaluator](#metricsevaluator--evaluationmetrics_calculatorpy)
   - [Modulo di Valutazione](#modulo-di-valutazione--evaluation_py)
   - [Modalità di Addestramento](#modalità-di-addestramento)
   - [Modelli Disponibili](#modelli-disponibili)
   - [Output del Development](#output-del-development)
5. [Test Unitari](#5-test-unitari)
   - [Test Preprocessing](#test-preprocessing-test1--test5)
   - [Test Modelli](#test-modelli)
6. [Decisioni Architetturali](#6-decisioni-architetturali)
7. [Requisiti](#7-requisiti)

---

## 2. Avvio Rapido

```bash
# Avvia l'intero sistema (preprocessing + modelli)
python Orchestrator.py
```

Il sistema guida interattivamente attraverso tutte le fasi. Se i CSV del preprocessing sono già pronti, è possibile saltare direttamente alla selezione del modello.

Per eseguire solo il preprocessing in modo indipendente:
```bash
python preprocessing/pipeline.py
```

---

## 3. Fase 1 — Pre-Processing

### Panoramica della Pipeline

La pipeline è composta da operazioni modulari (`OP`). Ogni modulo riceve un DataFrame in input e restituisce un output strutturato tramite `NamedTuple` o `dataclass`.

Dopo OP3, il sistema pone **due domande interattive** che determinano il percorso di esecuzione:

```
train.csv / test.csv
       │
       ▼
  OP1 — Lettura file (multi-formato)
       │
       ▼
  OP3 — Feature Engineering (split PassengerId, Cabin, Name)
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  SCELTA 1: Metodo di split              │
  │  1 → Hold-out                           │
  │  2 → K-Fold Cross Validation            │
  │  3 → [Fantasma] Dataset intero          │
  └─────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  SCELTA 2: Tipo di modello              │
  │  1 → Alberi (XGBoost, CatBoost, ecc.)  │
  │  2 → Reti Neurali / Modelli Lineari     │
  └─────────────────────────────────────────┘
       │
       ├── Hold-out:   OP4 → OP5 → OP8 → OP9  (su train e test separati)
       ├── K-Fold:     OP4 → OP5 → OP8 → OP9  (per ogni fold)
       └── Fantasma:   OP4 → OP5 → OP8 → OP9  (intero dataset, per submission)

  In tutti i percorsi, al termine:
  OP2 — EDA / Valutazione  ·  OP6 — Matrice di Correlazione
```

---

### OP1 — `op1_read_file.py`
**Caricamento e conversione del dataset**

Legge il file di input e lo converte in CSV se necessario. Supporta i formati:

| Formato | Estensione |
|---|---|
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| JSON | `.json` |
| Parquet | `.parquet` |

```python
risultato = run_load_and_convert_to_csv("data/train.csv")
df = risultato.df_output
```

---

### OP2 — `op2_data_evaluation.py`
**Analisi esplorativa (EDA)**

Esegue un'analisi di base sul DataFrame: conteggio dei valori nulli per colonna, analisi delle colonne categoriche (`object`), dimensioni e tipi. Restituisce il DataFrame originale invariato (step di sola ispezione).

---

### OP3 — `op3_split_dataset.py`
**Feature Engineering — Split delle colonne composte**

| Colonna originale | Feature estratte | Descrizione |
|---|---|---|
| `PassengerId` | `Group`, `GroupSize` | Identificatore del gruppo e numero di membri |
| `Cabin` | `Deck`, `Num`, `Side` | Ponte, numero cabina, lato (P/S) |
| `Name` | `Names`, `Surnames` | Nome proprio e cognome |

Le colonne originali vengono rimosse. Nel test set, `PassengerId` viene mantenuto per la submission finale (`i_train=0`).

```python
risultato = run_split_dataset(df, i_train=1)  # i_train=0 per il test set
```

---

### OP4 — `op4_handler_nullvalue.py`
**Gestione dei valori mancanti**

Il modulo più complesso della pipeline. Combina **regole logiche di dominio** con **strategie statistiche** di imputazione.

#### Regole di dominio

| Regola | Dettaglio |
|---|---|
| `CryoSleep = True` → spese = 0 | I passeggeri in criosonno non possono spendere |
| `Age < 13` → spese = 0 | I bambini non hanno crediti |
| `Age < 18` → `VIP = False` | I minorenni non possono essere VIP |
| `Deck ∈ {G, T}` → `VIP = False` | I ponti economici non ospitano passeggeri VIP |
| `HomePlanet = Earth` → `VIP = False` | I terrestri non risultano mai VIP |
| `Deck ∈ {A, B, C, T}` → `HomePlanet = Europa` | Regola empirica da analisi del dataset |
| `Deck = G` → `HomePlanet = Earth` | Regola empirica da analisi del dataset |
| Spesa > 0 → `CryoSleep = False` | Inferenza inversa dalla spesa |
| `Deck = T` + CryoSleep mancante → `CryoSleep = False` | Regola specifica per il ponte T |

#### Strategie di imputazione

Per le feature categoriche (`HomePlanet`, `Destination`, `CryoSleep`, `VIP`, `Deck`, `Num`, `Side`, `Surnames`):

1. **Moda del gruppo** — per passeggeri con `GroupSize > 1`, si usa il valore più frequente nel gruppo.
2. **Imputazione probabilistica** — per passeggeri solitari, si campiona dalla distribuzione globale del train set.

Per le feature numeriche:
- Costi (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`): NaN → `0`
- `Age`: NaN → media del train (passata come parametro al test per evitare leakage)

> I dizionari di probabilità vengono calcolati **solo sul train** e passati come parametro al test.

---

### OP5 — `op5_sumcosts_names.py`
**Aggregazione costi e pulizia colonne**

- Crea `TotalSpending` = somma delle cinque colonne di spesa (calcolata prima di binarizzazioni).
- Rimuove `Names` (nome proprio non predittivo).
- `Surnames` viene mantenuta fino a OP9.

> `TotalSpending` è storicamente una delle feature più predittive in questo dataset.

---

### OP6 — `op6_correlation_matrix.py`
**Matrice di correlazione**

Esegue OHE completo (`drop_first=False`) e calcola la matrice di correlazione. Salva il grafico in `outputs/op6/correlation_matrix.png` e restituisce la correlazione ordinata con il target `Transported`.

> La Dummy Variable Trap viene evitata in OP9 (non qui), dove vengono rimosse le colonne ridondanti (`CryoSleep_False`, `VIP_False`, `Side_S`).

---

### OP7 — Divisione del Dataset

| Opzione | File | Comportamento |
|---|---|---|
| **1 — Hold-out** | `op7_holdout_evaluator.py` | Split classico train/test, proporzione scelta dall'utente |
| **2 — K-Fold** | `op7_kfold_evaluator.py` | K-Fold non stratificato con shuffle, numero di fold scelto dall'utente |
| **3 — Fantasma** | *(non mostrata nel menu)* | Intero dataset senza split, per la submission finale Kaggle |

> La Versione Fantasma si attiva digitando `3` quando viene chiesta la scelta, ma non appare nel menu. Usarla solo dopo aver validato il modello con Holdout o K-Fold.

---

### OP8 — `op8_scaling.py`
**Normalizzazione e standardizzazione**

Produce **due versioni parallele** del dataset:

| Versione | Destinazione | Trasformazioni |
|---|---|---|
| `df_tree` | Modelli ad albero (XGBoost, CatBoost, RF) | Nessuna |
| `df_nn` | Reti neurali e modelli lineari | `log1p` sulle colonne di spesa + `StandardScaler` |

Colonne trasformate con `log1p`: `TotalSpending`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`.

> Lo scaler viene fittato **solo sul train** e applicato al test.

---

### OP9 — `op9_encoding.py`
**Encoding delle variabili categoriche**

| Versione | Encoder | Note |
|---|---|---|
| `df_tree` | Nessuno (passthrough) | XGBoost e CatBoost gestiscono le categoriche nativamente |
| `df_nn` | One-Hot Encoding | Rimozione colonne ridondanti per evitare multicollinearità |

Colonne codificate: `HomePlanet`, `CryoSleep`, `Destination`, `VIP`, `Deck`, `Side`.
Colonne sempre rimosse: `Group`, `Surnames`.
Colonne OHE rimosse (Dummy Variable Trap): `CryoSleep_False`, `VIP_False`, `Side_S`.

---

### `relations.py` — Analisi Regole di Dominio

Script esplorativo indipendente dalla pipeline principale. Ha guidato la definizione delle regole di OP4. Esegue tre analisi:

1. **Regole di esclusione** — combinazioni impossibili tra variabili categoriche (frequenza = 0).
2. **Coesione di gruppo/famiglia** — percentuale di passeggeri dello stesso gruppo che condividono `HomePlanet`, `Destination`, `Cabin`.
3. **Regole finanziarie** — verifica regole logiche su CryoSleep e bambini.

```bash
python preprocessing/relations.py
```

---

### `pipeline.py` — Pipeline Principale

Orchestratore del preprocessing. Dopo OP3 vengono poste due domande interattive:

```
Scegli il metodo di divisione del dataset:
1: Hold-out  |  2: K-Fold  |  [3: Fantasma — non mostrato]

Scegli il tipo di modello per cui preparare i dati:
1: Alberi Decisionali  |  2: Reti Neurali / Modelli Lineari
```

### Output del Pre-Processing

| Modalità | Tipo modello | File CSV generati |
|---|---|---|
| Hold-out | Alberi | `holdout_tree_train.csv`, `holdout_tree_test.csv` |
| Hold-out | Reti Neurali | `holdout_nn_train.csv`, `holdout_nn_test.csv` |
| K-Fold | Alberi | `kfold_N_tree_train.csv`, `kfold_N_tree_test.csv` |
| K-Fold | Reti Neurali | `kfold_N_nn_train.csv`, `kfold_N_nn_test.csv` |
| Fantasma | Alberi | `processed_full_tree.csv`, `processed_full_tree_test.csv` |
| Fantasma | Reti Neurali | `processed_full_nn.csv`, `processed_full_nn_test.csv` |

Tutti i file sono in `data/preprocessed_folds/`. I dizionari di imputazione JSON vengono salvati in `data/probability_dictionaries/`.

---

## 4. Fase 2 — Development & Evaluation

### Orchestrator — `Orchestrator.py`

Punto di ingresso dell'intera fase di sviluppo. Gestisce il flusso con **due cicli annidati**: il ciclo esterno gestisce il ritorno al preprocessing, quello interno la selezione dei modelli.

```
[AVVIO]
   │
   ▼
┌──────────────────────────────────────┐
│  FASE 1: Preprocessing               │
│  S → Esegue pipeline.py             │
│  N → Salta (CSV già pronti)         │
└──────────────────────────────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│  FASE 2: Selezione Modello           │
│  [1] LightGBM                        │
│  [2] Random Forest                   │
│  [3] Support Vector Classifier       │
│  [4] XGBoost                         │
│  [5] CatBoost                        │
│  [6] Rete Neurale (PyTorch)          │
│  [7] Torna al Preprocessing          │
│  [0] Esci e Salva Report             │
└──────────────────────────────────────┘
   │
   ▼ (dopo ogni run)
┌──────────────────────────────────────┐
│  MENU POST-ADDESTRAMENTO             │
│  INVIO → Menu Modelli                │
│  P     → Preprocessing               │
│  0     → Salva Report ed Esci        │
└──────────────────────────────────────┘
```

#### Raccolta delle metriche

Ogni modello scrive le metriche in `outputs/temp_metrics.json` tramite `MetricsEvaluator`. L'Orchestratore lo legge, lo aggiunge alla lista `session_metrics` e lo cancella immediatamente.

```
Modello → export_to_orchestrator() → temp_metrics.json → Orchestratore → session_metrics[]
```

La lista `session_metrics` è definita fuori dai cicli: tornare al preprocessing (opzione `7` o `P`) **non azzera** le metriche già raccolte.

#### Leaderboard

All'uscita (opzione `0`), viene generato `outputs/leaderboard_sessione_corrente.csv` con tutti i modelli della sessione:
Esempio:
| Model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| XGBoost (Holdout) | 0.8134 | 0.8201 | 0.8022 | 0.8110 | 0.8891 |
| CatBoost (K-Fold) | 0.8156 | ... | ... | ... | ... |

---

### MetricsEvaluator — `Evaluation/metrics_calculator.py`

Classe OOP condivisa da tutti i modelli. Centralizza calcolo, visualizzazione e passaggio delle metriche all'Orchestratore.

```python
valutatore = MetricsEvaluator(
    y_true=y_test,
    y_pred=predictions,
    y_probs=probabilities,   # opzionale, per ROC-AUC
    dataset_name="XGBoost (Holdout)"
)
```

| Metodo | Descrizione |
|---|---|
| `calculate_metrics()` | Restituisce dizionario con `Model`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc` |
| `print_report()` | Stampa le metriche in console + chiama `export_to_orchestrator()` automaticamente |
| `plot_visuals()` | Genera matrice di confusione e curva ROC affiancate |
| `export_to_orchestrator()` | Scrive `outputs/temp_metrics.json` per il pickup dell'Orchestratore |

> `export_to_orchestrator()` risale due livelli (`parent.parent`) dalla propria posizione per trovare `outputs/`, funzionando correttamente da qualsiasi sottocartella modello.

---

### Modulo di Valutazione — `Evaluation_*.py`

Ogni modello ha un proprio modulo locale con:

- **`run_full_evaluation()`** — predizione, metriche, grafici (matrice di confusione + curva ROC). `verbose=False` nei fold K-Fold intermedi per non generare N grafici.
- **`print_kfold_summary()`** — medie delle metriche, deviazione standard dell'accuracy, matrice di confusione cumulativa al termine del K-Fold.
- **`plot_loss_curve(model)`** *(solo Rete Neurale)* — andamento della loss durante il training.

---

### Modalità di Addestramento

Ogni modello offre tre modalità selezionabili all'avvio:

| Modalità | Dataset usato | Output |
|---|---|---|
| **1 — Holdout** | `holdout_tree_train/test.csv` | Report metriche + grafici + export Orchestratore |
| **2 — K-Fold** | `kfold_N_tree_train/test.csv` | Sintesi K-Fold + matrice cumulativa + file TOTAL |
| **3 — Full Training** | `processed_full_tree.csv` | File `submission_*.csv` per Kaggle |

In K-Fold, LightGBM, Random Forest e SVC producono anche un **file TOTAL** (`submission_*_kfold_TOTAL.csv`) con tutte le predizioni dei fold unite e ordinate per `PassengerId`.

---

### Modelli Disponibili

#### XGBoost — `XGBoost/`

Gestisce nativamente le variabili categoriche (`enable_categorical=True`). Le colonne `object` vengono convertite in `category` tramite `fix_categorical_dtype()`.

| Iperparametro | Valore |
|---|---|
| `n_estimators` | 3500 |
| `learning_rate` | 0.02 |
| `max_depth` | 5 |
| `subsample` | 0.9 |
| `colsample_bytree` | 0.85 |
| `reg_lambda` / `reg_alpha` | 3 / 0.3 |
| `tree_method` | `hist` |

---

#### CatBoost — `CATBoost/`

Le colonne categoriche vengono passate come `cat_features` a `.fit()`. La predizione usa un oggetto `Pool` per garantire la corretta gestione dei tipi.

| Iperparametro | Valore |
|---|---|
| `iterations` | 1400 |
| `learning_rate` | 0.035 |
| `depth` | 5 |
| `l2_leaf_reg` | 7 |
| `subsample` | 0.8 |
| `rsm` | 0.85 |
| `od_wait` | 80 |

---

#### LightGBM — `LightGBM/`

La classe `LightGBMTrainer` espone due modalità:

| Modalità | Attivazione | Comportamento |
|---|---|---|
| **Fast Mode** | `fast_mode=True` (default) | Parametri predefiniti ottimali, addestramento immediato |
| **Tuning Mode** | `fast_mode=False` | `RandomizedSearchCV` con 10 combinazioni su K-Fold stratificato |

Parametri Fast Mode: `num_leaves=31`, `max_depth=7`, `learning_rate=0.05`, `n_estimators=300`.

**Trucco per la coerenza categorica** — train e test vengono concatenati, convertiti in `category` sull'intero dataset combinato, poi riseparati. Questo garantisce che nessuna categoria manchi nel test set.

Il modello viene serializzato con `joblib` in `outputs/modello_lightgbm_<dataset>.pkl`.

---

#### Random Forest — `Random Forest Classifier/`

Classe `RandomForestTrainer`, stessa struttura di `LightGBMTrainer` (Fast Mode + RandomizedSearchCV).

Parametri Fast Mode: `n_estimators=300`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=2`.

Random Forest non accetta stringhe: il main applica One-Hot Encoding su train+test combinati e converte tutto a `float`. Serializzato in `outputs/modello_rf_<dataset>.pkl`.

---

#### Support Vector Classifier — `Support_Vector_Classifier/`

Classe `SVCTrainer` con Pipeline sklearn interna: `StandardScaler` + `SVC(kernel='rbf', C=10, probability=True)`. Lo scaling è dentro la Pipeline, quindi non serve preprocessing manuale separato e il leakage è prevenuto automaticamente.

Stessa gestione One-Hot Encoding del Random Forest. Serializzato in `outputs/modello_svc_<dataset>.pkl`.

---

#### Rete Neurale PyTorch — `NN_Pytorch/`

Implementazione custom con architettura piramidale. **Versione integrata con l'Orchestratore.**

```
Input → Linear(128) → BN → ReLU → Dropout(0.3)
      → Linear(64)  → BN → ReLU → Dropout(0.2)
      → Linear(32)  → BN → ReLU
      → Linear(1)   → Sigmoid
```

| Parametro | Valore |
|---|---|
| `epochs` | 50 (Holdout/Full) / 30 (K-Fold) |
| `batch_size` | 64 |
| `optimizer` | AdamW (`weight_decay=1e-4`) |
| `scheduler` | `ReduceLROnPlateau` (factor=0.5, patience=5) |
| `loss` | BCELoss |

Lo `StandardScaler` viene salvato come `model.scaler` per il transform sul test senza refitting. La funzione `align_columns()` gestisce il disallineamento OHE tra train e test tramite `.reindex()`.

---

#### Rete Neurale sklearn — `Rete_Neurale/`

Implementazione alternativa con `MLPClassifier` in una Pipeline sklearn (`StandardScaler` + MLP). Architettura: `hidden_layer_sizes=(128, 64)`, `early_stopping=True`, `max_iter=2000`.

> Versione alternativa: supporta solo Holdout e K-Fold, non integrata con l'Orchestratore.

---

### Output del Development

| File | Quando viene creato |
|---|---|
| `outputs/temp_metrics.json` | Dopo ogni run (cancellato dall'Orchestratore dopo la lettura) |
| `outputs/leaderboard_sessione_corrente.csv` | All'uscita con opzione `0` |
| `outputs/modello_<tipo>_<dataset>.pkl` | Dopo ogni training (LightGBM, RF, SVC) |
| `outputs/submission_<tipo>_<dataset>.csv` | In Holdout e Full Training |
| `outputs/prob_<tipo>_<dataset>.csv` | In Holdout e Full Training (LightGBM, RF, SVC) |
| `outputs/submission_*_kfold_TOTAL.csv` | In K-Fold (predizioni aggregate) |
| `outputs/op6/correlation_matrix.png` | Dopo OP6 nel preprocessing |

---

## 5. Test Unitari

### Test Preprocessing (Test1 – Test5)

Ogni test crea un DataFrame sintetico, esegue il modulo da testare e verifica il risultato con asserzioni precise. Tutti sono isolati e indipendenti.

| Test | Modulo | Obiettivo |
|---|---|---|
| `Test1.py` | `op3` | Split corretto di `Cabin` e `PassengerId`, conteggio `GroupSize` |
| `Test2.py` | `op8` | Lo scaler fa `.fit()` solo sul train — nessun data leakage |
| `Test3.py` | `op4` | Zero NaN dopo l'imputazione, regole di dominio applicate |
| `Test4.py` | `op9` | Encoder robusto a categorie sconosciute, colonne allineate tra train e test |
| `Test5.py` | `op5` | `TotalSpending` matematicamente corretto, `Names` rimossa, `Surnames` conservata |

**Dettaglio Test2 (Data Leakage):** crea un train con Age media=25 e un test con Age media=90. Se lo scaler porta anche il test a media≈0, significa che ha fatto il fit anche sul test (leakage). Il test fallisce in quel caso.

**Dettaglio Test4 (Categoria aliena):** inserisce il pianeta `'Plutone'` nel test set, mai visto nel train. Verifica che `handle_unknown='ignore'` eviti il crash e che le colonne di train e test siano identiche dopo l'OHE.

```bash
# Singolo test
python -m unittest tests/Test1.py

# Tutti i test del preprocessing insieme
python -m unittest discover -s tests -p "Test*.py"
```

---

### Test Modelli

#### `test_catboost_model.py`

| Test | Verifica |
|---|---|
| `test_prepare_test` | `PassengerId` escluso dalle feature di input |
| `test_create_model` | Istanza del modello creata correttamente |
| `test_train_and_predict` | N predizioni per N campioni dopo il training |
| `test_evaluate_model` | Cross-validation score ∈ [0, 1] |
| `test_create_submission` | File CSV con colonne `PassengerId` e `Transported` |

```bash
python -m unittest test_catboost_model.py
```

---

## 6. Decisioni Architetturali

- **Due rami paralleli (tree vs nn)** a partire da OP8 — i modelli ad albero non richiedono né scaling né OHE; mantenere due versioni separate evita di degradare le performance degli alberi con trasformazioni inutili.
- **Imputazione group-first** — la coesione intra-gruppo è >85% per `HomePlanet`: la moda del gruppo è più affidabile dell'imputazione globale. L'imputazione probabilistica viene usata solo come fallback.
- **Regole di dominio prima dell'imputazione statistica** — applicare prima le regole logiche riduce il numero di valori da imputare e aumenta la coerenza del dataset.
- **Anti-leakage by design** — dizionari di probabilità, media dell'età e scaler vengono sempre fittati sul train e passati come parametri al test.
- **Versione Fantasma** — la modalità di training sull'intero dataset è volutamente nascosta dal menu per evitare utilizzi prematuri. Va usata solo dopo aver selezionato il modello migliore tramite Holdout o K-Fold.
- **temp_metrics.json come canale di comunicazione** — i modelli sono eseguiti come sottoprocessi indipendenti dall'Orchestratore (`subprocess.run`). Il file JSON temporaneo è il canale di passaggio delle metriche tra il processo figlio (modello) e il processo padre (Orchestratore).

---

## 7. Requisiti

```
# Pre-processing
pandas
numpy
scikit-learn
seaborn
matplotlib
openpyxl      # per .xlsx
pyarrow       # per .parquet

# Modelli
xgboost
catboost
lightgbm
torch
joblib
```

Installazione completa:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib openpyxl pyarrow xgboost catboost lightgbm torch joblib
```
