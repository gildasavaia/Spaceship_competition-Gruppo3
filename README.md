# Spaceship Titanic Challenge

Il contest "Spaceship Titanic" ci sfida a risolvere un mistero cosmico. Sfruttando le nostre competenze di data science, puntiamo a prevedere il destino dei passeggeri che hanno incontrato un'anomalia spazio-temporale durante il loro viaggio.

[Challenge Link](https://miosito.com)

# Project Files
I file chiavi  relativi a questo progetto sono i seguenti: 
- [train.csv](https://github.com/gildasavaia/Spaceship_competition-Gruppo3/blob/main/data/train.csv) : Set di dati di addestramento contenente informazioni personali ed etichette di riferimento.
- [test.csv](https://github.com/gildasavaia/Spaceship_competition-Gruppo3/blob/main/data/test.csv) : Set di dati per la predizione
- [sample_submission.csv](https://github.com/gildasavaia/Spaceship_competition-Gruppo3/blob/main/data/sample_submission.csv)

# Struttura del progetto
Il progetto Titanic Spaceship si articola in tre macro moduli principali, che rappresentano le fasi fondamentali del workflow:

- [Pre-processing]()
- [Development]()
- [Evaluation]()

#### Requisiti
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- openpyxl      # per .xlsx
- pyarrow       # per .parquet


##### Installazione:

``` pip install pandas numpy scikit-learn seaborn matplotlib openpyxl pyarrow ```


## Pre-processing
Il pre-processing è la fase in cui i dati grezzi vengono trasformati in un formato adatto all’analisi e al modeling.
Questa fase è fondamentale perché la qualità dei dati influisce direttamente sulle performance del modello.
Nel nostro progetto, il pre-processing non è gestito in modo manuale o frammentato, ma è orchestrato tramite un codice di controllo centrale, ovvero una pipeline strutturata. 
La pipeline è composta da operazioni modulari (OP), ognuna responsabile di un singolo step trasformativo. Ogni modulo riceve un DataFrame in input e restituisce un output strutturato tramite NamedTuple o dataclass.

### Pipeline di Pre-processing

Ogni modulo esegue una trasformazione mirata, garantendo:
- modularità del codice
- riutilizzabilità
- facilità di manutenzione
- chiarezza nel flusso dei dati
Questo approccio permette di avere un sistema robusto, scalabile e facilmente estendibile.

### Workflow moduli:
#### OP1 — op1_read_file.py
#### Caricamento e conversione del dataset

Legge il file di input e lo converte in CSV se necessario. Supporta i formati:

| Formato | Estensione |
| :--- | :--- |
| CSV | `.csv` |
| Excel | `.xlsx`, `.xls` |
| JSON | `.json` |
| Parquet | `.parquet` |

```from op1_read_file import run_load_and_convert_to_csv
risultato = run_load_and_convert_to_csv("data/train.csv")
df = risultato.df_output
```

#### OP2 — op2_data_evaluation.py
#### Analisi esplorativa (EDA)

Esegue un'analisi di base sul DataFrame per identificare problemi di qualità dei dati:

Conteggio dei valori nulli per colonna
Analisi delle colonne di tipo object (categoriche)
Dimensioni e tipi del dataset
Restituisce il DataFrame originale invariato (usato come step di ispezione).

#### OP3 — op3_split_dataset.py
#### Feature Engineering — Split delle colonne composte
Estrae feature secondarie da colonne originali codificate:

| Colonna originale | Feature estratte | Descrizione |
| :--- | :--- | :--- |
| PassengerId | `Group`, `GroupSize` | Identificatore del gruppo e numero di membri |
| Cabin | `Deck`, `Num`, `Side` | Ponte, numero cabina, lato (P/S) |
| Name | `Names`, `Surnames` | Nome proprio e cognome |

Le colonne originali (PassengerId, Cabin, Name) vengono rimosse. Nel dataset di test, PassengerId viene mantenuto per la submission finale.

```from op3_split_dataset import run_split_dataset
risultato = run_split_dataset(df, i_train=1)  # i_train=0 per il test set
df_eng = risultato.df_output
```

#### OP4 — op4_handler_nullvalue.py
#### Gestione dei valori mancanti

Il modulo più complesso della pipeline. Combina regole logiche di dominio con strategie statistiche di imputazione.

###### Regole di dominio applicate

| Regola | Dettaglio |
| :--- | :--- |
| `CryoSleep = True → spese = 0` | I passeggeri in criosonnno non possono spendere |
| `Age < 13 → tutte le spese = 0` | I bambini non hanno crediti |
| `Age < 18 → VIP = False` | I minorenni non possono essere VIP |
| `Deck ∈ {G, T} → VIP = False` | I ponti economici non ospitano passeggeri VIP |
| `HomePlanet = Earth → VIP = False` | I terrestri non risultano mai VIP nel dataset |
| `Deck ∈ {A, B, C, T} → HomePlanet = Europa` | Regola empirica da analisi del dataset |
| `Deck = G → HomePlanet = Earth` | Regola empirica da analisi del dataset |
| `Spesa > 0 → CryoSleep = False` | Inferenza inversa dalla spesa |
| `Deck = T + CryoSleep mancante → CryoSleep = False` | Regola specifica per il ponte T |

##### Strategie di imputazione
Per le feature categoriche (HomePlanet, Destination, CryoSleep, VIP, Deck, Num, Side, Surnames):
1. ##### Moda del gruppo:
   per i passeggeri con GroupSize > 1, si usa il valore più frequente del loro gruppo (alta coesione intra-gruppo osservata nel dataset).
Imputazione probabilistica: per i passeggeri solitari o per i gruppi il cui intero gruppo ha il valore mancante, si campiona proporzionalmente alla distribuzione globale della feature nel train set.
2. ##### Nota anti-data-leakage:
   i dizionari di probabilità vengono calcolati solo sul train set e passati come parametro al test set, evitando qualsiasi forma di leakage.

Per le feature numeriche:
1. ##### Costi (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck): NaN → 0
2. ##### Age: NaN → media calcolata sul train (o passata come parametro per il test)


#### OP5 — op5_sumcosts_names.py
#### Aggregazione costi e pulizia colonne

- Crea la feature TotalSpending come somma delle cinque colonne di spesa (calcolata prima di eventuali binarizzazioni).
- Rimuove la colonna Names (il nome proprio non è predittivo).
- La colonna Surnames viene mantenuta in questo step e rimossa in OP9.
- TotalSpending è storicamente una delle feature più predittive in questo dataset, in quanto i passeggeri teletrasportati tendono ad avere spese più basse (o nulle).


#### OP6 — op6_correlation_matrix.py
#### Matrice di correlazione

Calcola e visualizza la matrice di correlazione completa dopo aver eseguito:
- Conversione delle variabili booleane (CryoSleep, VIP, Transported) in 0/1
- One-Hot Encoding di tutte le variabili categoriche (senza drop_first per esplorazione completa)
- Salva il grafico come correlation_matrix.png nella cartella di output specificata.

Restituisce anche la correlazione ordinata con il target Transported.

Nota tecnica: viene usato drop_first=False per l'analisi esplorativa (OP6), mentre in OP9 viene applicato il drop_first implicito tramite eliminazione delle colonne ridondanti (es. CryoSleep_False, VIP_False, Side_S) per evitare la Dummy Variable Trap.

#### OP7 — Divisione del Dataset
Sono disponibili due strategie di valutazione, selezionabili interattivamente dalla pipeline principale:
- op7_holdout_evaluator.py — Hold-out
Split classico train/test. L'utente sceglie la proporzione del test set (es. 0.2 per 80/20).

```python
X_train, X_test, y_train, y_test = esegui_split_dati(df)
```

- op7_kfold_evaluator.py — K-Fold Cross Validation
  Split K-Fold non stratificato con shuffle. L'utente sceglie il numero di fold.

```folds = esegui_split_kfold_standard(df)
for X_train, X_test, y_train, y_test in folds:
    ...
```

#### OP8 — op8_scaling.py
#### Normalizzazione e standardizzazione
Produce due versioni parallele del dataset, ottimizzate per famiglie di modelli diverse:
Le colonne trasformate con log1p sono: TotalSpending, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck.

| Versione | Destinazione | Trasformazioni |
|---|---|---|
| `df_tree` | Modelli ad albero (XGBoost, CatBoost, Random Forest) | Nessuna trasformazione numerica |
| `df_nn` | Reti neurali e modelli lineari | `Log1p` su colonne skewed + `StandardScaler` |

Lo scaler viene fittato solo sul train e applicato al test per prevenire il data leakage.

#### OP9 — op9_encoding.py
#### Encoding delle variabili categoriche

| Versione | Encoder | Note |
|---|---|---|
| `df_tree` | Nessuno (`passthrough`) | CatBoost e XGBoost gestiscono le feature categoriche nativamente |
| `df_nn` | One-Hot Encoding | Rimozione delle colonne ridondanti per evitare multicollinearità |

Colonne categoriche codificate: HomePlanet, CryoSleep, Destination, VIP, Deck, Side.

Colonne sempre rimosse in questo step: Group, Surnames.

Colonne OHE rimosse per evitare la dummy variable trap: CryoSleep_False, VIP_False, Side_S.

#### relations.py — Analisi delle Regole di Dominio
##### Script di analisi esplorativa indipendente dalla pipeline principale. Esegue tre analisi:

1. Regole di esclusione: cerca combinazioni impossibili tra coppie di variabili categoriche (frequenza = 0 nella crosstab).
2. Coesione di gruppo/famiglia: misura quanto spesso i passeggeri dello stesso gruppo o con lo stesso cognome condividono HomePlanet, Destination e Cabin.
3. Regole finanziarie: verifica le regole logiche legate a CryoSleep e all'età dei passeggeri.
4. Questo script ha guidato la definizione delle regole di dominio implementate in OP4.

## Development
La fase di Development è il nucleo predittivo del progetto, in cui i dataset generati dalla pipeline di preprocessing,
sia per gli alberi decisionali che per le reti neurali, vengono dati in input agli algoritmi dei modelli sviluppati per
prevedere il destino dei passeggeri.

### Architettura dei Modelli

Il progetto sviluppa e confronta 6 diverse famiglie di modelli, ognuna ottimizzata per una specifica rappresentazione dei dati:

| Modello | Cartella di riferimento      | Caratteristiche                                                                                                                                                        |
| :--- |:-----------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **LightGBM** | `/LightGBM`                  | Algoritmo di Boosting basato su istogrammi, nativamente categorico e velocissimo.                                                                                      |
| **Random Forest** | `/Random Forest Classifier`  | Ensemble bagging robusto. Converte internamente le categoriche in One-Hot Encoding numerico.                                                                           |
| **Support Vector Classifier** | `/Support_Vector_Classifier` | Modello geometrico-spaziale. Integra una pipeline interna con StandardScaler per evitare distorsioni sulle distanze.                                                   |
| **XGBoost** | `/XGBoost`                   | Algoritmo di Gradient Boosting ottimizzato. Richiede una conversione esplicita dei tipi di dato categorici.                                                            |
| **CatBoost** | `/CATBoost`                  | Algoritmo di Boosting, sviluppato per gestire in modo simmetrico e nativo le variabili categoriche.                                                                    |
| **Rete Neurale** | `/Rete_Neurale`              | Architettura Deep Learning implementata in PyTorch per catturare relazioni non lineari complesse. |

