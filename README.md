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

## Pre-processing
Il pre-processing è la fase in cui i dati grezzi vengono trasformati in un formato adatto all’analisi e al modeling.
Questa fase è fondamentale perché la qualità dei dati influisce direttamente sulle performance del modello.
Nel nostro progetto, il pre-processing non è gestito in modo manuale o frammentato, ma è orchestrato tramite un codice di controllo centrale, ovvero una pipeline strutturata. 
Questa pipeline coordina 6 moduli distinti, ciascuno responsabile di un compito specifico all’interno del processo di preparazione dei dati.

### Pipeline di Pre-processing

Ogni modulo esegue una trasformazione mirata, garantendo:
- modularità del codice
- riutilizzabilità
- facilità di manutenzione
- chiarezza nel flusso dei dati
Questo approccio permette di avere un sistema robusto, scalabile e facilmente estendibile.

## Development 

## Evaluation

