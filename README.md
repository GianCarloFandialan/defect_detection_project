# Industrial Quality Control — Defect Detection

## Project Title and Overview

**Industrial Quality Control — Defect Detection**

Il progetto realizza un sistema di controllo qualità industriale che classifica le immagini di
piastre metalliche in **good** (senza difetti) o **defect** (con graffi, fori, macchie o crepe),
e localizza automaticamente la posizione del difetto rilevato. Il progetto mette a confronto, in
modo affiancato, due approcci end-to-end: una pipeline **classica** di Computer Vision (HOG +
SVM lineare) e una pipeline **deep learning** (VGG16 con fine-tuning), come proposto dal brief
del corso.

Il tema scelto fra quelli proposti dalla slide 3 del brief è _Industrial Quality Control System
— Defect Detection_, motivato dalla sua naturalezza come problema di **classificazione binaria**
con metriche standard, abbinato a uno stadio di **localizzazione** del difetto basato su
tecniche classiche di image processing (binarizzazione di Otsu, operazioni morfologiche e
analisi delle componenti connesse).

---

## Description of the Pipeline and Architecture

Il progetto implementa per intero la pipeline a 5 stadi richiesta dalla slide 4 del brief:

| #   | Stadio                               | Modulo                                            | Tecnica                                                                                                     |
| --- | ------------------------------------ | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 1   | Data Acquisition & Preprocessing     | `01_data_generation.py`, `02_preprocessing.py`    | Generazione dataset sintetico, gray-world white balance, median blur, equalizzazione dell'istogramma in HSV |
| 2   | Feature Engineering / Representation | `03_classical_pipeline.py`, `04_deep_pipeline.py` | HOG (Histograms of Oriented Gradients) + features VGG16 pre-allenate su ImageNet                            |
| 3   | Core Logic                           | `03_classical_pipeline.py`, `04_deep_pipeline.py` | SVM lineare (`cv2.ml.SVM`) + CNN (VGG16 con fine-tuning a due fasi)                                         |
| 4   | Post-processing                      | `05_postprocessing.py`                            | Otsu thresholding + morfologia (opening + closing) + `connectedComponentsWithStats`                         |
| 5   | Performance Evaluation               | `06_evaluation.py`                                | Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC curve                                          |

### Struttura del repository

```
defect_detection_project/
├── README.md                        ← questo file
├── requirements.txt                 ← dipendenze Python con versioni fissate
├── Technical_Analysis_Document.pdf  ← documento di analisi tecnica (slide 8)
├── .gitignore                       ← esclude env/, data/, modelli, cache
└── src/
    ├── 01_data_generation.py        ← genera 1000 immagini sintetiche
    ├── 02_preprocessing.py          ← white balance + denoising + equalizzazione
    ├── 03_classical_pipeline.py     ← HOG features + SVM lineare
    ├── 04_deep_pipeline.py          ← VGG16 transfer learning + fine-tuning
    ├── 05_postprocessing.py         ← Otsu + morfologia + bounding box
    ├── 06_evaluation.py             ← metriche, confusion matrix, ROC, confronto SVM vs CNN
    └── main.py                      ← orchestratore end-to-end (esegue tutti i moduli)
```

I moduli sono **indipendenti** e auto-eseguibili (`python src/0X_*.py`), e possono essere
importati anche da altri script. L'orchestratore `main.py` esegue tutti e 6 i moduli in
sequenza e implementa il principio di **idempotenza**: non rifa il lavoro già completato
(salta la generazione del dataset se le immagini esistono già, salta il training della SVM se
il modello è già stato salvato, e così via).

### Pipeline classica (HOG + SVM)

Il descrittore HOG estrae un vettore di feature da 8100 dimensioni per ogni immagine
calcolando le orientazioni dei gradienti su celle di 8×8 pixel, normalizzate in blocchi di
16×16 con 9 bin angolari. La SVM utilizza un kernel lineare (sufficiente per lo spazio ad alta
dimensionalità di HOG) ed è allenata su 800 immagini (400 per classe) con i campioni mescolati
in modo deterministico (seed = 42).

### Pipeline deep (VGG16 transfer learning)

Il backbone VGG16 viene caricato con i pesi pre-allenati di ImageNet e la sua testa di
classificazione originale viene sostituita con una testa custom per classificazione binaria
(Flatten → Dense 256 + ReLU → Dropout 0.5 → Dense 1 + Sigmoid). Il training avviene in due fasi:

- **Fase 1** (15 epoche, learning rate 2e-5): backbone congelato, si allena solo la testa.
- **Fase 2** (10 epoche, learning rate 1e-5): si sbloccano gli ultimi 4 layer convoluzionali
  di VGG16 per il fine-tuning sul dominio dei difetti metallici.

La data augmentation (rotazione, traslazione, shear, zoom, flip orizzontale) è applicata solo
al training set; il test set utilizza il solo `rescale=1./255` per ottenere metriche
riproducibili.

---

## Setup and Running Instructions

### Prerequisiti comuni

- **Python 3.10, 3.11 o 3.12** (la versione 3.13 non è ancora supportata da TensorFlow su
  Apple Silicon). Verifica la versione con `python3 --version` oppure `python --version`.
- **Git** (per clonare il repository).
- Circa **2 GB di spazio libero su disco** (300 MB di dipendenze + ~270 MB per il modello CNN
  salvato).
- Connessione internet alla prima esecuzione (Keras scarica ~58 MB di pesi VGG16 dai server
  Google).

### Step 1 — Clonare il repository

```bash
git clone https://github.com/GianCarloFandialan/defect_detection_project.git
cd defect-detection-cv
```

### Step 2 — Creare l'ambiente virtuale

Il comando varia leggermente in base al sistema operativo.

#### macOS Apple Silicon (M1 / M2 / M3) e macOS Intel

```bash
python3.12 -m venv env
source env/bin/activate
```

> **Nota M1/M2/M3:** se hai Python 3.13 installato dal sito ufficiale, usa esplicitamente
> `python3.12` come mostrato sopra. Se Python 3.12 non è installato, scaricalo da
> [python.org/downloads/macos](https://www.python.org/downloads/macos/).

#### Linux (Ubuntu / Debian / Fedora / Arch)

```bash
python3 -m venv env
source env/bin/activate
```

> Se Python 3.12 non è disponibile nei pacchetti di default, su Ubuntu:
>
> ```bash
> sudo apt install python3.12 python3.12-venv
> python3.12 -m venv env
> ```

#### Windows 10 / 11

Da Prompt dei Comandi (`cmd`):

```cmd
python -m venv env
env\Scripts\activate
```

Da PowerShell:

```powershell
.\env\Scripts\Activate.ps1
```

Se PowerShell rifiuta l'attivazione per criteri di sicurezza:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Su tutti i sistemi, una volta attivato l'ambiente vedrai `(env)` all'inizio del prompt.

### Step 3 — Installare le dipendenze

#### macOS Apple Silicon (M1 / M2 / M3) — installazione ottimizzata GPU

Per sfruttare la GPU integrata del chip Apple tramite l'API Metal:

```bash
pip install --upgrade pip
pip install numpy matplotlib opencv-python opencv-contrib-python scikit-learn seaborn
pip install "tensorflow-metal==1.2.0" "tensorflow<2.18"
```

`tensorflow-metal` è il plugin che attiva il supporto GPU su Apple Silicon. La coppia di
versioni indicate (`tensorflow 2.17.x` + `tensorflow-metal 1.2.0`) è la combinazione stabile
testata.

#### macOS Intel, Linux, Windows — installazione standard

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Oppure manualmente:

```bash
pip install numpy matplotlib opencv-python opencv-contrib-python scikit-learn seaborn tensorflow
```

> Su Linux con GPU NVIDIA, per ottenere l'accelerazione CUDA installa `tensorflow[and-cuda]`
> al posto di `tensorflow` (è richiesto un driver NVIDIA compatibile).

### Step 4 — Verificare l'installazione

```bash
python -c "import tensorflow as tf; import cv2; print('TF:', tf.__version__); print('cv2:', cv2.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Output atteso (i numeri di versione possono variare):

```
TF: 2.17.1
cv2: 4.13.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Se la riga `GPU` è una lista vuota `[]`, il progetto funzionerà comunque in CPU, ma il
training del modulo 04 (VGG16) sarà più lento.

### Step 5 — Eseguire la pipeline

Per lanciare la pipeline completa end-to-end:

```bash
python src/main.py
```

Tempi attesi:

| Step | Cosa fa                             | macOS M3 (GPU)  | CPU desktop     | Colab T4 GPU    |
| ---- | ----------------------------------- | --------------- | --------------- | --------------- |
| 1    | Generazione dataset (1000 immagini) | ~30 s           | ~30 s           | ~30 s           |
| 2    | Demo preprocessing                  | qualche secondo | qualche secondo | qualche secondo |
| 3    | Training HOG + SVM                  | 1-2 min         | 1-2 min         | 1-2 min         |
| 4    | **Training VGG16**                  | **~8-15 min**   | **20-45 min**   | **~5-10 min**   |
| 5    | Demo localizzazione                 | qualche secondo | qualche secondo | qualche secondo |
| 6    | Valutazione e confronto             | 1-2 min         | 1-2 min         | 1-2 min         |

In alternativa si possono eseguire i moduli uno alla volta:

```bash
python src/01_data_generation.py
python src/02_preprocessing.py
python src/03_classical_pipeline.py
python src/04_deep_pipeline.py
python src/05_postprocessing.py
python src/06_evaluation.py
```

---

## Summary of Results

Tutti i numeri riportati qui sotto derivano da un training reale sul dataset sintetico
generato dal modulo 1 (800 immagini di training, 200 di test, perfettamente bilanciate fra
le due classi).

| Modello                  | Accuracy   | Precision  | Recall     | F1-score   |
| ------------------------ | ---------- | ---------- | ---------- | ---------- |
| SVM + HOG (linear)       | 54.0%      | 54.3%      | 51.0%      | 52.6%      |
| **CNN VGG16 fine-tuned** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

**Confusion matrix sul test set (200 campioni, 100 per classe):**

```
SVM + HOG                CNN VGG16 fine-tuned
              pred good  pred defect              pred good  pred defect
true good        57         43         true good    100          0
true defect      49         51         true defect    0        100
```

**Osservazioni chiave:**

- Il classificatore SVM si attesta poco sopra il caso casuale (50% su un problema binario
  bilanciato), confermando che i descrittori HOG globali fanno fatica a catturare i difetti
  piccoli, localizzati e di geometria variabile presenti nel dataset.
- La CNN raggiunge la convergenza totale sul validation set già all'epoca 16 (inizio della
  fase di fine-tuning) e mantiene il punteggio perfetto fino alla fine. La differenza fra
  training accuracy (~98%) e validation accuracy (100%) è il segno opposto dell'overfitting:
  Dropout(0.5) disattiva metà dei neuroni della testa durante il training, quindi il modello
  impara di fatto "a metà capacità" ma in inferenza utilizza tutti i neuroni.
- Il risultato del 100% è realistico data la semplicità del dataset sintetico. Su dataset
  industriali reali come MVTec AD ci si attende un calo dell'accuracy a 92–96%.

**Il dettaglio dei risultati, l'analisi dei fallimenti e le considerazioni etiche sono
documentati nel `Technical_Analysis_Document.pdf` (slide 8 del brief).**

---

## Librerie utilizzate

| Libreria                 | Uso nel progetto                                            |
| ------------------------ | ----------------------------------------------------------- |
| OpenCV (`cv2`, `cv2.ml`) | Image processing, HOG, SVM, morfologia, connectedComponents |
| TensorFlow / Keras       | VGG16 pre-allenato, training, salvataggio modello           |
| scikit-learn             | Metriche di classificazione, shuffle deterministico         |
| NumPy                    | Array e operazioni numeriche sulle immagini                 |
| Matplotlib               | Visualizzazione grafici, confusion matrix, ROC              |
| Seaborn                  | Heatmap migliorate per le confusion matrix                  |

La lista completa con le versioni fissate è in `requirements.txt`.

---

## Note tecniche

**Dataset sintetico.** Il dataset è generato programmaticamente per garantire la piena
riproducibilità del progetto: chiunque cloni il repository ed esegua `main.py` ottiene
esattamente lo stesso dataset (seed di numpy fissato a 42). Le quattro tipologie di difetto
(graffio, foro, macchia semi-trasparente, crepa) sono disegnate con le primitive di OpenCV
(`cv2.line`, `cv2.circle`, `cv2.addWeighted`).

**Riproducibilità.** Tutti i passaggi casuali sono seedati. La SVM è deterministica (kernel
lineare con soluzione unica). Il training della CNN ha minime fluttuazioni dovute
all'augmentation casuale, ma converge sempre allo stesso plateau di accuracy.

**Idempotenza.** `main.py` riconosce se uno step è già stato completato (file di output
presente su disco) e lo salta. Per rieseguire da zero, basta cancellare i file generati:

```bash
rm -rf data svm_defect_detector.yml vgg16_defect_detector.keras
```

---

## Risoluzione problemi

**`ModuleNotFoundError: No module named 'cv2'` / `tensorflow`**
L'ambiente virtuale non è attivo. Esegui `source env/bin/activate` (Linux/Mac) oppure
`env\Scripts\activate` (Windows).

**`Could not find a version that satisfies the requirement tensorflow-metal`**
Su macOS Apple Silicon, accade con Python 3.13. Ricrea l'ambiente virtuale con Python 3.12:

```bash
rm -rf env && python3.12 -m venv env && source env/bin/activate
```

**`SystemError: null argument to internal routine` durante il salvataggio del modello CNN**
Bug noto del salvataggio in formato HDF5 su Apple Silicon. Il modello viene comunque salvato
correttamente in formato `.keras` e la pipeline prosegue normalmente.

**Le finestre di matplotlib non si aprono / restano bloccate**
Su macOS recente può servire `pip install pyobjc-framework-Cocoa`. Su Linux server senza
display, sostituire `plt.show()` con `plt.savefig('output.png')` nei singoli moduli.

---

## Autore

**Gian Carlo Fandialan**
