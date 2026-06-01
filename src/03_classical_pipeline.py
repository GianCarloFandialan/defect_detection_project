"""
MODULO 3 - PIPELINE CLASSICA (HOG + SVM)
Feature engineering con HOG (Histograms of Oriented Gradients) + classificazione
con SVM lineare (cv2.ml.SVM).

HOG cattura la distribuzione locale dei gradienti orientati, sensibile ai bordi
e quindi adatta a rilevare graffi, crepe e altri difetti caratterizzati da
discontinuita' nell'immagine.

La SVM lineare e' la scelta classica per HOG: il vettore di feature (8100 dim)
e' gia' ad alta dimensionalita' e nello spazio HOG le classi sono spesso
linearmente separabili, quindi un kernel non lineare (RBF) rischierebbe solo
overfitting.
"""

import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import importlib.util


# ---------------------------------------------------------------------------
# Import dinamico del modulo 2 (i nomi che iniziano con cifra non si importano
# normalmente con "import 02_preprocessing"). Usiamo importlib.util.
# ---------------------------------------------------------------------------

def _load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HERE = os.path.dirname(os.path.abspath(__file__))
preprocessing = _load_module_from_path(
    "preprocessing",
    os.path.join(_HERE, "02_preprocessing.py")
)


# ---------------------------------------------------------------------------
# Parametri HOG
# ---------------------------------------------------------------------------
# Valori scelti seguendo l'articolo originale Dalal-Triggs 2005, adattati a
# winSize quadrata (vs il classico 64x128 da pedestrian detector): le piastre
# hanno aspect ratio 1:1.
#
# Dimensione del vettore HOG risultante:
#   - 15x15 blocchi (cell=8, stride=8 in winSize=128) = 225 blocchi
#   - 4 celle per blocco * 9 bin per cella = 36 valori per blocco
#   - totale = 225 * 36 = 8100 valori

WIN_SIZE = (128, 128)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE = (8, 8)
NBINS = 9


def build_hog_descriptor():
    """Istanzia il descrittore HOG con i parametri sopra."""
    return cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE,
                             CELL_SIZE, NBINS)


# ---------------------------------------------------------------------------
# Estrazione feature da una lista di immagini
# ---------------------------------------------------------------------------

def extract_hog_features(image_paths, hog_descriptor, label, apply_preproc=True):
    """
    Estrae HOG da una lista di immagini e ritorna (features, labels).

    Per ogni path: carica BGR -> [preprocessing] -> grayscale -> resize a winSize
    -> hog.compute().

    Convenzione label: +1 = defect, -1 = good (label standard per SVM binaria).
    """
    features_list = []
    labels_list = []
    print(f"  Estrazione HOG per la classe label={label}: {len(image_paths)} campioni...")

    for image_path in image_paths:
        img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_color is None:
            print(f"  Attenzione: impossibile leggere {image_path}")
            continue

        if apply_preproc:
            img_color = preprocessing.preprocess_pipeline(img_color)

        # HOG lavora sui gradienti, calcolati su singolo canale -> grayscale.
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # img_gray.shape e' (h, w), WIN_SIZE e' (w, h): ordine INVERTITO.
        if (img_gray.shape[1], img_gray.shape[0]) != WIN_SIZE:
            img_gray = cv2.resize(img_gray, WIN_SIZE)

        feat = hog_descriptor.compute(img_gray)
        features_list.append(feat)
        labels_list.append(label)

    # cv2.ml.SVM richiede ESATTAMENTE questi dtype: float32 per X, int32 per y.
    # Senza i cast espliciti, OpenCV solleva errori opachi.
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    return features, labels


# ---------------------------------------------------------------------------
# Training SVM
# ---------------------------------------------------------------------------

def train_svm(train_features, train_labels, svm_save_path):
    """
    Allena una SVM lineare e la salva in formato YAML.

    Parametri SVM:
        SVM_C_SVC   = C-Support Vector Classification (formulazione standard)
        SVM_LINEAR  = kernel lineare (sufficiente per HOG ad alta dimensionalita')
        max_iter=1000, eps=1e-6 = criteri di stop
    """
    print("Inizio training SVM...")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria(
        (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6)
    )
    # ROW_SAMPLE = ogni riga della matrice e' un campione.
    svm.train(train_features, cv2.ml.ROW_SAMPLE, train_labels)
    print("Training SVM completato.")
    svm.save(svm_save_path)
    print(f"Modello SVM salvato in: {svm_save_path}")
    return svm


def collect_paths(split_dir):
    """
    Raccoglie i path delle immagini dalle cartelle 'defect' e 'good'.

    Convenzione: defect = positivi (+1), good = negativi (-1).
    """
    pos_dir = os.path.join(split_dir, "defect")
    neg_dir = os.path.join(split_dir, "good")
    pos_paths = [os.path.join(pos_dir, f) for f in sorted(os.listdir(pos_dir))]
    neg_paths = [os.path.join(neg_dir, f) for f in sorted(os.listdir(neg_dir))]
    return pos_paths, neg_paths


# ---------------------------------------------------------------------------
# Entry point: training su train set + quick check su test set
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    SVM_MODEL_PATH = "svm_defect_detector.yml"

    if not os.path.isdir(TRAIN_DIR):
        print(f"Errore: cartella '{TRAIN_DIR}' non trovata.")
        print("Esegui prima 01_data_generation.py")
        exit()

    hog = build_hog_descriptor()

    # --- TRAIN ---
    print("=== Estrazione features TRAIN ===")
    train_pos_paths, train_neg_paths = collect_paths(TRAIN_DIR)
    pos_feat, pos_lab = extract_hog_features(train_pos_paths, hog, label=1)
    neg_feat, neg_lab = extract_hog_features(train_neg_paths, hog, label=-1)

    train_features = np.concatenate((pos_feat, neg_feat), axis=0)
    train_labels = np.concatenate((pos_lab, neg_lab), axis=0)

    # Shuffle CRITICO: senza, la SVM vedrebbe prima tutti i positivi e poi
    # tutti i negativi -> bias nell'ottimizzazione. random_state=42 per
    # determinismo.
    train_features, train_labels = shuffle(
        train_features, train_labels, random_state=42
    )
    train_features = train_features.reshape(train_features.shape[0], -1)

    print(f"Campioni di training: {train_features.shape[0]}, "
          f"dim. feature: {train_features.shape[1]}")
    svm = train_svm(train_features, train_labels, SVM_MODEL_PATH)

    # --- TEST ---
    print("\n=== Estrazione features TEST e valutazione ===")
    test_pos_paths, test_neg_paths = collect_paths(TEST_DIR)
    test_pos_feat, test_pos_lab = extract_hog_features(test_pos_paths, hog, 1)
    test_neg_feat, test_neg_lab = extract_hog_features(test_neg_paths, hog, -1)

    test_features = np.concatenate((test_pos_feat, test_neg_feat), axis=0)
    test_labels = np.concatenate((test_pos_lab, test_neg_lab), axis=0)
    test_features = test_features.reshape(test_features.shape[0], -1)

    # svm.predict ritorna (status, predictions); ignoro status.
    _, predictions = svm.predict(test_features)
    accuracy = np.mean(predictions.flatten() == test_labels.flatten()) * 100
    print(f"Accuracy SVM sul test set: {accuracy:.2f}%")
