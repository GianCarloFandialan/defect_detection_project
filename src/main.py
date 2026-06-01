"""
MAIN - PIPELINE END-TO-END
Orchestra l'intera pipeline del progetto eseguendo in sequenza i 6 moduli
(slide 4 del brief):

    Step 1 - Data Acquisition           (modulo 01)
    Step 2 - Preprocessing demo         (modulo 02)
    Step 3 - HOG + SVM training         (modulo 03)
    Step 4 - VGG16 training             (modulo 04)
    Step 5 - Post-processing demo       (modulo 05)
    Step 6 - Evaluation completa        (modulo 06)

Ogni step e' idempotente: se il file di output esiste gia', lo step viene
saltato (utile per rieseguire dopo un crash senza perdere lavoro).

Esecuzione:
    python src/main.py
"""

import os
import sys
import importlib.util

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Import dinamico dei moduli (i nomi che iniziano con cifra non si importano
# normalmente). sys.modules[name] = mod registra il modulo cosi' altri moduli
# che lo importano lo trovano.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


data_gen = _load("data_gen",        "01_data_generation.py")
preprocessing = _load("preprocessing", "02_preprocessing.py")
classical = _load("classical",      "03_classical_pipeline.py")
deep = _load("deep",                "04_deep_pipeline.py")
postproc = _load("postproc",        "05_postprocessing.py")
evaluation = _load("evaluation",    "06_evaluation.py")


# ---------------------------------------------------------------------------
# Path comuni
# ---------------------------------------------------------------------------
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SVM_PATH = "svm_defect_detector.yml"
CNN_PATH = "vgg16_defect_detector.keras"


# ---------------------------------------------------------------------------
# STEP 1 - Generazione dataset
# ---------------------------------------------------------------------------

def step1_generate_data():
    print("\n" + "#" * 70)
    print("STEP 1 - DATA ACQUISITION (generazione dataset sintetico)")
    print("#" * 70)
    np.random.seed(42)                   # riproducibilita'
    if not os.path.isdir(os.path.join(TRAIN_DIR, "good")):
        data_gen.generate_dataset(DATA_DIR, n_train=400, n_test=100,
                                  img_size=(150, 150))
    else:
        print("Dataset gia' presente, salto la generazione.")


# ---------------------------------------------------------------------------
# STEP 2 - Demo preprocessing
# ---------------------------------------------------------------------------

def step2_preview_preprocessing():
    print("\n" + "#" * 70)
    print("STEP 2 - PREPROCESSING (white balance + denoise + equalizzazione)")
    print("#" * 70)

    sample = os.path.join(TRAIN_DIR, "defect", "defect_0000.png")
    if not os.path.exists(sample):
        print("Campione non trovato.")
        return

    img = cv2.imread(sample)
    proc = preprocessing.preprocess_pipeline(img)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Originale")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    plt.title("Dopo preprocessing")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# STEP 3 - Training pipeline classica
# ---------------------------------------------------------------------------

def step3_train_svm():
    print("\n" + "#" * 70)
    print("STEP 3 - FEATURE ENG. + CORE LOGIC: HOG + SVM (classico)")
    print("#" * 70)

    if os.path.exists(SVM_PATH):
        print(f"Modello SVM gia' presente in {SVM_PATH}, salto training.")
        return

    from sklearn.utils import shuffle

    hog = classical.build_hog_descriptor()
    pos_paths, neg_paths = classical.collect_paths(TRAIN_DIR)
    pos_feat, pos_lab = classical.extract_hog_features(pos_paths, hog, 1)
    neg_feat, neg_lab = classical.extract_hog_features(neg_paths, hog, -1)

    X = np.concatenate((pos_feat, neg_feat), axis=0)
    y = np.concatenate((pos_lab, neg_lab), axis=0)
    # Shuffle critico per non introdurre bias (vedi modulo 03).
    X, y = shuffle(X, y, random_state=42)
    X = X.reshape(X.shape[0], -1)

    classical.train_svm(X, y, SVM_PATH)


# ---------------------------------------------------------------------------
# STEP 4 - Training pipeline deep
# ---------------------------------------------------------------------------

def step4_train_cnn():
    print("\n" + "#" * 70)
    print("STEP 4 - FEATURE ENG. + CORE LOGIC: VGG16 transfer learning (deep)")
    print("#" * 70)

    if os.path.exists(CNN_PATH):
        print(f"Modello CNN gia' presente in {CNN_PATH}, salto training.")
        return

    model, conv_base = deep.build_model()
    model.summary()
    train_gen, test_gen = deep.build_generators(DATA_DIR)
    history, history_fine = deep.train_two_phases(model, conv_base,
                                                  train_gen, test_gen)
    model.save(CNN_PATH)
    deep.plot_history(history, history_fine)


# ---------------------------------------------------------------------------
# STEP 5 - Demo post-processing
# ---------------------------------------------------------------------------

def step5_postprocessing_demo():
    print("\n" + "#" * 70)
    print("STEP 5 - POST-PROCESSING (localizzazione del difetto)")
    print("#" * 70)

    import glob
    sample_paths = sorted(glob.glob(
        os.path.join(TEST_DIR, "defect", "*.png")))[:4]
    if not sample_paths:
        print("Nessun campione difettoso trovato.")
        return

    fig, axes = plt.subplots(len(sample_paths), 3,
                             figsize=(12, 3 * len(sample_paths)))
    if len(sample_paths) == 1:
        axes = np.array([axes])

    for row, p in enumerate(sample_paths):
        img = cv2.imread(p)
        boxes, mask = postproc.localize_defect(img, min_area=20)
        drawn = postproc.draw_boxes(img, boxes)

        axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title("Originale")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("Mask post-morph")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title(f"Difetti: {len(boxes)}")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# STEP 6 - Valutazione completa
# ---------------------------------------------------------------------------

def step6_evaluation():
    print("\n" + "#" * 70)
    print("STEP 6 - PERFORMANCE EVALUATION (slide 6)")
    print("#" * 70)

    from sklearn.metrics import classification_report

    svm_metrics = cnn_metrics = None

    if os.path.exists(SVM_PATH):
        y_true_s, y_pred_s = evaluation.evaluate_svm(SVM_PATH, TEST_DIR)
        svm_metrics = evaluation.compute_metrics(y_true_s, y_pred_s, "SVM (HOG)")
        print(classification_report(y_true_s, y_pred_s,
                                    target_names=["good", "defect"],
                                    zero_division=0))
        evaluation.plot_confusion_matrix(y_true_s, y_pred_s, "SVM (HOG)")

    if os.path.exists(CNN_PATH):
        # evaluate_cnn ritorna 3 valori (in piu' le probabilita' per ROC).
        y_true_c, y_pred_c, y_probs_c = evaluation.evaluate_cnn(CNN_PATH, TEST_DIR)
        cnn_metrics = evaluation.compute_metrics(y_true_c, y_pred_c, "CNN (VGG16)")
        print(classification_report(y_true_c, y_pred_c,
                                    target_names=["good", "defect"],
                                    zero_division=0))
        evaluation.plot_confusion_matrix(y_true_c, y_pred_c, "CNN (VGG16)")
        evaluation.plot_roc_curve(y_true_c, y_probs_c, "CNN (VGG16)")

    if svm_metrics and cnn_metrics:
        evaluation.comparison_report(svm_metrics, cnn_metrics)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    step1_generate_data()
    step2_preview_preprocessing()
    step3_train_svm()
    step4_train_cnn()
    step5_postprocessing_demo()
    step6_evaluation()
    print("\n>>> Pipeline completa terminata con successo. <<<")
