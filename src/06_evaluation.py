"""
MODULO 6 - PERFORMANCE EVALUATION
Valutazione completa dei due modelli con:
    - Accuracy, Precision, Recall, F1-score (sklearn.metrics)
    - Confusion Matrix (seaborn heatmap)
    - ROC curve + AUC (solo per la CNN: la SVM lineare non da' probabilita')
    - Report comparativo SVM vs CNN
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)

import importlib.util


# ---------------------------------------------------------------------------
# Import dinamico del modulo 3 per riutilizzare HOG + collect_paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


classical = _load("classical", "03_classical_pipeline.py")


# ---------------------------------------------------------------------------
# Valutazione SVM
# ---------------------------------------------------------------------------

def evaluate_svm(svm_model_path, test_dir):
    """
    Carica la SVM, estrae HOG sul test set, calcola predizioni.

    Ritorna (y_true, y_pred) in formato {0, 1} - sklearn.metrics preferisce
    questa convenzione rispetto a {-1, +1} usata internamente da OpenCV.
    """
    print("=== Valutazione SVM (HOG + linear SVM) ===")
    svm = cv2.ml.SVM_load(svm_model_path)
    hog = classical.build_hog_descriptor()

    pos_paths, neg_paths = classical.collect_paths(test_dir)
    pos_feat, _ = classical.extract_hog_features(pos_paths, hog, label=1)
    neg_feat, _ = classical.extract_hog_features(neg_paths, hog, label=-1)

    X_test = np.concatenate((pos_feat, neg_feat), axis=0)
    X_test = X_test.reshape(X_test.shape[0], -1)

    y_true = np.concatenate((np.ones(len(pos_paths)),
                             np.zeros(len(neg_paths))))

    _, raw_preds = svm.predict(X_test)
    raw_preds = raw_preds.flatten()
    # Conversione {-1, +1} -> {0, 1}.
    y_pred = (raw_preds > 0).astype(int)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Valutazione CNN
# ---------------------------------------------------------------------------

def evaluate_cnn(model_path, test_dir, img_size=(150, 150), batch_size=20):
    """
    Carica la CNN, valuta sul test set.

    Ritorna (y_true, y_pred, y_pred_probs):
        - y_true / y_pred  : in formato {0, 1}
        - y_pred_probs     : probabilita' raw in [0, 1] (servono per la ROC)
    """
    print("=== Valutazione CNN (VGG16 transfer learning) ===")
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = tf.keras.models.load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # shuffle=False CRITICO: senza, y_true e y_pred non si allineerebbero.
    test_gen = test_datagen.flow_from_directory(
        os.path.join(test_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    y_pred_probs = model.predict(test_gen).flatten()
    # Soglia 0.5 = scelta standard per output sigmoide; abbassare a 0.3
    # favorirebbe il recall in contesti safety-critical.
    y_pred = (y_pred_probs > 0.5).astype(int)
    y_true = test_gen.classes

    # flow_from_directory assegna le classi in ORDINE ALFABETICO:
    #   defect -> 0, good -> 1.
    # Noi vogliamo l'inverso (defect=positivo=1) per coerenza con la SVM.
    # Se l'assegnazione e' "invertita" rispetto a quello che vogliamo, swappiamo.
    class_indices = test_gen.class_indices
    if class_indices.get("good") == 1 and class_indices.get("defect") == 0:
        y_true = 1 - y_true
        y_pred = 1 - y_pred
        y_pred_probs = 1 - y_pred_probs

    return y_true, y_pred, y_pred_probs


# ---------------------------------------------------------------------------
# Metriche e plot
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, model_name=""):
    """Calcola Accuracy, Precision, Recall, F1 per la classe positiva."""
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        # zero_division=0: evita warning se non ci sono predizioni positive.
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
    }
    print(f"\n--- Metriche [{model_name}] ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name=""):
    """Heatmap della confusion matrix (good=0, defect=1)."""
    # labels=[0, 1] forza l'ordine (good prima, defect dopo).
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_names = ["good", "defect"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_probs, model_name=""):
    """Curva ROC + AUC. Richiede PROBABILITA' (non label binarie)."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc = roc_auc_score(y_true, y_pred_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC (AUC = {auc:.3f})")
    # Diagonale di riferimento: classificatore casuale (AUC=0.5).
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--",
             label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return auc


def comparison_report(svm_metrics, cnn_metrics):
    """Tabella di confronto SVM vs CNN."""
    print("\n" + "=" * 55)
    print("CONFRONTO FINALE: SVM (HOG) vs. CNN (VGG16 fine-tuned)")
    print("=" * 55)
    print(f"{'Metric':<12} {'SVM':>10} {'CNN':>10} {'Vincitore':>15}")
    print("-" * 55)
    for k in ["Accuracy", "Precision", "Recall", "F1"]:
        winner = "CNN" if cnn_metrics[k] > svm_metrics[k] else "SVM"
        if cnn_metrics[k] == svm_metrics[k]:
            winner = "tie"
        print(f"{k:<12} {svm_metrics[k]:>10.4f} "
              f"{cnn_metrics[k]:>10.4f} {winner:>15}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = "data"
    TEST_DIR = os.path.join(DATA_DIR, "test")
    SVM_PATH = "svm_defect_detector.yml"
    CNN_PATH = "vgg16_defect_detector.keras"

    svm_metrics = None
    cnn_metrics = None

    # --- SVM ---
    if os.path.exists(SVM_PATH):
        y_true_svm, y_pred_svm = evaluate_svm(SVM_PATH, TEST_DIR)
        svm_metrics = compute_metrics(y_true_svm, y_pred_svm, "SVM (HOG)")
        print("\nReport completo SVM:")
        print(classification_report(y_true_svm, y_pred_svm,
                                    target_names=["good", "defect"],
                                    zero_division=0))
        plot_confusion_matrix(y_true_svm, y_pred_svm, "SVM (HOG)")
    else:
        print(f"Modello SVM non trovato in {SVM_PATH}. "
              f"Esegui prima 03_classical_pipeline.py")

    # --- CNN ---
    if os.path.exists(CNN_PATH):
        y_true_cnn, y_pred_cnn, y_probs_cnn = evaluate_cnn(CNN_PATH, TEST_DIR)
        cnn_metrics = compute_metrics(y_true_cnn, y_pred_cnn, "CNN (VGG16)")
        print("\nReport completo CNN:")
        print(classification_report(y_true_cnn, y_pred_cnn,
                                    target_names=["good", "defect"],
                                    zero_division=0))
        plot_confusion_matrix(y_true_cnn, y_pred_cnn, "CNN (VGG16)")
        # ROC solo per la CNN.
        plot_roc_curve(y_true_cnn, y_probs_cnn, "CNN (VGG16)")
    else:
        print(f"Modello CNN non trovato in {CNN_PATH}. "
              f"Esegui prima 04_deep_pipeline.py")

    if svm_metrics and cnn_metrics:
        comparison_report(svm_metrics, cnn_metrics)
