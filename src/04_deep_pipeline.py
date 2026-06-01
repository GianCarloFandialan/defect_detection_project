"""
MODULO 4 - PIPELINE DEEP (VGG16 transfer learning)
Backbone VGG16 pre-allenato su ImageNet + testa di classificazione binaria,
con training a due fasi (feature extraction -> fine-tuning).

Strategia:
    FASE 1 (15 epoche, LR=2e-5): backbone congelato, si allena solo la testa.
    FASE 2 (10 epoche, LR=1e-5): si sbloccano gli ultimi 4 layer convoluzionali
                                  di VGG16 per il fine-tuning sul dominio
                                  specifico dei difetti metallici.

L'uso di un modello pre-allenato e' giustificato dalla dimensione limitata
del dataset (~800 immagini training): allenare VGG16 da zero (138M parametri)
porterebbe a overfitting garantito.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
IMG_WIDTH, IMG_HEIGHT = 150, 150       # VGG16 accetta qualunque dim multipla di 32
BATCH_SIZE = 20
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 10
MODEL_SAVE_PATH = "vgg16_defect_detector.keras"   # .keras = formato moderno (no warning HDF5)


# ---------------------------------------------------------------------------
# Costruzione del modello
# ---------------------------------------------------------------------------

def build_model():
    """
    Carica VGG16 (senza testa originale) e ci monta sopra la nostra testa
    binaria. Ritorna (model, conv_base) - conv_base serve poi per modificare
    .trainable nella fase 2 di fine-tuning.
    """
    conv_base = VGG16(
        weights="imagenet",                  # download dei pesi pre-allenati
        include_top=False,                   # rimuove la testa 1000-classi ImageNet
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    )
    # Congela tutto VGG16: solo la nostra testa sara' addestrata in fase 1.
    conv_base.trainable = False
    print(f"VGG16 base: {len(conv_base.layers)} layer, trainable={conv_base.trainable}")

    # Testa di classificazione binaria.
    # Su input 150x150 il backbone produce un tensore (4, 4, 512).
    # Flatten lo appiattisce a 8192 elementi.
    # Dropout(0.5) e' il principale regolarizzatore.
    # Sigmoid finale -> probabilita' di classe defect in [0, 1].
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    return model, conv_base


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def build_generators(data_dir):
    """
    Costruisce i generator per train e test.
    Train: rescale + data augmentation.
    Test:  rescale soltanto (le metriche devono essere riproducibili).
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,                # pixel [0,255] -> [0,1] (stabilita' del training)
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="binary",             # genera label 0/1 (richiesto da binary_crossentropy)
    )

    # shuffle=False sul test set: senza, y_true e y_pred non si allineerebbero.
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    return train_generator, test_generator


# ---------------------------------------------------------------------------
# Training a due fasi
# ---------------------------------------------------------------------------

def train_two_phases(model, conv_base, train_gen, test_gen):
    """
    Fase 1: solo la testa apprende (backbone congelato).
    Fase 2: si sbloccano gli ultimi 4 layer di VGG16 con LR molto basso.
    """
    steps_per_epoch = train_gen.samples // BATCH_SIZE       # 800 / 20 = 40
    validation_steps = test_gen.samples // BATCH_SIZE       # 200 / 20 = 10

    # === FASE 1 ===
    print("--- FASE 1: training della testa di classificazione ---")
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=2e-5),
        loss="binary_crossentropy",       # loss standard per output sigmoide binario
        metrics=["accuracy"],
    )
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS_PHASE1,
        validation_data=test_gen,
        validation_steps=validation_steps,
    )

    # === FASE 2 - fine-tuning ===
    print("--- FASE 2: fine-tuning degli ultimi blocchi conv ---")
    conv_base.trainable = True
    # Ricongela tutti i layer TRANNE gli ultimi 4. I layer iniziali codificano
    # feature generiche (bordi, texture) che sono utili per qualunque task,
    # mentre i layer profondi codificano feature di alto livello che vanno
    # adattate al dominio specifico dei difetti.
    for layer in conv_base.layers[:-4]:
        layer.trainable = False

    print("Ricompilazione con learning rate molto basso (1e-5).")
    # LR=1e-5 (meta' della fase 1): un LR alto qui distruggerebbe i pesi
    # pre-allenati, vanificando il transfer learning.
    # ATTENZIONE: model.compile() va richiamato dopo aver cambiato .trainable.
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history_fine = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        # epochs e' il numero TOTALE di epoche raggiunte alla fine, non quante
        # aggiungerne. initial_epoch=15 fa partire il contatore da 15.
        epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        initial_epoch=EPOCHS_PHASE1,
        validation_data=test_gen,
        validation_steps=validation_steps,
    )
    return history, history_fine


# ---------------------------------------------------------------------------
# Plot della history
# ---------------------------------------------------------------------------

def plot_history(history, history_fine):
    """Plot accuracy e loss; linea rossa = inizio fine-tuning."""
    acc = history.history["accuracy"] + history_fine.history["accuracy"]
    val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
    loss = history.history["loss"] + history_fine.history["loss"]
    val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(x=EPOCHS_PHASE1 - 1, color="r", linestyle="--",
                label="Inizio fine-tuning")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(x=EPOCHS_PHASE1 - 1, color="r", linestyle="--",
                label="Inizio fine-tuning")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = "data"

    if not os.path.isdir(os.path.join(DATA_DIR, "train")):
        print("Errore: dataset non trovato. Esegui prima 01_data_generation.py")
        exit()

    model, conv_base = build_model()
    model.summary()

    train_gen, test_gen = build_generators(DATA_DIR)
    history, history_fine = train_two_phases(model, conv_base, train_gen, test_gen)

    model.save(MODEL_SAVE_PATH)
    print(f"Modello salvato in: {MODEL_SAVE_PATH}")

    plot_history(history, history_fine)
