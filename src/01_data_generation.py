"""
MODULO 1 - DATA ACQUISITION
Genera un dataset sintetico di piastre metalliche con due classi:
    - good   : superficie uniforme senza difetti
    - defect : superficie con uno fra 4 tipi di difetto (graffio, foro,
               macchia semi-trasparente, crepa)

La scelta del dataset sintetico (vs MVTec AD) e' motivata da:
    - riproducibilita' assoluta (no download esterni)
    - controllo esatto della distribuzione delle classi e dei tipi di difetto
"""

import os
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Generazione di singole immagini
# ---------------------------------------------------------------------------

def create_good_plate(img_size=(150, 150)):
    """Crea una piastra "good": grigio chiaro uniforme + leggero rumore."""
    # Sfondo grigio chiaro (livello 180/255) che simula superficie metallica.
    img = np.full((*img_size, 3), 180, dtype=np.uint8)

    # Rumore gaussiano per simulare la microtexture metallica.
    # NB: dtype=int8 (non uint8) perche' il rumore puo' essere negativo.
    noise = np.zeros_like(img, dtype=np.int8)
    cv2.randn(noise, 0, 8)  # media 0, deviazione standard 8

    # cv2.add usa aritmetica saturata: i valori oltre [0,255] vengono clippati
    # invece di andare in overflow.
    img = cv2.add(img, noise, dtype=cv2.CV_8UC3)
    return img


def create_defect_plate(img_size=(150, 150)):
    """Crea una piastra "defect": good + un difetto scelto casualmente."""
    img = create_good_plate(img_size)
    defect_type = np.random.randint(0, 4)  # 4 tipi equiprobabili

    if defect_type == 0:
        # GRAFFIO: 1-2 linee scure di direzione casuale.
        n_scratches = np.random.randint(1, 3)
        for _ in range(n_scratches):
            x1 = np.random.randint(0, img_size[0])
            y1 = np.random.randint(0, img_size[1])
            x2 = np.random.randint(0, img_size[0])
            y2 = np.random.randint(0, img_size[1])
            cv2.line(img, (x1, y1), (x2, y2), (40, 40, 40), 2)

    elif defect_type == 1:
        # FORO: cerchio quasi nero pieno (thickness=-1 = riempi).
        # Margine di 20px dai bordi per evitare difetti troncati.
        cx = np.random.randint(20, img_size[0] - 20)
        cy = np.random.randint(20, img_size[1] - 20)
        radius = np.random.randint(5, 12)
        cv2.circle(img, (cx, cy), radius, (15, 15, 15), -1)

    elif defect_type == 2:
        # MACCHIA: cerchio grigio scuro semi-trasparente.
        # Il difetto piu' difficile per HOG perche' il contrasto e' basso
        # (gradiente debole). Lo otteniamo fondendo overlay e originale al 60/40.
        cx = np.random.randint(20, img_size[0] - 20)
        cy = np.random.randint(20, img_size[1] - 20)
        radius = np.random.randint(8, 20)
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), radius, (90, 90, 90), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    else:
        # CREPA: spezzata di 4-7 segmenti corti (random walk).
        x, y = np.random.randint(20, img_size[0] - 20, size=2)
        n_segments = np.random.randint(4, 8)
        for _ in range(n_segments):
            dx = np.random.randint(-15, 15)
            dy = np.random.randint(-15, 15)
            # np.clip impedisce alla crepa di uscire dai bordi.
            x2 = np.clip(x + dx, 0, img_size[0] - 1)
            y2 = np.clip(y + dy, 0, img_size[1] - 1)
            cv2.line(img, (x, y), (x2, y2), (30, 30, 30), 1)
            x, y = x2, y2  # il prossimo segmento parte dove finiva questo

    return img


# ---------------------------------------------------------------------------
# Generazione del dataset completo
# ---------------------------------------------------------------------------

def generate_dataset(base_dir, n_train=400, n_test=100, img_size=(150, 150)):
    """
    Crea la struttura di cartelle e salva i campioni.

    Struttura risultante (compatibile con flow_from_directory di Keras):
        base_dir/
            train/{good,defect}/  n_train immagini per classe
            test/{good,defect}/   n_test  immagini per classe
    """
    print("Generazione del dataset sintetico in corso...")

    splits = {"train": n_train, "test": n_test}
    generators = {"good": create_good_plate, "defect": create_defect_plate}

    for split_name, n_samples in splits.items():
        for class_name, gen_func in generators.items():
            out_dir = os.path.join(base_dir, split_name, class_name)
            # mkdir parents=True crea anche le cartelle padre mancanti;
            # exist_ok=True evita errore se esiste gia' (script ri-eseguibile).
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            for i in range(n_samples):
                img = gen_func(img_size)
                # Padding a 4 cifre per mantenere ordinamento alfabetico
                # coerente con quello numerico (good_0009 < good_0010).
                filename = f"{class_name}_{i:04d}.png"
                cv2.imwrite(os.path.join(out_dir, filename), img)

            print(f"  [{split_name}/{class_name}] {n_samples} immagini salvate")

    print("Generazione del dataset completata.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Seed fissato per riproducibilita': rieseguendo lo script si ottiene
    # esattamente lo stesso dataset.
    np.random.seed(42)

    DATA_DIR = "data"
    if not os.path.isdir(os.path.join(DATA_DIR, "train", "good")):
        generate_dataset(DATA_DIR, n_train=400, n_test=100, img_size=(150, 150))
    else:
        print("Il dataset esiste gia'. Salto la generazione.")
