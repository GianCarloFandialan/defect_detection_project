"""
MODULO 2 - PREPROCESSING
Pipeline di 3 stadi applicata alle immagini prima della feature extraction:
    1. gray-world white balance  -> rimuove dominanti cromatiche
    2. median blur               -> riduce rumore salt-and-pepper
    3. equalizzazione HSV (V)    -> migliora il contrasto senza alterare i colori

L'ordine WB -> denoise -> equalize non e' casuale:
    - WB prima, perche' lavora sulle medie dei canali originali.
    - Denoise prima dell'equalizzazione, altrimenti quest'ultima amplificherebbe
      anche il rumore.
    - Equalizzazione per ultima, come step di enhancement finale.
"""

import cv2
import numpy as np


def gray_world_white_balance(img):
    """
    Bilanciamento del bianco con assunzione Gray-World.

    Ipotesi: in una scena media la media R/G/B dovrebbe essere uguale (grigio
    neutro). Se un canale ha media maggiore, viene attenuato; se inferiore,
    amplificato, per riportare tutti e tre alla stessa media globale.
    """
    # Cast a float32 OBBLIGATORIO: le moltiplicazioni successive andrebbero
    # in overflow su uint8 (es. 180 * 1.5 = 270 -> wrap-around a 14).
    img = img.astype(np.float32)

    # OpenCV usa ordine BGR (non RGB): canale 0=B, 1=G, 2=R.
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Scalatura: il fattore (avg_gray / avg_canale) e' >1 se il canale e' sotto
    # la media (lo amplifica) o <1 se sopra (lo attenua).
    img[:, :, 0] *= (avg_gray / avg_b)
    img[:, :, 1] *= (avg_gray / avg_g)
    img[:, :, 2] *= (avg_gray / avg_r)

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def denoise(img, ksize=5):
    """
    Filtro mediano contro rumore salt-and-pepper.

    A differenza del Gaussian blur, il mediano preserva i bordi perche'
    non e' influenzato dagli outlier (un pixel anomalo non sposta la mediana).
    ksize deve essere dispari (richiesto da OpenCV per avere pixel centrale).
    """
    return cv2.medianBlur(img, ksize)


def enhance_contrast(img):
    """
    Equalizza l'istogramma del canale V (Value) in HSV.

    Lavorare in HSV invece che sui 3 canali BGR separatamente evita le
    distorsioni cromatiche: si tocca solo la luminosita', i colori restano.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)             # solo il canale luminosita'
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


def preprocess_pipeline(img):
    """Sequenza completa: white balance -> denoise -> equalizzazione."""
    img = gray_world_white_balance(img)
    img = denoise(img, ksize=5)
    img = enhance_contrast(img)
    return img


# ---------------------------------------------------------------------------
# Demo visiva (eseguibile come script principale)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    sample_path = os.path.join("data", "train", "defect", "defect_0000.png")
    if not os.path.exists(sample_path):
        print(f"File non trovato: {sample_path}")
        print("Esegui prima 01_data_generation.py per creare il dataset.")
    else:
        original = cv2.imread(sample_path)
        processed = preprocess_pipeline(original)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # matplotlib si aspetta RGB, OpenCV usa BGR: serve la conversione.
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Originale")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        plt.title("Dopo preprocessing")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
