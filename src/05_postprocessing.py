"""
MODULO 5 - POST-PROCESSING (localizzazione del difetto)
Pipeline classica di image processing per localizzare il difetto su
un'immagine gia' classificata come 'defect':

    1. grayscale + median blur leggero
    2. binarizzazione di Otsu (con THRESH_BINARY_INV: difetti scuri -> bianchi)
    3. morfologia: opening (rimuove rumore) + closing (riempie buchi)
    4. connectedComponents -> bounding box delle regioni di area >= min_area
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def localize_defect(img_bgr, min_area=20):
    """
    Localizza i difetti in img_bgr e ritorna (boxes, mask).
        boxes : lista di tuple (x, y, w, h) - bounding box dei difetti
        mask  : maschera binaria finale dopo morfologia

    min_area filtra le regioni microscopiche (rumore residuo).
    """
    # Step 1: grayscale + blur leggero (kernel 3, non 5 come nel modulo 2,
    # per non eliminare difetti veri che potrebbero essere piccoli).
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 3)

    # Step 2: Otsu thresholding.
    # thresh=0 viene ignorato quando si usa THRESH_OTSU (la soglia e' auto).
    # THRESH_BINARY_INV inverte: i difetti SCURI diventano BIANCHI nella mask,
    # convenzione necessaria perche' connectedComponents cerca foreground bianco.
    _, binary = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 3: morfologia per pulire la maschera.
    # OPEN  (erosione+dilatazione): rimuove pixel bianchi isolati (rumore).
    # CLOSE (dilatazione+erosione): riempie buchi interni alle regioni.
    # iterations=2 sul close ripete l'operazione per buchi piu' grandi.
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: connectedComponents con statistiche per ogni regione.
    # connectivity=8 considera connessi anche i pixel diagonali.
    # stats[i, :] = [LEFT, TOP, WIDTH, HEIGHT, AREA].
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    boxes = []
    # Si parte da 1 perche' l'etichetta 0 e' sempre lo sfondo.
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            boxes.append((x, y, w, h))

    return boxes, closed


def draw_boxes(img_bgr, boxes, color=(0, 0, 255), thickness=2):
    """Disegna i bounding box (default rosso in BGR) su una copia dell'immagine."""
    # .copy() evita di modificare l'originale (side effect indesiderato).
    out = img_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


# ---------------------------------------------------------------------------
# Demo visiva
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import glob

    # I primi 4 campioni difettosi del test set.
    sample_paths = sorted(glob.glob("data/test/defect/defect_00*.png"))[:4]
    if not sample_paths:
        print("Nessun campione trovato. Esegui prima 01_data_generation.py.")
        exit()

    fig, axes = plt.subplots(len(sample_paths), 3,
                             figsize=(12, 3 * len(sample_paths)))
    if len(sample_paths) == 1:
        axes = np.array([axes])

    for row, path in enumerate(sample_paths):
        img = cv2.imread(path)
        boxes, mask = localize_defect(img, min_area=20)
        drawn = draw_boxes(img, boxes)

        axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title("Originale")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("Mask binaria")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title(f"Defect bbox ({len(boxes)} regioni)")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.show()
