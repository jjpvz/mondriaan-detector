import joblib
import cv2 as cv
import numpy as np
import pandas as pd
from features import preprocess_image, maskColor, prepare_features
import os

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelbestand niet gevonden: {model_path}")
    clf = joblib.load(model_path)
    return clf

def capture_from_webcam() -> np.ndarray:
    """Open de webcam en maak een foto zodra je op SPATIE drukt. q/Esc = stoppen."""
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kon de webcam niet openen (index 0).")

    print("Webcam geopend. Druk op [SPATIE] om een foto te maken. Druk op [q] of [Esc] om te stoppen.")

    frame_to_return = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kon geen frame van de webcam lezen.")
            break

        # (optioneel) Overlay instructie
        cv.putText(frame, "Press SPACE to capture, q/Esc to quit",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow("Webcam - Capture", frame)
        key = cv.waitKey(1) & 0xFF

        if key == 32:  # SPATIE
            frame_to_return = frame.copy()
            print("Foto gemaakt!")
            break
        elif key in (ord('q'), 27):  # q of Esc
            break

    cap.release()
    cv.destroyAllWindows()
    return frame_to_return

def processing_image(frame):
    processed_features = []
        
    pre_img = preprocess_image(frame)
        
    mask_upper = maskColor(pre_img, 0, 11, False)
    mask_lower = maskColor(pre_img, 169, 180, False)
    red_mask = cv.bitwise_or(mask_upper, mask_lower)
    yellow_mask = maskColor(pre_img, 22, 38, True)
    blue_mask = maskColor(pre_img, 90, 130, True)
    
    features = prepare_features(pre_img, red_mask, yellow_mask, blue_mask)
    processed_features.append(features)
        
    return processed_features
