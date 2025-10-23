import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from settings import MODEL_PATH, TEST_IMAGE_PATH
from main_helpers import capture_from_webcam, load_model, processing_image, show_prediction_window, show_input_selection_window
from processing_tools import resize_image, display_image_cv

clf = load_model(MODEL_PATH)

files_selected = []
path = Path(r"C:\GIT\mondriaan-detector\data")
for img in path.glob("*.JPG"):
    files_selected.append(img)

for img in files_selected:
    frame = cv.imread(str(img))
    image_label = img.stem
    if frame is None:
        print(f"Kon afbeelding niet laden: {img}")
        exit(0)
    resized_frame = resize_image(frame, 1920, 1080)
    rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
    #display_image_cv(rgb_frame, "test image")  
    data = processing_image(rgb_frame)
    df = pd.DataFrame(data)
    print(df.head())
    pred = clf.predict(df)
    prob = clf.predict_proba(df)
    pred_label = clf.classes_[np.argmax(prob[0])]
    max_p = float(np.max(prob[0]))
    print(f"{image_label} Voorspelling: {pred_label}, Zekerheid: {max_p:.2f}")
    show_prediction_window(resized_frame, pred_label, max_p)
    
