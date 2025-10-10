import cv2 as cv
import numpy as np
import pandas as pd
from settings import MODEL_PATH, TEST_IMAGE_PATH
from capture_def import capture_from_webcam, load_model, processing_image
from features import resize_image

# make a MODEL_PATH and TEST_IMAGE_PATH in settings.py
# MODEL_PATH = "mondriaan_svm_model.joblib"
# TEST_IMAGE_PATH = folder_path / "mondriaan3 (1).JPG" 
# Test image can be changed to any image you want to test
# joblib model must be in the same folder as this script

# change to True to use webcam, False to use test image
Camera_activate = False

clf = load_model(MODEL_PATH)

if Camera_activate:
    frame = capture_from_webcam()
    if frame is None:
        print("Geen foto gemaakt. Programma beëindigd.")
        exit(0)
else:
    frame = cv.imread(str(TEST_IMAGE_PATH))


resized_frame = resize_image(frame, 1920, 1080)
rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)

data = processing_image(rgb_frame)
df = pd.DataFrame(data)

pred = clf.predict(df)

print(f"Voorspelling: {pred}")

