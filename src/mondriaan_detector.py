import cv2 as cv
import numpy as np
import pandas as pd
from settings import MODEL_PATH, TEST_IMAGE_PATH
from capture_tools import capture_from_webcam, load_model, processing_image, show_prediction_window
from processing_tools import resize_image

# make a MODEL_PATH and TEST_IMAGE_PATH in settings.py
# MODEL_PATH = "mondriaan_svm_model.joblib"
# TEST_IMAGE_PATH = folder_path / "mondriaan3 (1).JPG" 
# Test image can be changed to any image you want to test
# joblib model must be in the same folder as the src folder or give full path

# change to True to use webcam, False to use test image
Camera_activate = True

# load the model from model path if not found, raise error
clf = load_model(MODEL_PATH)

# capture image from webcam or load test image
if Camera_activate:
    frame = capture_from_webcam()
    if frame is None:
        print("Geen foto gemaakt. Programma beëindigd.")
        exit(0)
else:
    frame = cv.imread(str(TEST_IMAGE_PATH))
    if frame is None:
        print("Geen foto gevonden. Programma beëindigd.")
        exit(0)

# resize the image to 1920x1080 for a standard input size
resized_frame = resize_image(frame, 1920, 1080)
# convert BGR to RGB as model was trained on RGB images
rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
# process the image to extract features and prepare for prediction
data = processing_image(rgb_frame)
# create a dataframe from the processed data
df = pd.DataFrame(data)
# make prediction using the loaded model
pred = clf.predict(df)
# get prediction probabilities, for confidence level
prob = clf.predict_proba(df)

#if max(prob[0]) < 0.6:
#    pred = "mondriaan_onbekend"

pred_label = clf.classes_[np.argmax(prob[0])]
max_p = float(np.max(prob[0]))
if max_p >= 0.8:
    final_pred = pred_label
else:
    final_pred = "mondriaan_onbekend"

print(f"Voorspelling: {final_pred}")
print(f"Zekerheid voor ({pred_label}): {max_p*100:.2f}%")

#print(f"Voorspelling: {pred}")

#print(f"Zekerheid voor {pred}: %.2f%%" % (max(prob[0]) * 100))
show_prediction_window(resized_frame, final_pred, prob)

