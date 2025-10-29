import cv2 as cv
import numpy as np
import pandas as pd
from main_helpers import capture_from_webcam, load_model, processing_image, show_prediction_window, show_input_selection_window, resource_path
from processing_tools import resize_image, display_image_cv
from test_tools import Timer

"""
Main script for Mondriaan detector
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

This script allows the user to capture an image from the webcam or select an image file,
processes the image to extract features, and uses a pre-trained SVM model to predict which 
type of Mondriaan painting the image contains or if it is not a Mondriaan painting.

To use this script:
make sure the name of the model file is the same as in MODEL_FILE variable below
make sure the model file is in the same folder as the src folder or give full path in MODEL_PATH variable below
Run the script, a window will appear to choose between webcam or file input
If webcam is chosen, a window will open showing the webcam feed, press space to capture an image
If file is chosen, a file dialog will open to select an image file
The script will then process the image and display the prediction result in a new window
"""

# Define model file name
MODEL_FILE = "mondriaan_svm_model.joblib"

# Get the full model path
MODEL_PATH = resource_path(MODEL_FILE)

# load the model from model path if not found, raise error
clf = load_model(MODEL_PATH)

# Show input selection window
use_camera, image_path = show_input_selection_window()

# Check if user cancelled
if use_camera is None:
    print("Programma geannuleerd door gebruiker.")
    exit(0)

# capture image from webcam or load selected image
if use_camera:
    frame = capture_from_webcam()
    if frame is None:
        print("Geen foto gemaakt. Programma beëindigd.")
        exit(0)
else:
    frame = cv.imread(image_path)
    if frame is None:
        print(f"Kon afbeelding niet laden: {image_path}")
        exit(0)


# resize the image to 1920x1080 for a standard input size
resized_frame = resize_image(frame, 1920, 1080)
# convert BGR to RGB as model was trained on RGB images
rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
# process the image to extract features and prepare for prediction
data = processing_image(rgb_frame)
# create a dataframe from the processed data
df = pd.DataFrame(data)
print(df.head())



# Extract color coverage feature for black image filtering
blue_pct = df['blue_pct'].iloc[0]
red_pct = df['red_pct'].iloc[0]
yellow_pct = df['yellow_pct'].iloc[0]
color_coverage = red_pct + yellow_pct + blue_pct
print(f"Kleur dekking: {color_coverage:.2f}%")

# make prediction using the loaded model
pred = clf.predict(df)

# get prediction probabilities, for confidence level
prob = clf.predict_proba(df)

pred_label = clf.classes_[np.argmax(prob[0])]
max_p = float(np.max(prob[0]))

# set prediction to temporary prediction for further processing
temp_pred = pred_label

# Check for confidence threshold
if temp_pred != "niet_mondriaan":
    if max_p < 0.6:
        temp_pred = "niet_mondriaan"
    elif max_p > 0.6 and max_p < 0.8:
        temp_pred = "mondriaan_onbekend"
    else:
         # Apply additional rules based on color coverage for each Mondriaan class
        match temp_pred:
                case "mondriaan1":
                    if color_coverage < 5.0 or color_coverage > 20.0:
                        temp_pred = "niet_mondriaan"
                        max_p = 0
                    else:
                        temp_pred = temp_pred
                case "mondriaan2":
                    if color_coverage < 35.0 or color_coverage > 50.0:
                        temp_pred = "niet_mondriaan"
                        max_p = 0
                    else:
                        temp_pred = temp_pred
                case "mondriaan3":
                    if color_coverage < 7.0 or color_coverage > 25.0:
                        temp_pred = "niet_mondriaan"
                        max_p = 0
                    else:
                        temp_pred = temp_pred
                case "mondriaan4":
                    if color_coverage < 5.0 or color_coverage > 20.0:
                        temp_pred = "niet_mondriaan"
                        max_p = 0
                    else:
                        temp_pred = temp_pred
                case _:
                    temp_pred = temp_pred            

final_pred = temp_pred

print(f"Voorspelling: {final_pred}")
print(f"Zekerheid voor ({pred_label}): {max_p*100:.2f}%")

show_prediction_window(resized_frame, final_pred, max_p)


