import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from settings import MODEL_PATH, TEST_IMAGE_PATH
from main_helpers import capture_from_webcam, load_model, processing_image, show_directory_selection_window, show_prediction_window, show_input_selection_window
from processing_tools import resize_image, display_image_cv

# Set this true to run full test, otherwise only ML model
test_full = True

# set this true to run a single test image, otherwise run all images in test set folder
single_test_image = False

# set this true if image label is the name of the image, this works never with single_test_image = True
# to meet this condition the images in the folder must be named like: mondriaan1.JPG, mondriaan2 .JPG, niet_mondriaan (5).JPG etc.
image_label_equal = False

# set to None to disable automatic close
automatic_close_ms = 2000  # milliseconds

if single_test_image == True and image_label_equal == True:
    print("Fout: single_test_image en image_label_equal kunnen niet beide waar zijn.")
    exit(0)

clf = load_model(MODEL_PATH)

times = 0
wrong_times = 0

# single image test
if single_test_image:
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
    # make prediction using the loaded model
    
    if test_full:
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

        show_prediction_window(resized_frame, final_pred, max_p, automatic_close_ms)

    else:
        # make prediction using the loaded model
        pred = clf.predict(df)
        # get prediction probabilities, for confidence level
        prob = clf.predict_proba(df)
        pred_label = clf.classes_[np.argmax(prob[0])]
        max_p = float(np.max(prob[0]))

        print(f"Voorspelling: {pred_label}")
        print(f"Zekerheid voor ({pred_label}): {max_p*100:.2f}%")

        show_prediction_window(resized_frame, pred_label, max_p)
    exit(0)

# multi image test
else:
    files_selected = []
    path_str = show_directory_selection_window()
    if path_str is None:
        print("Geen map geselecteerd. Programma wordt afgesloten.")
        exit(0)
    path = Path(path_str)
    for img in path.glob("*.JPG"):
        files_selected.append(img)

    for img in files_selected:
        frame = cv.imread(str(img))  
        if image_label_equal == True:
            image_label = img.stem.split(" (")[0]
        
        if frame is None:
            print(f"Kon afbeelding niet laden: {img}")
            exit(0)
        
        resized_frame = resize_image(frame, 1920, 1080)

        rgb_frame = cv.cvtColor(resized_frame, cv.COLOR_BGR2RGB)
        
        display_image_cv(rgb_frame, "input image", automatic_close_ms)  
        
        data = processing_image(rgb_frame)
        
        df = pd.DataFrame(data)
        print(df.head())
        
        if test_full == True:
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

        else:
            # make prediction using the loaded model
            pred = clf.predict(df)
            # get prediction probabilities, for confidence level
            prob = clf.predict_proba(df)
            pred_label = clf.classes_[np.argmax(prob[0])]
            max_p = float(np.max(prob[0]))
            final_pred = pred_label
        
        print(f"Voorspelling: {final_pred}")
        print(f"Zekerheid voor ({final_pred}): {max_p*100:.2f}%")

        if image_label_equal == True:
            if image_label == "mondriaan1" or image_label == "mondriaan2" or image_label == "mondriaan3" or image_label == "mondriaan4" or image_label == "niet_mondriaan" or image_label == "mondriaan_onbekend":
                if image_label == final_pred:
                    times += 1  
                else:
                    wrong_times += 1
            else:
                print("Waarschuwing: image_label is geen geldige klasse voor vergelijking, programma wordt afgebroken.")
                exit(0)
        
        show_prediction_window(resized_frame, final_pred, max_p, automatic_close_ms)


    if image_label_equal == True:
        print(f"Correcte voorspellingen: {times}, Foute voorspellingen: {wrong_times}")
        print(f"Nauwkeurigheid: {(times / (times + wrong_times))*100:.2f}%")
    exit(0)