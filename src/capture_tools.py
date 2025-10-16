import joblib
import cv2 as cv
import numpy as np
import pandas as pd
from processing_tools import preprocess_image, maskColor, prepare_features
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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


# Display result in a GUI window
def show_prediction_window(image, prediction, probability):
    # Create main window
    root = tk.Tk()
    root.title("Mondriaan Detector - Resultaat")
    root.geometry("800x600")
    root.configure(bg='#f0f0f0')
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Mondriaan Detector", 
                           font=('Arial', 18, 'bold'))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
    
    # Display image (resize for display)
    display_image = cv.resize(image, (300, 225))  # 4:3 aspect ratio
    display_image_rgb = cv.cvtColor(display_image, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(display_image_rgb)
    photo = ImageTk.PhotoImage(pil_image)
    
    image_label = ttk.Label(main_frame, image=photo)
    image_label.grid(row=1, column=0, padx=(0, 20), pady=(0, 20))
    
    # Result frame
    result_frame = ttk.LabelFrame(main_frame, text="Resultaat", padding="15")
    result_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
    
    # Prediction result
    prediction_text = "✓ Dit is een Mondriaan!" if prediction[0] == 1 else "✗ Dit is geen Mondriaan"
    if prediction == "mondriaan1":
        prediction_text = "✓ Dit is Mondriaan 1!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {max(probability[0]) * 100:.2f}%"
    elif prediction == "mondriaan2":
        prediction_text = "✓ Dit is Mondriaan 2!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {max(probability[0]) * 100:.2f}%"
    elif prediction == "mondriaan3":
        prediction_text = "✓ Dit is Mondriaan 3!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {max(probability[0]) * 100:.2f}%"
    elif prediction == "mondriaan4":
        prediction_text = "✓ Dit is Mondriaan 4!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {max(probability[0]) * 100:.2f}%"
    elif prediction == "niet_mondriaan":
        prediction_text = "✗ Dit is geen Mondriaan"
        prediction_color = '#DC143C'
        confidence_text = f"Zekerheid: {max(probability[0]) * 100:.2f}%"
    elif prediction == "mondriaan_onbekend":
        prediction_text = "✗ Te lage zekerheid, maak nieuwe foto"
        prediction_color = '#FFA500'
        confidence_text = f"Zekerheid: {max(probability[0]) * 100:.2f}%"

    
    prediction_label = ttk.Label(result_frame, text=prediction_text, 
                                font=('Arial', 14, 'bold'),
                                foreground=prediction_color)
    prediction_label.grid(row=0, column=0, pady=(0, 10))
    
    # Confidence or additional info

    confidence_label = ttk.Label(result_frame, text=confidence_text, 
                                font=('Arial', 10))
    confidence_label.grid(row=1, column=0, pady=(0, 20))
    
    # Close button
    close_button = ttk.Button(result_frame, text="Sluiten", 
                             command=root.destroy,
                             style='Accent.TButton')
    close_button.grid(row=2, column=0, pady=(10, 0))
    
    # Keep reference to photo to prevent garbage collection
    image_label.photo = photo
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the GUI
    root.mainloop()

    return