import joblib
import cv2 as cv
import numpy as np
import pandas as pd
from processing_tools import preprocess_image, mask_feature_color, prepare_features, display_image_cv
from pathlib import Path
import os, sys
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

"""
Main helper functions for Mondriaan detector
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

This file contains helper functions for the main script, including:
- loading the model
- capturing image from webcam
- processing the image to extract features
- displaying the prediction result in a GUI window
"""


# function to load the model
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelbestand niet gevonden: {model_path}")
    clf = joblib.load(model_path)
    return clf

# function to get resource path
def resource_path(relative_path):
    """Geeft het juiste pad naar een resource, ongeacht of het script of exe draait."""
    try:
        base_path = sys._MEIPASS  # runtime pad (PyInstaller)
    except AttributeError:
        base_path = os.path.abspath(".")  # dev pad (Python)
    return os.path.join(base_path, relative_path)

# function to capture image from webcam
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

        
        cv.imshow("Webcam - Druk op SPATIE om foto te maken, Q/ESC om te stoppen", frame)
        key = cv.waitKey(1) & 0xFF

        if key == 32:  # SPATIE
            frame_to_return = frame.copy()
            print("Foto gemaakt!")
            break
        elif key in (ord('q'), 27):  # q or Esc
            break

    cap.release()
    cv.destroyAllWindows()
    return frame_to_return

# function to process image and extract features
def processing_image(frame):
    """
    function to process an image completly and get the features for prediction
    this function does:
    1. preprocess the image
    2. create color masks for red, yellow and blue
    3. extract features from the preprocessed image and color masks
        
    Args:
        frame: input image (numpy array)
    Returns:
        processed_features: list of extracted features for prediction
    """
    
    
    processed_features = []
    # kernel for morphological operations
    k = cv.getStructuringElement(cv.MORPH_RECT, (15,15))
    # preprocess image for feature extraction    
    pre_img = preprocess_image(frame)

    # uncomment to display preprocessed image
    #display_image_cv(pre_img, "Voorverwerkte afbeelding")

    # create color masks    
    red_mask = mask_feature_color(pre_img, [(0, 11), (169, 180)], 110, 70)
    yellow_mask = mask_feature_color(pre_img, [(18, 38)], 70, 90)
    blue_mask_temp = mask_feature_color(pre_img, [(105, 130)], 100, 60)
    # apply morphological opening to extra clean up blue mask
    blue_mask = cv.morphologyEx(blue_mask_temp, cv.MORPH_OPEN, k, iterations=1)

    # uncomment to display color masks
    #display_image_cv(red_mask, "Rood masker")
    #display_image_cv(yellow_mask, "Geel masker")
    #display_image_cv(blue_mask, "Blauw masker")

    # extract features and add image_id and label
    features = prepare_features(pre_img, red_mask, yellow_mask, blue_mask)
    processed_features.append(features)
        
    return processed_features


# Display result in a GUI window
def show_prediction_window(image, prediction, probability, auto_close_ms=None):
    """
    shows a GUI window with the image, prediction and probability
    Args:
        image: input image (numpy array)
        prediction: predicted label (str)
        probability: probability of the prediction (float)
        auto_close_ms: time in milliseconds to auto-close the window (int or None)
    Returns: None
    """
    # Create main window
    root = tk.Tk()
    root.title("Mondriaan Detector - Resultaat")
    root.geometry("800x400")
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
    prediction_text = "✓ Dit is een Mondriaan!" 
    if prediction == "mondriaan1":
        prediction_text = "✓ Dit is Mondriaan 1!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "mondriaan2":
        prediction_text = "✓ Dit is Mondriaan 2!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "mondriaan3":
        prediction_text = "✓ Dit is Mondriaan 3!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "mondriaan4":
        prediction_text = "✓ Dit is Mondriaan 4!"
        prediction_color = '#2E8B57'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "niet_mondriaan":
        prediction_text = "✗ Dit is geen Mondriaan"
        prediction_color = '#DC143C'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"
    elif prediction == "mondriaan_onbekend":
        prediction_text = "✗ Te lage zekerheid, maak nieuwe foto"
        prediction_color = '#FFA500'
        confidence_text = f"Zekerheid: {probability * 100:.2f}%"

    
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
    
    # Set focus to close button and bind Enter key
    close_button.focus_set()
    root.bind('<Return>', lambda event: root.destroy())
    root.bind('<KP_Enter>', lambda event: root.destroy())
    
    # Auto-close functionality
    if auto_close_ms is not None:
        root.after(auto_close_ms, root.destroy)
    
    # Start the GUI
    root.mainloop()

    return

# function to show input selection window, for camera or file
def show_input_selection_window():
    """
    shows a GUI window to select input method (camera or file)
    args: None
    Returns: tuple (use_camera: bool, image_path: str) or (None, None) if cancelled
    """
    result = {'use_camera': None, 'image_path': None, 'cancelled': True}
    
    def on_camera_selected():
        result['use_camera'] = True
        result['cancelled'] = False
        root.quit()
    
    def on_file_selected():
        # Open file dialog
        # determine project root: one level above this src file
        try:
            project_root = Path(__file__).resolve().parent.parent
        except Exception:
            project_root = None

        initial_dir = str(project_root) if project_root and project_root.exists() else os.path.expanduser("~")

        file_path = filedialog.askopenfilename(
            title="Selecteer een afbeelding",
            filetypes=[
                ("Afbeeldingen", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff *.tif"),
                ("Alle bestanden", "*.*")
            ],
            initialdir=initial_dir
        )
        
        if file_path:
            result['use_camera'] = False
            result['image_path'] = file_path
            result['cancelled'] = False
            root.quit()
    
    def on_cancel():
        result['cancelled'] = True
        root.quit()
    
    # Create main window
    root = tk.Tk()
    root.title("Mondriaan Detector - Input Selectie")
    root.geometry("500x500")
    root.configure(bg='#f0f0f0')
    root.resizable(False, False)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (250)
    y = (root.winfo_screenheight() // 2) - (200)
    root.geometry(f"500x400+{x}+{y}")
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="30")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Mondriaan Detector", 
                           font=('Arial', 20, 'bold'))
    title_label.grid(row=0, column=0, pady=(0, 10))
    
    # Subtitle
    subtitle_label = ttk.Label(main_frame, text="Kies uw input methode", 
                              font=('Arial', 12))
    subtitle_label.grid(row=1, column=0, pady=(0, 30))
    
    # Camera option
    camera_frame = ttk.LabelFrame(main_frame, text="Camera", padding="20")
    camera_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
    main_frame.columnconfigure(0, weight=1)
    camera_frame.columnconfigure(0, weight=1)
    
    camera_desc = ttk.Label(camera_frame, text="Gebruik uw webcam om direct\neen foto te maken", 
                           font=('Arial', 10), justify='center')
    camera_desc.grid(row=0, column=0, pady=(0, 15))
    
    camera_button = ttk.Button(camera_frame, text="📷 Gebruik Camera", 
                              command=on_camera_selected,
                              style='Accent.TButton')
    camera_button.grid(row=1, column=0)
    
    # File option
    file_frame = ttk.LabelFrame(main_frame, text="Bestand", padding="20")
    file_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 30))
    file_frame.columnconfigure(0, weight=1)
    
    file_desc = ttk.Label(file_frame, text="Selecteer een afbeelding\nvan uw computer", 
                         font=('Arial', 10), justify='center')
    file_desc.grid(row=0, column=0, pady=(0, 15))
    
    file_button = ttk.Button(file_frame, text="📁 Selecteer Bestand", 
                            command=on_file_selected)
    file_button.grid(row=1, column=0)
    
    # Cancel button
    cancel_button = ttk.Button(main_frame, text="Annuleren", 
                              command=on_cancel)
    cancel_button.grid(row=4, column=0, pady=(10, 0))
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    
    # Start the GUI
    root.mainloop()
    root.destroy()
    
    if result['cancelled']:
        return None, None
    else:
        return result['use_camera'], result['image_path']

# function to show directory selection window
def show_directory_selection_window():
    """
    shows a GUI window to select a directory.
    args: None
    Returns: str (directory_path) or None if cancelled
    """
    result = {'directory_path': None, 'cancelled': True}
    
    def on_directory_selected():
        # Open directory dialog
        # determine project root: one level above this src file
        try:
            project_root = Path(__file__).resolve().parent.parent
        except Exception:
            project_root = None

        initial_dir = str(project_root) if project_root and project_root.exists() else os.path.expanduser("~")

        dir_path = filedialog.askdirectory(
            title="Selecteer een map",
            initialdir=initial_dir
        )
        
        if dir_path:
            result['directory_path'] = dir_path
            result['cancelled'] = False
            root.quit()
    
    def on_cancel():
        result['cancelled'] = True
        root.quit()
    
    # Create main window
    root = tk.Tk()
    root.title("Mondriaan Detector - Map Selectie")
    root.geometry("500x400")
    root.configure(bg='#f0f0f0')
    root.resizable(False, False)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (250)
    y = (root.winfo_screenheight() // 2) - (200)
    root.geometry(f"500x400+{x}+{y}")
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="30")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Configure grid weights
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)
    
    # Title
    title_label = ttk.Label(main_frame, text="Mondriaan Detector", 
                           font=('Arial', 20, 'bold'))
    title_label.grid(row=0, column=0, pady=(0, 10))
    
    # Subtitle
    subtitle_label = ttk.Label(main_frame, text="Selecteer een map", 
                              font=('Arial', 12))
    subtitle_label.grid(row=1, column=0, pady=(0, 30))
    
    # Directory selection option
    directory_frame = ttk.LabelFrame(main_frame, text="Map Selectie", padding="20")
    directory_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 30))
    main_frame.columnconfigure(0, weight=1)
    directory_frame.columnconfigure(0, weight=1)
    
    directory_desc = ttk.Label(directory_frame, text="Selecteer een map\nvan uw computer", 
                              font=('Arial', 10), justify='center')
    directory_desc.grid(row=0, column=0, pady=(0, 15))
    
    directory_button = ttk.Button(directory_frame, text="📁 Selecteer Map", 
                                 command=on_directory_selected,
                                 style='Accent.TButton')
    directory_button.grid(row=1, column=0)
    
    # Cancel button
    cancel_button = ttk.Button(main_frame, text="Annuleren", 
                              command=on_cancel)
    cancel_button.grid(row=3, column=0, pady=(10, 0))
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    
    # Start the GUI
    root.mainloop()
    root.destroy()
    
    if result['cancelled']:
        return None
    else:
        return result['directory_path']



