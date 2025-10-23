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

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelbestand niet gevonden: {model_path}")
    clf = joblib.load(model_path)
    return clf

def resource_path(relative_path):
    """Geeft het juiste pad naar een resource, ongeacht of het script of exe draait."""
    try:
        base_path = sys._MEIPASS  # runtime pad (PyInstaller)
    except AttributeError:
        base_path = os.path.abspath(".")  # dev pad (Python)
    return os.path.join(base_path, relative_path)

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

        # Toon het frame zonder overlay tekst, instructies staan in window titel
        cv.imshow("Webcam - Druk op SPATIE om foto te maken, Q/ESC om te stoppen", frame)
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
    k = cv.getStructuringElement(cv.MORPH_RECT, (15,15))
    # preprocess image for feature extraction    
    pre_img = preprocess_image(frame)

    #display_image_cv(pre_img, "Voorverwerkte afbeelding")

    # create color masks    
    red_mask = mask_feature_color(pre_img, [(0, 11), (169, 180)], 110, 70)
    yellow_mask = mask_feature_color(pre_img, [(18, 38)], 70, 90)
    blue_mask_temp = mask_feature_color(pre_img, [(105, 130)], 100, 60)
    blue_mask = cv.morphologyEx(blue_mask_temp, cv.MORPH_OPEN, k, iterations=1)

    #display_image_cv(red_mask, "Rood masker")
    #display_image_cv(yellow_mask, "Geel masker")
    #display_image_cv(blue_mask, "Blauw masker")

    # extract features and add image_id and label
    features = prepare_features(pre_img, red_mask, yellow_mask, blue_mask)
    processed_features.append(features)
        
    return processed_features


# Display result in a GUI window
def show_prediction_window(image, prediction, probability):
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
    prediction_text = "✓ Dit is een Mondriaan!" #if prediction[0] == 1 else "✗ Dit is geen Mondriaan"
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
    
    # Start the GUI
    root.mainloop()

    return


def show_input_selection_window():
    """
    Toont een GUI window om de input methode te kiezen: camera of bestand.
    Returns: tuple (use_camera: bool, image_path: str or None)
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