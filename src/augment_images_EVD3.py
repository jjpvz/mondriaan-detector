import cv2
import numpy as np
import albumentations as A
import os
import random

# Hoofdmap waar al je M1, M2, M3, ... mappen staan
base_dir = r"C:\GIT\mondriaan-detector\alldata"

# Lijst met submappen die je wilt verwerken
folders = ["mondriaan1", "mondriaan2", "mondriaan3", "mondriaan4", "niet_mondriaan"]

# Augmentatie-instellingen
border_color = (0, 0, 0)  # zwart
transform = A.Compose([
    A.Rotate(limit=3, border_mode=cv2.BORDER_CONSTANT, fill_value=border_color, p=0.8),
    A.Affine(shear={'x': (-20, 20)}, fill_value=border_color, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.2),
   # A.HorizontalFlip(p=0.5),
   # A.VerticalFlip(p=0.2),
   # A.RandomBrightnessContrast(p=0.3),
   # A.GaussNoise(var_limit=(3.0, 15.0), p=0.3)  # Subtiele ruis, minder vaak
])

# Verwerk elke map apart
for folder in folders:
    input_dir = os.path.join(base_dir, folder)
    output_dir = os.path.join(input_dir, f"{folder}-1")
    os.makedirs(output_dir, exist_ok=True)

    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    all_images.sort()

    if not all_images:
        print(f"⚠️ Geen afbeeldingen gevonden in {folder}, overslaan...")
        continue

    start_num = 131
    target_num = 230
    counter = start_num
    total_needed = target_num - start_num + 1

    print(f"\n📂 Verwerken van map: {folder} ({len(all_images)} originele foto's)")

    for _ in range(total_needed):
        filename = random.choice(all_images)
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Kan {filename} niet lezen, overslaan...")
            continue

        augmented = transform(image=image)
        augmented_image = augmented["image"]

        output_path = os.path.join(output_dir, f"{folder} ({counter}).jpg")
        cv2.imwrite(output_path, augmented_image)
        counter += 1

    print(f"✅ Klaar met {folder}: nieuwe foto's opgeslagen in {folder}-1\n")

print("🎉 Alle mappen zijn verwerkt!")
