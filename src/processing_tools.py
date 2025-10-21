from matplotlib import pyplot as plt
from pathlib import Path
import cv2 as cv
import os
import numpy as np
from scipy.stats import entropy
import pandas as pd

def display_image(img, title=None,):
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def resize_image(img, standard_width, standard_height):
    """
    scale image to a standard size

    Parameters:
    - img: input image
    - standard_width: desired width
    - standard_height: desired height

    Returns:
    - Image with exact standard dimensions
    """

    # check orientation and rotate if neccesary

    h, w = img.shape[:2]
    if h > w:  # portrait image
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]

        
    # calculate scale to fit within standard dimensions
    scale = min(standard_width / w, standard_height / h)

    # New dimensions after scaling
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with same aspect ratio
    resized = cv.resize(img, (new_w, new_h))
        
    # make canvas with standard dimensions (black instead of white)
    canvas = np.full((standard_height, standard_width, 3), (0, 0, 0), dtype=np.uint8)

    # Center the image on the canvas
    y_offset = (standard_height - new_h) // 2
    x_offset = (standard_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas



def mask_feature_color(img_rgb, h_range, s_min, v_min, invert=False):
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    masks = []
    for (h1, h2) in h_range:  # lijst van (minH, maxH) tuples
        lower = (h1, s_min, v_min)
        upper = (h2, 255, 255)
        masks.append(cv.inRange(hsv, lower, upper))
    mask = masks[0]
    for m in masks[1:]:
        mask = cv.bitwise_or(mask, m)
    # kleine schoonmaak
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel, iterations=1)
    if invert:
        mask = cv.bitwise_not(mask)
    return mask

def getLargestContour(img_BW):
    contours, _ = cv.findContours(img_BW.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)

    return np.squeeze(contour)

def getContourExtremes(contour):
    left = contour[contour[:, 0].argmin()]
    right = contour[contour[:, 0].argmax()]
    top = contour[contour[:, 1].argmin()]
    bottom = contour[contour[:, 1].argmax()]

    return np.array((left, right, top, bottom))

def order_quad_points(pts):
   # sort the 4 points as top-left, top-right, bottom-right, bottom-left
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def get_quadrilateral_from_contour(contour):
    # get corners from contour
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        return order_quad_points(approx.reshape(-1, 2))
    # fallback: minAreaRect if no quad found
    rect = cv.minAreaRect(contour.reshape(-1,1,2))
    box = cv.boxPoints(rect)
    return order_quad_points(box)


def warp_image(cropped_img):
    
    gray_img = cv.cvtColor(cropped_img, cv.COLOR_RGB2GRAY)
    contour_in_cropped = getLargestContour(gray_img)
    # 1) src points from contour
    src = get_quadrilateral_from_contour(contour_in_cropped)  # TL,TR,BR,BL

    # 2) targetsize of quad
    (tl, tr, br, bl) = src
    w_top  = np.linalg.norm(tr - tl)
    w_bot  = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right= np.linalg.norm(br - tr)
    maxW = int(round(max(w_top, w_bot)))
    maxH = int(round(max(h_left, h_right)))

    # 3) doelpunten en homografie
    dst = np.array([
        [0,     0],
        [maxW-1,0],
        [maxW-1,maxH-1],
        [0,     maxH-1]
    ], dtype=np.float32)

    M = cv.getPerspectiveTransform(src, dst)

    # 4) warping
    warped = cv.warpPerspective(cropped_img, M, (maxW, maxH),
                                flags=cv.INTER_LINEAR,
                                borderMode=cv.BORDER_REPLICATE)
    return warped

def preprocess_image(img_rgb):
    """
    Preprocess a painting image.
    Steps:
    1. Remove background
    2. Extract ROI using largest contour
    Returns:
        cropped_img
    """
    
    # --- Background removal ---
    mask = mask_feature_color(img_rgb, [(67, 95)], 50, 50, True)
    masked_img = cv.bitwise_and(img_rgb, img_rgb, mask=mask)

    # --- Extract ROI via largest contour ---
    gray = cv.cvtColor(masked_img, cv.COLOR_RGB2GRAY)
    contour = getLargestContour(gray)
    left, right, top, bottom = getContourExtremes(contour)
    x_min, x_max = left[0], right[0]
    y_min, y_max = top[1], bottom[1]
    cropped_img = masked_img[y_min:y_max, x_min:x_max]
    
    warped_img = warp_image(cropped_img)
    
    return warped_img

def apply_morph_operations(mask, masked_img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 1. Erode
    eroded = cv.erode(mask, kernel, iterations=1)
    not_eroded = cv.bitwise_not(eroded)
    img_eroded = cv.bitwise_and(masked_img, masked_img, mask=not_eroded)

    # 2. Dilate
    dilated = cv.dilate(mask, kernel, iterations=1)
    not_dilated = cv.bitwise_not(dilated)
    img_dilated = cv.bitwise_and(masked_img, masked_img, mask=not_dilated)

    # 3. Close
    closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    not_closed = cv.bitwise_not(closed)
    img_closed = cv.bitwise_and(masked_img, masked_img, mask=not_closed)

    # 4. Open
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    not_opened = cv.bitwise_not(opened)
    img_opened = cv.bitwise_and(masked_img, masked_img, mask=not_opened)

    # 5. Open → Close
    open_close = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    open_close = cv.morphologyEx(open_close, cv.MORPH_CLOSE, kernel)
    not_oc = cv.bitwise_not(open_close)
    img_oc = cv.bitwise_and(masked_img, masked_img, mask=not_oc)

    return {
        "erode": img_eroded,
        "dilate": img_dilated,
        "close": img_closed,
        "open": img_opened,
        "open_close": img_oc
    }

def color_percentage(mask):
    total_pixels = mask.size
    color_pixels = cv.countNonZero(mask)
    return (color_pixels / total_pixels) * 100

def center_of_mass(mask, min_area=500):
    # min_area: low threshold to consider as valid color mass
    color_pixels = int(cv.countNonZero(mask))
    if color_pixels < min_area:
        return None  # no reliable color mass

    m = cv.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return None

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return (float(cx), float(cy))

def compute_aspect_ratio(image):
    height, width = image.shape[:2]
    if height == 0:
        return np.nan
    return width / height

def calc_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def color_diversity(img: np.ndarray, s_threshold=30, v_threshold=30) -> float:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    mask = cv.bitwise_and(cv.inRange(s, s_threshold, 255),
                          cv.inRange(v, v_threshold, 255))
    hist = cv.calcHist([h], [0], mask, [180], [0, 180]).flatten()
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    return float(entropy(hist + 1e-6))

def unique_hues(img: np.ndarray, threshold=0.01) -> int:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    mask = cv.bitwise_and(cv.inRange(s, 30, 255), cv.inRange(v, 30, 255))
    hist = cv.calcHist([h], [0], mask, [180], [0, 180]).flatten()
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    return int(np.sum(hist > threshold))

def hue_variance(img: np.ndarray) -> float:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    mask = (s > 30) & (v > 30)
    return float(np.var(h[mask]))

def contour_features(mask: np.ndarray):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'area': 0.0, 'perimeter': 0.0}
    largest = max(contours, key=cv.contourArea)
    area = cv.contourArea(largest)
    perimeter = cv.arcLength(largest, True)
    return {'area': area, 'perimeter': perimeter}

def prepare_features(cropped_img, red_mask, yellow_mask, blue_mask):
    h, w = cropped_img.shape[:2]
    image_center = (w/2, h/2)
    image_diag = np.sqrt(w**2 + h**2)

    # Aspect ratio
    aspect_ratio = compute_aspect_ratio(cropped_img)
    
    # Centers of mass
    red_com = center_of_mass(red_mask)
    yellow_com = center_of_mass(yellow_mask)
    blue_com = center_of_mass(blue_mask)

    def norm_dist(p):
        return np.nan if p is None else calc_distance(p, image_center) / image_diag

    # Distances to center (normalized)
    red_dist = norm_dist(red_com)
    yellow_dist = norm_dist(yellow_com)
    blue_dist = norm_dist(blue_com)

    # Color percentages
    red_pct = color_percentage(red_mask)
    yellow_pct = color_percentage(yellow_mask)
    blue_pct = color_percentage(blue_mask)
    color_coverage = red_pct + yellow_pct + blue_pct

    features = {
        "aspect_ratio": aspect_ratio,
        "red_dist": red_dist,
        "yellow_dist": yellow_dist,
        "blue_dist": blue_dist,
        "red_pct": red_pct,
        "yellow_pct": yellow_pct,
        "blue_pct": blue_pct,
        "color_coverage": color_coverage
    }
    return features

# pipeline function to import en resize images
def img_import_resize(folder_path_all, show_progress=True):
    files_selected = []
   
    # Loop over subfolders
    for subfolder in folder_path_all.iterdir():
        if subfolder.is_dir():
            imgs_in_sub = list(subfolder.glob("*.*"))  # all files in subfolder
            files_selected.extend(imgs_in_sub)

    if show_progress:
        print(f"Gevonden {len(files_selected)} bestanden om te verwerken...")

    # read images and resize them directly to reduce memory usage
    img_BGR = []
    total_files = len(files_selected)
    
    for i, f in enumerate(files_selected):
        img = cv.imread(str(f))
        if img is not None:
            resized_img = resize_image(img,1920,1080)
            img_BGR.append(resized_img)
            
        # Show progress
        if show_progress and total_files > 0:
            progress = ((i + 1) / total_files) * 100
            print(f"\rAfbeeldingen laden en verkleinen: {progress:.1f}% ({i + 1}/{total_files})", end="", flush=True)

    if show_progress:
        print()  # New line after progress
        print("BGR naar RGB conversie...")

    # Convert BGR to RGB
    img_rgb = []
    for i, img in enumerate(img_BGR):
        img_rgb.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        
        # Show conversion progress
        if show_progress and len(img_BGR) > 0:
            progress = ((i + 1) / len(img_BGR)) * 100
            print(f"\rConversie BGR->RGB: {progress:.1f}% ({i + 1}/{len(img_BGR)})", end="", flush=True)
    
    if show_progress:
        print()  # New line after progress
        print("Import en resize voltooid!")
        
    return img_rgb, files_selected

# pipeline function to preprocess images and extract features
def processing_image(image_set, paths, show_progress=True):
    processed_features = []
    total_images = len(image_set)
    k = cv.getStructuringElement(cv.MORPH_RECT, (15,15))
    
    if show_progress:
        print(f"Verwerken van {total_images} afbeeldingen voor feature extractie...")
    
    for i, img in enumerate(image_set):
        # image will be preprocessed, roi extracted and background removed
        pre_img = preprocess_image(img)
        
        red_mask = mask_feature_color(pre_img, [(0, 11), (169, 180)], 110, 70)
        yellow_mask = mask_feature_color(pre_img, [(18, 38)], 70, 90)
        blue_mask_temp = mask_feature_color(pre_img, [(105, 130)], 100, 60)
        blue_mask = cv.morphologyEx(blue_mask_temp, cv.MORPH_OPEN, k, iterations=1)
        
        # extract features and add image_id and label
        features = prepare_features(pre_img, red_mask, yellow_mask, blue_mask)
        features['image_id'] = paths[i].name
        features['label'] = paths[i].parent.name
        processed_features.append(features)
        
        # Show progress
        if show_progress and total_images > 0:
            progress = ((i + 1) / total_images) * 100
            print(f"\rFeature extractie: {progress:.1f}% ({i + 1}/{total_images})", end="", flush=True)
    
    if show_progress:
        print()  # New line after progress
        print("Feature extractie voltooid!")
        
    return processed_features

def scatter_features_raw(df, x_feat, y_feat):
    plt.figure(figsize=(8,6))

    for label in df["label"].unique():
        mask = df["label"] == label
        df_label = df.loc[mask, [x_feat, y_feat]]
        
        plt.scatter(
            df_label[x_feat],
            df_label[y_feat],
            alpha=0.6,
            label=label
        )

    plt.xlabel(x_feat)
    plt.ylabel(y_feat)
    plt.title(f"Scatterplot: {x_feat} vs {y_feat} (individual points)")
    plt.legend()
    plt.show()
