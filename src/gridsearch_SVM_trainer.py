from sklearn import base
from sklearn.calibration import CalibratedClassifierCV
from processing_tools import img_import_resize, processing_image
from pathlib import Path
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from settings import folder_path_all
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# takes image from setting.py folder_path_all
# in img_import_resize the images are imported from subfolders, are resized to 1920x1080 and returned in img_set
# paths are the paths of the images, used to extract the label from the parent folder name
#img_set, paths = img_import_resize(folder_path_all)

# in processing_image the images are preprocessed, color masks are made and features are extracted
# the features are returned as a list of dictionaries
#features_list = processing_image(img_set, paths)

# create a dataframe from the features list if needed
# Load dataset: prefer cached CSV, otherwise generate from image folders
data_csv_path = Path("data.csv")
if data_csv_path.exists():
    print(f"Lezen van dataset uit: {data_csv_path}")
    dataset = pd.read_csv(data_csv_path)
else:
    print("data.csv niet gevonden, features genereren uit afbeeldingen (dit kan even duren)...")
    # Import and resize images
    img_set, paths = img_import_resize(folder_path_all)
    # Preprocess images and extract features
    features_list = processing_image(img_set, paths)
    # Build DataFrame and save for caching
    dataset = pd.DataFrame(features_list)
    # Basic validation
    if 'image_id' not in dataset.columns or 'label' not in dataset.columns:
        raise RuntimeError("Gegenereerde dataset mist vereiste kolommen: 'image_id' en/of 'label'.")
    dataset.to_csv(data_csv_path, index=False)
    print(f"Gegenereerde dataset opgeslagen naar {data_csv_path}")


# separate features and labels
X = dataset.drop(['image_id','label'], axis=1)
Y = dataset['label']

# split the dataset in training and test set, 80% training, 20% test
# stratify=Y to maintain the same class distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)


# below is the pipeline description
# first the standardscaler is applied to make sure all features are on the same scale, mean = 0 and variance = 1
# then the SVC is initialized with a RBF kernel, wich makes a 3 dimensional field to separate the classes (works better than linear and polynomial)
# gamma is set to 0.001 to control the influence of a single training example, with a low value will a single example (like outliers) have a far reach and will make the decision boundary smoother
# C is the regularization parameter, set to 100 to try to classify all training examples correctly, but not too high to avoid overfitting
# probability=True to enable probability estimates, to calculate the confince of the predictions
# random_state=42 for reproducibility
clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ('scaler', StandardScaler()), 
    ('svc', SVC(kernel='rbf', gamma=0.001, C=100., probability=True, random_state=42))
])

# hyperparameter grid for GridSearchCV
param_grid = {
    "imputer__strategy": ["mean", "median", "most_frequent"],
    "imputer__add_indicator": [True, False],
    "svc__C": [0.1, 1, 10, 50, 80, 100, 200],
    "svc__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1],
    "svc__kernel": ["rbf", "linear", "poly"]
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=4,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
    refit=False
)

grid_search.fit(X_train, y_train)



print("\n=== GridSearch Resultaten ===")
print("Beste parameters:", grid_search.best_params_)
print(f"Beste CV-score (accuracy): {grid_search.best_score_:.4f}")

