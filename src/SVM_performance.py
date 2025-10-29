from sklearn import base
from sklearn.calibration import CalibratedClassifierCV
from processing_tools import img_import_resize, processing_image
import cv2 as cv
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from settings import folder_path_all
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from test_tools import plot_learning_curve, save_plot


"""
SVM performance evaluation script
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

This script trains and evaluates an SVM model to classify Mondriaan paintings
using extracted features from images. It evaluates the model's performance,


To use this script:
make sure the dataset is available as data.csv in the same folder as this script.
Run the script, it will load the dataset, train the SVM model, evaluate its performance,
and display the confusion matrix and learning curve.
"""

data_csv_path = Path("data.csv")
if data_csv_path.exists():
    print(f"Lezen van dataset uit: {data_csv_path}")
    dataset = pd.read_csv(data_csv_path)

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
    ("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
    ('scaler', StandardScaler()), 
    ('svc', SVC(kernel='rbf', gamma=1, C=80., probability=True, random_state=42))
])

# train the model
clf.fit(X_train, y_train)

# test the model
y_pred = clf.predict(X_test)

train_score = clf.score(X_train, y_train)
test_score  = clf.score(X_test, y_test)

print(f"Train accuracy: {train_score:.3f}")
print(f"Test accuracy:  {test_score:.3f}")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

# perform cross-validation to evaluate model stability
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plt = plot_learning_curve(clf, X_train, y_train, cv=cv, n_jobs=-1)
#save_plot(plt)
plt.show()

