from sklearn import base
from sklearn.calibration import CalibratedClassifierCV
from processing_tools import img_import_resize, processing_image
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
img_set, paths = img_import_resize(folder_path_all)

# in processing_image the images are preprocessed, color masks are made and features are extracted
# the features are returned as a list of dictionaries
features_list = processing_image(img_set, paths)

# create a dataframe from the features list
dataset = pd.DataFrame(features_list)

dataset.to_csv("data.csv", index=False)

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


print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))


# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix for Digits Classification')
plt.tight_layout()
plt.show()



# perform cross-validation to evaluate model stability
cv_scores = cross_val_score(clf, X_train, y_train, cv=4, scoring="accuracy")

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

test_score = clf.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.4f}")

