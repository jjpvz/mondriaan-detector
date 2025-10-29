from processing_tools import img_import_resize, processing_image
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from settings import folder_path_all
import pandas as pd
import matplotlib.pyplot as plt
import joblib


"""
KNN trainer script
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

This script trains a K-Nearest Neighbors (KNN) model to classify Mondriaan paintings
using extracted features from images. It performs hyperparameter tuning using GridSearchCV,
evaluates the model, and saves the trained model to a file.
"""

"""
To use this script:
make sure the images are in the folder defined in settings.py folder_path_all
the images should be in subfolders named after their labels
The script will train a KNN model and save it as 'mondriaan_knn_model.joblib'
"""

# load dataset
img_set, paths = img_import_resize(folder_path_all)
# extract features
features_list = processing_image(img_set, paths)

# set up dataset for training
dataset = pd.DataFrame(features_list)
X = dataset.drop(['image_id','label'], axis=1)
Y = dataset['label']

# split dataset into training and testing sets 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1, stratify=Y)

# set up KNN with pipeline and hyperparameter grid
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# define hyperparameter grid for GridSearchCV
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

# set up cross-validation scheme
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# perform grid search with cross-validation
search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    refit=True,
    verbose=0
)

# fit the model
search.fit(X_train, y_train)
clf = search.best_estimator_

print(f"Best parameters: {search.best_params_}")
print(f"Best cross-validation score: {search.best_score_:.4f}")

# evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

# cross-validation scores
cv_scores = cross_val_score(clf, X_train, y_train, cv=4, scoring="accuracy")
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

# final test set score
test_score = clf.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.4f}")

# save the trained model
model_path = "mondriaan_knn_model.joblib"
joblib.dump(clf, model_path)
print(f"Model saved as: {model_path}")