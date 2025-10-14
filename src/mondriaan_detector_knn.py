from features import img_import_resize, processing_image
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


# toelichting op keuzes

img_set, paths = img_import_resize(folder_path_all)
features_list = processing_image(img_set, paths)

dataset = pd.DataFrame(features_list)
X = dataset.drop(['image_id','label'], axis=1)
Y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1, stratify=Y)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    refit=True,
    verbose=0
)

search.fit(X_train, y_train)
clf = search.best_estimator_

print(f"Best parameters: {search.best_params_}")
print(f"Best cross-validation score: {search.best_score_:.4f}")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

cv_scores = cross_val_score(clf, X_train, y_train, cv=4, scoring="accuracy")
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

test_score = clf.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.4f}")
