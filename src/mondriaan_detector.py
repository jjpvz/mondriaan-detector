from features import img_import_resize, processing_image
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from settings import folder_path_all
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# toelichting op keuzes

# takes image from setting.py folder_path_all
img_set, paths = img_import_resize(folder_path_all)
features_list = processing_image(img_set, paths)

dataset = pd.DataFrame(features_list)
X = dataset.drop(['image_id','label'], axis=1)
Y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(gamma=0.001, C=100., random_state=42))
])

clf.fit(X_train, y_train)


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

model_path = "mondriaan_svm_model.joblib"
joblib.dump(clf, model_path)
print(f"Model opgeslagen als: {model_path}")