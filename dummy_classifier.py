#Experiment: The accuracy trap



import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

X,y = make_classification(
    n_samples=10000,
    n_features=20,
    weights=[0.99, 0.01],
    random_state=42
)

print("Class distribution")
print(f"class 0: {np.sum(y == 0)} samples")
print(f"class 1: {np.sum(y == 1)} samples")

dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_clf.fit(X,y)

y_pred_dummy = dummy_clf.predict(X)

accuracy_dummy = accuracy_score(y, y_pred_dummy)
f1_dummy = f1_score(y, y_pred_dummy)

print(f"Dummy Classifier Accuracy: {accuracy_dummy:.2%}")
print(f"Dummy Classifier F1_score: {f1_dummy:.2f}")
cm_dummy = confusion_matrix(y, y_pred_dummy)
print("Confusion Matrix (Dummyclassifier):")
print(cm_dummy)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X, y)

y_pred_lr = lr.predict(X)
accuracy_lr = accuracy_score(y, y_pred_lr)
f1_lr = f1_score(y, y_pred_lr)

print(f"Logistic Regression Accuracy: {accuracy_lr:.2%}")
print(f"Logistic Regression F1 score: {f1_lr:.2%}")

cm_lr = confusion_matrix(y, y_pred_lr)
print('confusion matrix (LogisticRegression):')
print(cm_lr)
