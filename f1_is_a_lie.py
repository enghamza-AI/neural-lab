# Import libraries
from sklearn.datasets import make_classification   
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt


X, y = make_classification(
    n_samples=1000,
    n_features=20,
    weights=[0.9, 0.1],  
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]


precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

min_length = min(len(precision), len(recall), len(thresholds))
precision = precision[:min_length]
recall = recall[:min_length]
thresholds = thresholds[:min_length]


f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
f05_scores = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall + 1e-9)
f2_scores = (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall + 1e-9)


plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1 (β=1)', color='blue', linewidth=2)
plt.plot(thresholds, f05_scores, label='F0.5 (β=0.5)', color='red', linewidth=2)
plt.plot(thresholds, f2_scores, label='F2 (β=2)', color='green', linewidth=2)

plt.xlabel('Threshold', fontsize=12)
plt.ylabel('F-Score', fontsize=12)
plt.title('F1, F0.5, and F2 Across Thresholds', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.show()
