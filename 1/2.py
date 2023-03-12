import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

x1p2 = np.random.normal(20, 3, 20)
x2p2 = np.random.normal(4, 4, 20)

x1m1 = np.random.normal(15, 4, 80)
x2m1 = np.random.normal(21, 4, 80)

X1 = np.concatenate([x1p2, x1m1])
X2 = np.concatenate([x2p2, x2m1])
y = np.concatenate([np.ones(20), -np.ones(80)])

plt.scatter(x1p2, x2p2, label='Class 1')
plt.scatter(x1m1, x2m1, label='Class -1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

data = pd.DataFrame({'x1': X1, 'x2': X2, 'class': y})

X = data[['x1', 'x2']]
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(np.column_stack([X1, X2]))

print('Accuracy:', accuracy_score(y, y_pred))
ConfusionMatrixDisplay.from_predictions(y, y_pred)
plt.show()

fpr, tpr, _ = roc_curve(y, y_pred)
precision, recall, _ = precision_recall_curve(y, y_pred)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve')
plt.show()
