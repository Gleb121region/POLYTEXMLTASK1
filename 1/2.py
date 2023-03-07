import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB


class BayesClassifier:
    def __init__(self):
        self.class_1_prior = None
        self.class_minus_1_prior = None
        self.class_1_params = None
        self.class_minus_1_params = None

    def fit(self, X, y):
        self.class_1_prior = np.sum(y == 1) / len(y)
        self.class_minus_1_prior = np.sum(y == -1) / len(y)
        self.class_1_params = [(np.mean(X[y == 1][:, i]), np.std(X[y == 1][:, i])) for i in range(X.shape[1])]
        self.class_minus_1_params = [(np.mean(X[y == -1][:, i]), np.std(X[y == -1][:, i])) for i in range(X.shape[1])]

    def predict(self, X):
        class_1_probabilities = []
        class_minus_1_probabilities = []

        for i in range(X.shape[0]):
            class_1_prob = self.class_1_prior
            class_minus_1_prob = self.class_minus_1_prior

            for j in range(X.shape[1]):
                class_1_prob *= norm.pdf(X[i][j], *self.class_1_params[j])
                class_minus_1_prob *= norm.pdf(X[i][j], *self.class_minus_1_params[j])

            class_1_probabilities.append(class_1_prob)
            class_minus_1_probabilities.append(class_minus_1_prob)

        return np.array(class_1_probabilities) > np.array(class_minus_1_probabilities)


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

classifier = GaussianNB()
classifier.fit(np.column_stack([X1, X2]), y)

y_pred = classifier.predict(np.column_stack([X1, X2]))

print('Accuracy:', accuracy_score(y, y_pred))
ConfusionMatrixDisplay.from_predictions(y, y_pred)
plt.show()

fpr, tpr, thresholds = roc_curve(y, y_pred)
precision, recall, thresholds = precision_recall_curve(y, y_pred)

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
