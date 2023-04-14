import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('data/vehicle.csv')
X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
    'Decision Tree depth 1': DecisionTreeClassifier(max_depth=1),
    'Decision Tree depth 2': DecisionTreeClassifier(max_depth=2),
    'Decision Tree depth 3': DecisionTreeClassifier(max_depth=3),
    'Random forest classifier': RandomForestClassifier(n_estimators=10, random_state=42),
    'SVC': SVC(probability=True, random_state=42)
}

n_estimators = np.arange(1, 21, 1)
results = []
for name, classifier in classifiers.items():
    classifier_results = []
    for n in n_estimators:
        model = AdaBoostClassifier(base_estimator=classifier, n_estimators=n)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classifier_results.append(accuracy)
    results.append(classifier_results)

for i, classifier in enumerate(classifiers):
    plt.plot(n_estimators, results[i], label=str(classifier.title()))
plt.legend()
plt.xlabel('Number of Classifiers')
plt.ylabel('Accuracy')
plt.title('Classification quality with different number of classifiers')
plt.show()
