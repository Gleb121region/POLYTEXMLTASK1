import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('data/glass.csv')
X = data.drop('Type', axis=1)
y = data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# базовые классификаторы
tree = DecisionTreeClassifier()
forest = RandomForestClassifier(n_estimators=10, random_state=1)
knn = KNeighborsClassifier()
svm = SVC(probability=True, random_state=1)

# список базовых классификаторов
classifiers = [tree, forest, knn, svm]

for clf in classifiers:
    train_scores = []
    test_scores = []
    for n_estimators in range(1, 21):
        bagging = BaggingClassifier(base_estimator=clf, n_estimators=n_estimators, random_state=1)
        bagging.fit(X_train, y_train)
        train_scores.append(bagging.score(X_train, y_train))
        test_scores.append(bagging.score(X_test, y_test))
    # построение графика зависимости качества классификации от количества классификаторов
    plt.plot(range(1, 21), train_scores, label='train accuracy')
    plt.plot(range(1, 21), test_scores, label='test accuracy')
    plt.xlabel('number of classifiers')
    plt.ylabel('accuracy')
    plt.title('Bagging with {} as base classifier'.format(type(clf).__name__))
    plt.legend()
    plt.show()
