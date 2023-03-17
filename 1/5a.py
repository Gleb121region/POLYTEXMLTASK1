import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = pd.read_csv('data/glass.csv',
                   names=['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'],
                   skiprows=1,
                   header=None)
data = data.drop(['Id'], axis=1)
X = data.drop('Type', axis=1)
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

cv_scores = {}
legends = ('entropy', 'gini', 'log_loss')
depth_range = range(1, 20)
for legend in legends:
    cv_scores[legend] = []
    for i in depth_range:
        tree = DecisionTreeClassifier(criterion=legend, max_depth=i, random_state=1)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores[legend].append(accuracy)
        print(f'legend:{legend}')
        print(f'i:{i}')
        print(f'Accuracy: {accuracy:.3f}')
        if legend == 'entropy':
            export_graphviz(tree,
                            out_file='trees/Entropy_tree' + str(i) + '.dot',
                            feature_names=X.columns,
                            class_names=LabelEncoder().fit_transform(y).astype(str),
                            filled=True
                            )
        elif legend == 'gini':
            export_graphviz(tree,
                            out_file='trees/Gini_tree' + str(i) + '.dot',
                            feature_names=X.columns,
                            class_names=LabelEncoder().fit_transform(y).astype(str),
                            filled=True
                            )
        elif legend == 'log_loss':
            export_graphviz(tree,
                            out_file='trees/Log_loss_tree' + str(i) + '.dot',
                            feature_names=X.columns,
                            class_names=LabelEncoder().fit_transform(y).astype(str),
                            filled=True
                            )
    plt.plot(depth_range, cv_scores[legend], label=legend)

plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
