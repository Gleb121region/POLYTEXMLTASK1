import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = pd.read_csv('/Users/popovgleb/PycharmProjects/polytexTask/1/data/spam7.csv')
X = data.drop('yesno', axis=1)
y = data['yesno']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')

export_graphviz(tree,
                out_file='/Users/popovgleb/PycharmProjects/polytexTask/1/trees/tree5b.dot',
                feature_names=X.columns,
                class_names=LabelEncoder().fit_transform(y).astype(str),
                filled=True
                )
