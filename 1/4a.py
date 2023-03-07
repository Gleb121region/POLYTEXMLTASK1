import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def get_data(file_name):
    data = pd.read_csv(file_name,
                       names=['Id', 'X1', 'X2', 'Color'],
                       sep='\t',
                       skiprows=1,
                       header=None)
    encoder = LabelEncoder()
    for col in data.columns[:-1]:
        data[col] = encoder.fit_transform(data[col])

    data = data.drop(['Id'], axis=1)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def linear_kernel_model(X, y):
    model = SVC(kernel='linear')
    # model = LinearSVC(C=C, max_iter=10000),
    # model = SVC(kernel="rbf", gamma=0.7, C=C),
    # model = SVC(kernel="poly", degree=3, gamma="auto", C=C),
    model.fit(X, y)

    title = "LinearSVC (linear kernel)"

    X0, X1 = X['X1'], X['X2']

    DecisionBoundaryDisplay.from_estimator(model, X,
                                           response_method="predict",
                                           cmap=plt.cm.coolwarm, xlabel='X1', ylabel='X2')
    plt.scatter(X0, X1, c=y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of reference vectors:', len(model.support_vectors_))
    return model


X_train, y_train = get_data('data/svmdata_a.txt')
X_test, y_test = get_data('data/svmdata_a_test.txt')

model = linear_kernel_model(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print('Confusion matrix on train sample:')
ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train)
plt.show()
print('Confusion matrix on test sample:')
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
plt.show()
