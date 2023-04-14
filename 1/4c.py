import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
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


def builder_plt(X, y):
    classifiers = (SVC(kernel='linear', gamma='auto', C=0.1),
                   SVC(kernel='rbf', gamma='scale', C=23.3),
                   SVC(kernel='sigmoid', gamma='scale', C=0.1),
                   SVC(kernel='poly', degree=1, gamma='scale', C=0.1),
                   SVC(kernel='poly', degree=2, gamma='scale', C=0.2),
                   SVC(kernel='poly', degree=3, gamma='scale', C=1.3),
                   SVC(kernel='poly', degree=4, gamma='scale', C=44.7),
                   SVC(kernel='poly', degree=5, gamma='scale', C=100))
    titles = ('SVC with linear kernel',
              'SVC with RBF kernel',
              'SVC with Sigmoid kernel',
              'SVC with polynomial (degree=1) kernel',
              'SVC with polynomial (degree=2) kernel',
              'SVC with polynomial (degree=3) kernel',
              'SVC with polynomial (degree=4) kernel',
              'SVC with polynomial (degree=5) kernel')
    for title, model in zip(titles, classifiers):
        model.fit(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        print("Test accuracy: {}".format(test_accuracy))
        X0, X1 = X['X1'], X['X2']
        DecisionBoundaryDisplay.from_estimator(model, X, response_method="predict", cmap=plt.cm.Pastel1,
                                               xlabel='X1', ylabel='X2')
        plt.scatter(X0, X1, c=y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title(title)
        plt.show()
        print('Number of reference vectors:', len(model.support_vectors_))


if __name__ == '__main__':
    X_train, y_train = get_data('/Users/popovgleb/PycharmProjects/polytexTask/1/data/svmdata_c.txt')
    X_test, y_test = get_data('/Users/popovgleb/PycharmProjects/polytexTask/1/data/svmdata_c_test.txt')
    builder_plt(X_train, y_train)
