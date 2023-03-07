# Импортируем необходимые библиотеки
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV
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


X_train, y_train = get_data('/Users/popovgleb/PycharmProjects/polytexTask/1/data/svmdata_c.txt')
X_test, y_test = get_data('/Users/popovgleb/PycharmProjects/polytexTask/1/data/svmdata_c_test.txt')

# Построение моделей SVM с различными ядрами
models = {'linear': SVC(kernel='linear'),
          '1': SVC(kernel='poly', degree=1),
          '2': SVC(kernel='poly', degree=2),
          '3': SVC(kernel='poly', degree=3),
          '4': SVC(kernel='poly', degree=4),
          '5': SVC(kernel='poly', degree=5),
          'sigmoid': SVC(kernel='sigmoid'),
          'rbf': SVC(kernel='rbf')}

# Поиск оптимального значения штрафного параметра с помощью GridSearchCV для каждой модели
parameters = {
    'C': [10 ** -3, 10 ** -2, 10 ** -1,
          10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]}


def linear_kernel_model(X, y):
    for name, model in models.items():
        print(name)
        # Поиск оптимального значения штрафного параметра с помощью GridSearchCV
        clf = GridSearchCV(model, parameters, cv=5)
        clf.fit(X_train, y_train)
        # Выбор оптимального значения штрафного параметра
        optimal_C = clf.best_params_['C']
        print("Optimal value of C: {}".format(optimal_C))

        # Создание финальной модели SVM с оптимальным значением штрафного параметра
        if name.isdigit():
            final_model = SVC(kernel='poly', degree=int(name), C=optimal_C)
            final_model.fit(X_train, y_train)
        else:
            final_model = SVC(kernel=name, C=optimal_C)
            final_model.fit(X_train, y_train)

        # Проверка 0-ошибки на тестовой выборке
        test_accuracy = final_model.score(X_test, y_test)
        print("Test accuracy: {}".format(test_accuracy))

        X0, X1 = X['X1'], X['X2']

        disp = DecisionBoundaryDisplay.from_estimator(
            final_model,
            X,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            xlabel='X1',
            ylabel='X2',
        )
        plt.scatter(X0, X1, c=y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
        print('Number of reference vectors:', len(final_model.support_vectors_))


linear_kernel_model(X_train, y_train)
