import numpy as np
import pandas as pd
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


def linear_kernel_model(X, y):
    model = SVC(kernel='linear')
    # Поиск оптимального значения штрафного параметра с помощью GridSearchCV
    parameters = {'C': np.arange(.01, 600, .01)}
    clf = GridSearchCV(model, parameters, cv=5)
    clf.fit(X, y)
    # Выбор оптимального значения штрафного параметра
    optimal_C = clf.best_params_['C']
    print("Optimal value of C: {}".format(optimal_C))

    final_model = SVC(kernel='linear', C=optimal_C)
    final_model.fit(X_train, y_train)

    test_accuracy = final_model.score(X_test, y_test)
    print("Test accuracy: {}".format(test_accuracy))
    return final_model


if __name__ == '__main__':
    # Загрузка данных из файлов
    X_train, y_train = get_data('data/svmdata_b.txt')
    X_test, y_test = get_data('data/svmdata_b_test.txt')
    # Обучение модели с линейным ядром
    grid = linear_kernel_model(X_train, y_train)
