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


# Загрузка данных из файлов
X_train, y_train = get_data('data/svmdata_e.txt')
X_test, y_test = get_data('data/svmdata_e_test.txt')

# Построение моделей SVM с различными ядрами
models = {'linear': SVC(kernel='linear', gamma='auto'),
          '1': SVC(kernel='poly', degree=1, gamma='auto'),
          '2': SVC(kernel='poly', degree=2, gamma='auto'),
          '3': SVC(kernel='poly', degree=3, gamma='auto'),
          '4': SVC(kernel='poly', degree=4, gamma='auto'),
          '5': SVC(kernel='poly', degree=5, gamma='auto'),
          'sigmoid': SVC(kernel='sigmoid', gamma='auto'),
          'rbf': SVC(kernel='rbf', gamma='auto')}

# Поиск оптимального значения штрафного параметра с помощью GridSearchCV для каждой модели
parameters = {
    'gamma': [10 ** -1, 10 ** 0]
}


def linear_kernel_model(X, y):
    for name, model in models.items():
        # Поиск оптимального значения штрафного параметра с помощью GridSearchCV
        clf = GridSearchCV(model, parameters, cv=5)
        clf.fit(X_train, y_train)
        # Выбор оптимального значения штрафного параметра
        optimal_gamma = clf.best_params_['gamma']
        print('Optimal value of gamma: {}'.format(optimal_gamma))

        # Создание финальной модели SVM с оптимальным значением штрафного параметра
        if name.isdigit():
            final_model = SVC(kernel='poly', degree=name, gamma=optimal_gamma)
            final_model.fit(X_train, y_train)
        else:
            final_model = SVC(kernel=name, gamma=optimal_gamma)
            final_model.fit(X_train, y_train)

        # Проверка 0-ошибки на тестовой выборке
        test_accuracy = final_model.score(X_test, y_test)
        print('Test accuracy: {}'.format(test_accuracy))

        X0, X1 = X['X1'], X['X2']

        disp = DecisionBoundaryDisplay.from_estimator(
            final_model, X,
            response_method='predict',
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
