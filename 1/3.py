import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def get_data(file_name):
    """
    Загрузка данных из файла и преобразование строковых значений в числовые
    """
    data = pd.read_csv(file_name,
                       names=['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'],
                       header=None)

    # Кодирование строковых значений в числовые значения
    encoder = LabelEncoder()
    for col in data.columns[:-1]:
        data[col] = encoder.fit_transform(data[col])

    # Удаляем признак Id number
    data = data.drop(['Id'], axis=1)

    # Разделяем данные на признаки и целевую переменную
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Разделение данных на обучающую и тестовую выборки
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_neighbors=5, weights='distance', p=2):
    """
    Обучение модели классификатора с заданными параметрами
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    knn.fit(X_train, y_train)
    return knn


def predict(model, X):
    """
    Предсказание классов для данных
    """
    return model.predict(X)


def evaluate_model(y_true, y_pred):
    """
    Оценка точности, полноты, F-меры и других метрик
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print('Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}\nF1: {:.3f}'.format(accuracy, precision, recall, f1))


def cross_validate(X_train, y_train, neighbors_range=range(1, 50)):
    """
    Кросс-валидация модели с различными значениями количества ближайших соседей и метриками расстояния
    """
    metrics = {'euclidean': 'euclidean',
               'manhattan': 'manhattan',
               'chebyshev': 'chebyshev',
               'minkowski': 'minkowski'}
    cv_scores = {}

    for m in metrics:
        cv_scores[m] = []
        for k in neighbors_range:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metrics[m])
            scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
            cv_scores[m].append(scores.mean())

        plt.plot(neighbors_range, cv_scores[m], label=m)

    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X, y = get_data('data/glass.csv')

    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    knn = train_model(X_train, y_train)

    y_pred = predict(knn, X_test)

    evaluate_model(y_test, y_pred)

    cross_validate(X_train, y_train)

    glass = pd.DataFrame(
        {'RI': [1.516], 'Na': [11.7], 'Mg': [1.01], 'Al': [1.19], 'Si': [72.59],
         'K': [0.43], 'Ca': [11.44], 'Ba': [0.02], 'Fe': [0.1]})

    knn = train_model(X, y)

    print(predict(knn, glass))
