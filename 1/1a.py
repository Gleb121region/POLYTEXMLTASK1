import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


def read_data(file_path):
    """
    Считывание данных из файла
    """
    return pd.read_csv(file_path, header=None)


def encode_data(data):
    """
    Кодирование строковых значений в числовые значения
    """
    encoder = LabelEncoder()
    for col in data.columns[:-1]:
        data[col] = encoder.fit_transform(data[col])
    return data


def split_data(data, test_size):
    """
    Разделение данных на обучающий и тестовый наборы
    """
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Обучение модели на обучающем наборе данных
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def calculate_accuracy(model, X, y):
    """
    Вычисление показателя точности для данных
    """
    return model.score(X, y)


def calculate_accuracies(data, test_sizes):
    """
    Вычисление показателей точности для различных размеров тестового набора
    """
    train_accuracies = []
    test_accuracies = []

    for size in test_sizes:
        X_train, X_test, y_train, y_test = split_data(data, size)

        model = train_model(X_train, y_train)

        train_accuracy = calculate_accuracy(model, X_train, y_train)
        test_accuracy = calculate_accuracy(model, X_test, y_test)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    return train_accuracies, test_accuracies


def plot_results(test_sizes, train_accuracies, test_accuracies):
    """
    Построение графика результатов
    """
    plt.plot(test_sizes, train_accuracies, label='Train Accuracy')
    plt.plot(test_sizes, test_accuracies, label='Test Accuracy')
    plt.legend()
    plt.xlabel('Test Set Size')
    plt.ylabel('Accuracy')
    plt.title('Effect of Test Set Size on Naive Bayes Classification Accuracy')
    plt.show()


if __name__ == '__main__':
    test_sizes = np.arange(.01, 0.99, .01)
    file_path_tic_tac_toe = 'data/tic_tac_toe.txt'
    tic_tac_toe = read_data(file_path_tic_tac_toe)
    tic_tac_toe = encode_data(tic_tac_toe)
    train_accuracies_tic_tac_toe, test_accuracies_tic_tac_toe = calculate_accuracies(tic_tac_toe, test_sizes)
    plot_results(test_sizes, train_accuracies_tic_tac_toe, test_accuracies_tic_tac_toe)
