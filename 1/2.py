import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def generate_data():
    """
    Генерация случайных данных для классификации
    """
    x1p2 = np.random.normal(20, 3, 20)
    x2p2 = np.random.normal(4, 4, 20)

    x1m1 = np.random.normal(15, 4, 80)
    x2m1 = np.random.normal(21, 4, 80)

    X1 = np.concatenate([x1p2, x1m1])
    X2 = np.concatenate([x2p2, x2m1])
    y = np.concatenate([np.ones(20), -np.ones(80)])

    return X1, X2, y


def plot_data(X1, X2, y):
    """
    Построение графика данных
    """
    plt.scatter(X1[y == 1], X2[y == 1], label='Class 1')
    plt.scatter(X1[y == -1], X2[y == -1], label='Class -1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


def create_dataframe(X1, X2, y):
    """
    Создание DataFrame из данных
    """
    return pd.DataFrame({'x1': X1, 'x2': X2, 'class': y})


def split_data(data, train_size):
    """
    Разделение данных на обучающий и тестовый наборы
    """
    X = data[['x1', 'x2']]
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Обучение модели на обучающем наборе данных
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    """
    Предсказание классов для данных
    """
    return model.predict(X)


def calculate_accuracy(y_true, y_pred):
    """
    Вычисление показателя точности для данных
    """
    return accuracy_score(y_true, y_pred)


def plot_confusion_matrix(y_true, y_pred):
    """
    Построение матрицы ошибок для данных
    """
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    cm_display.plot()
    plt.show()


def plot_roc_curve(y_true, y_pred):
    """
    Построение ROC-кривой для данных
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()


def plot_pr_curve(y_true, y_pred):
    """
    Построение PR-кривой для данных
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.show()


if __name__ == '__main__':
    X1, X2, y = generate_data()

    plot_data(X1, X2, y)

    data = create_dataframe(X1, X2, y)

    X_train, X_test, y_train, y_test = split_data(data, train_size=0.2)

    grid = train_model(X_train, y_train)

    y_pred = predict(grid, np.column_stack([X1, X2]))

    accuracy = calculate_accuracy(y, y_pred)

    plot_confusion_matrix(y, y_pred)

    plot_roc_curve(y, y_pred)

    plot_pr_curve(y, y_pred)
