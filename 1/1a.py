from decimal import Decimal

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Считывание данных
data = pd.read_csv('/Users/popovgleb/PycharmProjects/polytexTask/1/data/tic_tac_toe.txt', header=None)

# Кодирование строковых значений в числовые значения
encoder = LabelEncoder()
for col in data.columns[:-1]:
    data[col] = encoder.fit_transform(data[col])

# Разделите данные на обучающий и тестовый наборы
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Классификатор GaussianNB и подгоните его под модель
model = GaussianNB()
model.fit(X_train, y_train)

# Рассчитать показатели точности для обучающего и тестового наборов
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

# Создайте список различных размеров тестового набора
test_sizes = [float(Decimal(i) / 100) for i in range(1, 99)]
train_accuracies = []
test_accuracies = []

# Вычислите показатели точности для каждого размера тестового набора
for size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)

    model = GaussianNB()
    model.fit(X_train, y_train)

    train_accuracies.append(model.score(X_train, y_train))
    test_accuracies.append(model.score(X_test, y_test))

# Рисуем результаты на графике
plt.plot(test_sizes, train_accuracies, label='Train Accuracy')
plt.plot(test_sizes, test_accuracies, label='Test Accuracy')
plt.legend()
plt.xlabel('Test Set Size')
plt.ylabel('Accuracy')
plt.title('Effect of Test Set Size on Naive Bayes Classification Accuracy')
plt.show()
