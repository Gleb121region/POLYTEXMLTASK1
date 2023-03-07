import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


def get_data(file_name):
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


X, y = get_data('data/glass.csv')

# Разбиваем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем модель классификатора с различными параметрами
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)

# Обучаем модель
knn.fit(X_train, y_train)

# Делаем предсказания
y_pred = knn.predict(X_test)

# Оцениваем точность, полноту, F-меру и другие метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print('Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}\nF1: {:.3f}'.format(accuracy, precision, recall, f1))

# Создаем список с различными значениями количества ближайших соседей
neighbors = list(range(1, 50))

# Создаем словарь с различными метриками расстояния
metrics = {'euclidean': 'euclidean',
           'manhattan': 'manhattan',
           'chebyshev': 'chebyshev',
           'minkowski': 'minkowski'}

# Создаем пустой словарь для хранения значений ошибки
cv_scores = {}

# Для каждой метрики расстояния
for m in metrics:
    # Создаем пустой список для хранения значений ошибки
    cv_scores[m] = []
    # Для каждого значения k
    for k in neighbors:
        # создаем модель KNN с текущим значением k и метрикой m
        knn = KNeighborsClassifier(n_neighbors=k, metric=metrics[m])
        # проходим кросс-валидацию
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        # добавляем значение ошибки в словарь
        cv_scores[m].append(scores.mean())

    # Создание график
    plt.plot(neighbors, cv_scores[m], label=m)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Определите, к какому типу стекла относится экземпляр с характеристиками:
# RI =1.516 Na =11.7 Mg =1.01 Al =1.19 Si =72.59 K=0.43 Ca =11.44 Ba =0.02 Fe =0.1
glass = pd.DataFrame(
    {'RI': [1.516], 'Na': [11.7], 'Mg': [1.01], 'Al': [1.19], 'Si': [72.59],
     'K': [0.43], 'Ca': [11.44], 'Ba': [0.02], 'Fe': [0.1]})

knn = KNeighborsClassifier(n_neighbors=100, metric='euclidean', weights='distance')
knn.fit(X, y)

print(knn.predict(glass))
