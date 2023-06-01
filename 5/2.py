from itertools import combinations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/reglab.txt', delimiter='\t', skiprows=1)

# разделяем данные на признаки и целевую переменную
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# определяем максимальное количество признаков
d = X.shape[1]

# задаем начальное значение RSS
min_RSS = float('inf')

# задаем начальное значение подмножества признаков
opt_subset = None

# перебираем все значения k
for k in range(d + 1):
    # генерируем все возможные комбинации признаков длиной k
    for subset in combinations(X.columns, k):
        if len(subset) > 0:
            # обучаем модель с текущим подмножеством признаков
            model = LinearRegression().fit(X[list(subset)], y)
            # вычисляем остаточную сумму квадратов RSS
            RSS = mean_squared_error(y, model.predict(X[list(subset)])) * len(y)
            # если RSS меньше текущего минимального значения
            if RSS < min_RSS:
                # обновляем минимальное значение RSS
                min_RSS = RSS
                # сохраняем текущее подмножество признаков
                opt_subset = subset

# выводим оптимальное подмножество признаков и значение RSS
print('Optimal subset of features:', opt_subset)
print('Minimum RSS:', min_RSS)

# from itertools import combinations
#
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
#
# data = pd.read_csv('data/reglab.txt', delim_whitespace=True)
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values
# n, d = X.shape
#
# best_subset = []
# best_RSS = np.inf
# for k in range(d+1):
#     for subset in combinations(range(d), k):
#         if len(subset) > 0:  # добавляем проверку на пустые подмножества
#             X_subset = X[:, subset]
#             model = LinearRegression().fit(X_subset, y)
#             RSS = mean_squared_error(y, model.predict(X_subset)) * n
#             if RSS < best_RSS:
#                 best_subset = subset
#                 print(RSS)
#                 best_RSS = RSS
# print(f"Best subset: {best_subset}")


