import pandas as pd
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

data = pd.read_csv('data/cars.csv')
print(data.head())
# построение регрессионной модели
model = smf.ols('dist ~ speed', data=data).fit()
# оценка длины тормозного пути при скорости 40 миль в час
pred = model.predict(pd.DataFrame({'speed': [40]}))
print("При скорости 40 миль в час, длина тормозного пути составляет: ", pred[0], " футов")

# построение графика регрессионной модели
plt.scatter(data['speed'], data['dist'])
plt.plot(data['speed'], model.predict(data['speed']), color='red')
plt.xlabel('Скорость (миль/час)')
plt.ylabel('Длина тормозного пути (футы)')
plt.title('Зависимость длины тормозного пути от скорости')
plt.show()
