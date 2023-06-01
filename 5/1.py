import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

# Загрузка данных
data = pd.read_csv('data/reglab1.txt', delimiter='\t')

print('X/Y')
# Регрессия
slope, intercept, r_value, p_value, std_err = linregress(data['x'], data['y'])
print('slope:', slope)
print('intercept:', intercept)
print('r_value:', r_value)
print('p_value:', p_value)
print('std_err:', std_err)

# График
plt.scatter(data['x'], data['y'])
plt.plot(data['x'], intercept + slope * data['x'], 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print('\n')

print('X/Z')
# Регрессия
slope, intercept, r_value, p_value, std_err = linregress(data['x'], data['z'])
print('slope:', slope)
print('intercept:', intercept)
print('r_value:', r_value)
print('p_value:', p_value)
print('std_err:', std_err)

# График
plt.scatter(data['x'], data['z'])
plt.plot(data['x'], intercept + slope * data['x'], 'r')
plt.xlabel('x')
plt.ylabel('z')
plt.show()
print('\n')

print('Y/Z')
# Регрессия
slope, intercept, r_value, p_value, std_err = linregress(data['y'], data['z'])
print('slope:', slope)
print('intercept:', intercept)
print('r_value:', r_value)
print('p_value:', p_value)
print('std_err:', std_err)

# График
plt.scatter(data['y'], data['z'])
plt.plot(data['y'], intercept + slope * data['y'], 'r')
plt.xlabel('y')
plt.ylabel('x')
plt.show()
print('\n')

# Регрессия
X = data[['x', 'y']]
y = data['z']
reg = LinearRegression().fit(X, y)

print('Coefficient of determination:', reg.score(X, y))
print('intercept:', reg.intercept_)
print('slope:', reg.coef_)

# График
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['x'], data['y'], data['z'], c='black', marker='o')
x_surf = np.arange(0, 1, 0.01)
y_surf = np.arange(0, 1, 0.01)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_surf = reg.intercept_ + reg.coef_[0] * x_surf + reg.coef_[1] * y_surf
ax.plot_surface(x_surf, y_surf, z_surf, cmap='RdYlBu')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
