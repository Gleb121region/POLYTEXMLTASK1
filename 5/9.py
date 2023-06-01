import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree

data = pd.read_csv('data/nsw74psid1.csv')

X = data.drop(['re78'], axis=1)
y = data['re78']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

svm = SVR(kernel='linear')
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print('R2 linear regression:', r2_score(y_test, y_pred_lr))
print('R2 SVM regression:', r2_score(y_test, y_pred_svm))
print('R2 decision tree regression:', r2_score(y_test, y_pred_tree))

plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True)
plt.show()
