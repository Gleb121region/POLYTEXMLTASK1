import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Загрузка данных
data = pd.read_csv('data/titanic_train.csv')

# Выделение целевой переменной и признаков
X = data.drop(['PassengerId', 'Survived'], axis=1)
y = data['Survived']

# Кодирование категориальных признаков
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X['Name'] = le.fit_transform(X['Name'])
X['Ticket'] = le.fit_transform(X['Ticket'])
X['Embarked'] = le.fit_transform(X['Embarked'].fillna('NA'))
X['Cabin'] = le.fit_transform(X['Cabin'].fillna('NA'))

# Замена пропущенных значений
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение нескольких базовых моделей
models = [
    ('Tree', DecisionTreeClassifier()),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('SGDClassifier', SGDClassifier(max_iter=10000)),
    ('MultinomialNB', MultinomialNB()),
    ('svm', SVC(probability=True, gamma='scale')),
    ('SVC_linear', SVC(kernel='linear', C=20.0, gamma='auto', class_weight='balanced')),
    ('SVC_poly_1', SVC(kernel='poly', degree=1, C=20.0, gamma='scale', class_weight='balanced')),
    ('SVC_poly_2', SVC(kernel='poly', degree=2, C=20.0, gamma='scale', class_weight='balanced')),
    ('SVC_poly_3', SVC(kernel='poly', degree=3, C=20.0, gamma='scale', class_weight='balanced')),
    ('SVC_poly_4', SVC(kernel='poly', degree=4, C=20.0, gamma='scale', class_weight='balanced')),
    ('SVC_poly_5', SVC(kernel='poly', degree=5, C=20.0, gamma='scale', class_weight='balanced')),
    ('SVC_sigmoid', SVC(kernel='sigmoid', C=20.0, gamma='scale', class_weight='balanced')),
    ('SVC_rbf', SVC(kernel='rbf', C=20.0, gamma='scale', class_weight='balanced'))
]
for name, model in models:
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f'{name}: CV accuracy = {cv_score:.3f}')
    model.fit(X_train, y_train)

# Получение прогнозов на тестовой выборке
test_preds = []
for name, model in models:
    preds = model.predict_proba(X_test)[:, 1]
    test_preds.append(preds)

# Создание матрицы признаков для мета-модели
X_meta = pd.DataFrame({'model_1': test_preds[0], 'model_2': test_preds[1]})

# Обучение мета-модели
meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_meta, y_test)

# Получение прогнозов мета-модели на тестовой выборке
meta_preds = meta_model.predict(X_meta)

# Оценка качества мета-модели
accuracy = accuracy_score(y_test, meta_preds)
print(f'Meta-model accuracy = {accuracy:.3f}')
