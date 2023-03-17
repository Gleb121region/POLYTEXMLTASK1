import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
    ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier


def get_data(file_name):
    data = pd.read_csv(file_name,
                       sep='\t',
                       names=['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines', 'age',
                              'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                              'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                              'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                              'NumberOfDependents'],
                       skiprows=1,
                       header=None)
    X = data.iloc[:, 1:]
    y = data['SeriousDlqin2yrs']
    return X, y


def print_result(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Вычисляем метрики качества работы модели
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    # Выводим результаты
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1 score:', f1)
    print('roc_auc:', roc_auc)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(str(model))
    plt.show()


# Разделение данных на тренировочную и тестовую выборки
X_train, y_train = get_data('data/bank_scoring_train.csv')
X_test, y_test = get_data('data/bank_scoring_test.csv')

print_result(RandomForestClassifier())
print_result(LogisticRegression())
print_result(DecisionTreeClassifier())
print_result(GradientBoostingClassifier())
