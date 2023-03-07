import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

spam = pd.read_csv('/Users/popovgleb/PycharmProjects/polytexTask/1/data/spam.csv')

def plot_accuracy(data, title):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_accuracy = []
    test_accuracy = []

    for i in range(1, len(X_train) + 1):
        model = GaussianNB()
        model.fit(X_train[:i], y_train[:i])

        train_accuracy.append(model.score(X_train[:i], y_train[:i]))
        test_accuracy.append(model.score(X_test, y_test))

    plt.plot(range(1, len(X_train) + 1), train_accuracy, label='Train')
    plt.plot(range(1, len(X_train) + 1), test_accuracy, label='Test')
    plt.title(title)
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.show()

plot_accuracy(spam, 'Spam Dataset')
