# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Loading datasets
spam = pd.read_csv('/Users/popovgleb/PycharmProjects/polytexTask/1/data/spam.csv')


# Defining function to plot accuracy score depending on training and test samples
def plot_accuracy(data, title):
    # Splitting data into train and test sets
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Defining arrays to store accuracy scores
    train_accuracy = []
    test_accuracy = []

    # Iterating over different sizes of training samples
    for i in range(1, len(X_train) + 1):
        # Fitting Naive Bayes model on the training set
        clf = GaussianNB()
        clf.fit(X_train[:i], y_train[:i])

        # Calculating accuracy scores on training and test sets
        train_accuracy.append(clf.score(X_train[:i], y_train[:i]))
        test_accuracy.append(clf.score(X_test, y_test))

    # Plotting accuracy scores
    plt.plot(range(1, len(X_train) + 1), train_accuracy, label='Train')
    plt.plot(range(1, len(X_train) + 1), test_accuracy, label='Test')
    plt.title(title)
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.show()


# Plotting accuracy score depending on training and test samples for spam dataset
plot_accuracy(spam, 'Spam Dataset')
