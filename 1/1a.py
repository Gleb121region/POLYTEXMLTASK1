import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Read in data
data = pd.read_csv('1/data/tic_tac_toe.txt', header=None)

# Encode string values to numerical values
encoder = LabelEncoder()
for col in data.columns[:-1]:
    data[col] = encoder.fit_transform(data[col])

# Split data into training and test sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Naive Bayes Classifier and fit the model
clf = GaussianNB()
clf.fit(X_train, y_train)

# Calculate accuracy scores for training and test sets
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

# Create a list of different test set sizes
test_sizes = [0.2, 0.4, 0.6, 0.8]
train_accuracies = []
test_accuracies = []

# Calculate accuracy scores for each test set size
for size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    train_accuracies.append(clf.score(X_train, y_train))
    test_accuracies.append(clf.score(X_test, y_test))

# Plot the results in a graph 
plt.plot(test_sizes, train_accuracies, label='Train Accuracy')
plt.plot(test_sizes, test_accuracies, label='Test Accuracy')
plt.legend()
plt.xlabel('Test Set Size')
plt.ylabel('Accuracy')
plt.title('Effect of Test Set Size on Naive Bayes Classification Accuracy')
plt.show()
