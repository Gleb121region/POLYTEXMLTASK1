import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# todo: переделать! поменять значения x1_1 … и т.д

# Generate random data
x1_1 = np.random.normal(10, 4, 50)
x1_2 = np.random.normal(20, 3, 50)
x2_1 = np.random.normal(14, 4, 50)
x2_2 = np.random.normal(18, 4, 50)

# Plot data
plt.scatter(x1_1, x2_1, label='Class -1')
plt.scatter(x1_2, x2_2, marker='x', label='Class 1')
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.legend()
plt.show()

# Create data frame
x1 = np.concatenate((x1_1, x1_2))
x2 = np.concatenate((x2_1, x2_2))
class_label = np.concatenate((np.repeat('-1', 50), np.repeat('1', 50)))
data = pd.DataFrame({'x1': x1, 'x2': x2, 'class': class_label})

# Calculate misclassification rates for different test set sizes
test_sizes = [0.2, 0.4, 0.6, 0.8]
mistake_rates = []
for size in test_sizes:
    # Split data into training and test sets
    X = data[['x1', 'x2']]
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)

    # Create a Naive Bayes Classifier and fit the model
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Calculate misclassification rate on test set
    y_pred = clf.predict(X_test)
    mistake_rate = (y_pred != y_test).sum() / len(y_test)

    mistake_rates.append(mistake_rate)

# Plot the results in a graph
plt.plot(test_sizes, mistake_rates)
plt.xlabel('Test Set Size')
plt.ylabel('Misclassification Rate')
plt.title('Effect of Test Set Size on Naive Bayes Classification Accuracy')
plt.show()
