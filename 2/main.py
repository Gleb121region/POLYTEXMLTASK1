import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pandas import read_csv
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_model(model, x_train, y_train, epochs):
    model.fit(x_train, y_train, epochs=epochs, verbose=0)
    return model


def create_an_image(file_path):
    np_array = read_csv(file_path).to_numpy()
    plt.figure(figsize=(6, 6))
    legend = ('class -1', 'class 1')
    x1 = np_array[np_array[:, 2] == -1]
    x2 = np_array[np_array[:, 2] == 1]
    plt.scatter(x1[:, 0], x1[:, 1], label='X1')
    plt.scatter(x2[:, 0], x2[:, 1], label='X2')
    plt.title(file_path)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(legend)
    plt.grid(True)
    plt.show()


def load_data(file_path):
    np_array = read_csv(file_path).to_numpy()
    y = np_array[:, -1]
    y[y == -1] = 0
    y[y == 1] = 1
    y = y.astype('int')
    x = np_array[:, :-1]
    x = x.astype('float')
    return train_test_split(x, y, test_size=0.2, random_state=42)


def create_an_image_accuracy_to_epochs(title):
    x_train, x_test, y_train, y_test = load_data(title)
    acc = []
    x = np.arange(1, 100, 1)
    for i in x:
        temp = []
        for _ in range(10):
            tf.keras.backend.clear_session()
            model = tf.keras.models.Sequential([tf.keras.layers.Input(2), tf.keras.layers.Dense(1, activation='relu')])
            model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
            model = train_model(model, x_train, y_train, epochs=i)
            _, test_acc = model.evaluate(x_test, y_test, verbose=0)
            temp.append(test_acc)
        acc.append(np.mean(temp))
    plt.plot(x, acc)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


def task_first(title):
    x_train, x_test, y_train, y_test = load_data(title)
    activations = ('relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'exponential', 'elu', 'selu', 'tanh')
    optimizers = ('RMSprop', 'Adadelta', 'Adam', 'Adagrad', 'Adamax', 'FTRL', 'NAdam', 'SGD')
    act = {}
    for activation in activations:
        opt = {}
        model = tf.keras.models.Sequential([tf.keras.layers.Input(2), tf.keras.layers.Dense(1, activation=activation)])
        for optimizer in optimizers:
            temp = []
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            for _ in range(10):
                tf.keras.backend.clear_session()
                model.fit(x_train, y_train, epochs=100, verbose=0)
                _, test_acc = model.evaluate(x_test, y_test, verbose=0)
                temp.append(test_acc)
            opt[optimizer] = np.mean(temp)
        act[activation] = opt
        plt.title(activation + ' in ' + title)
        plt.ylabel('Accuracy')
        plt.bar(list(opt.keys()), opt.values(), color='b')
        plt.show()
    for a in act:
        for o, acc in act[a].items():
            print(a, ":", o, ":", acc)


def task_second():
    n = 10
    acc_sum = 0
    ls_sum = 0
    for _ in range(n):
        tf.keras.backend.clear_session()
        x_train, x_test, y_train, y_test = load_data('nn_1.csv')
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(2), tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(2, activation='sigmoid')
        ])
        model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=50, verbose=0)
        loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        acc_sum += test_acc
        ls_sum += loss
    print("accuracy:" + str(acc_sum / n))
    print("loss:" + str(ls_sum / n))


def task_third():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    fig, axes = plt.subplots(8, 8, figsize=(6, 6))
    axes = axes.ravel()
    for i in range(64):
        axes[i].imshow(x_train[i], cmap=plt.cm.binary)
        axes[i].axis('off')
        axes[i].set_title(str(y_train[i]))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(16, activation='tanh'),
        tf.keras.layers.Dense(16, activation='relu'), tf.keras.layers.Dense(10, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)

    loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("Test Accuracy:", test_acc)
    print("Loss:", loss)

    prediction_values = (model.predict(x_test) > 0.5).astype("int32")

    fig, axes = plt.subplots(6, 20, figsize=(15, 7))
    axes = axes.ravel()
    for i in range(120):
        axes[i].imshow(x_test[i].reshape((28, 28)), cmap=plt.cm.gray_r)
        axes[i].axis('off')
        axes[i].set_title(str(prediction_values[i]))

    plt.show()


if __name__ == '__main__':
    # create_an_image(r'C:\Users\Gelog\Desktop\POLYTEXMLTASK1\2\nn_0.csv')
    # create_an_image(r'C:\Users\Gelog\Desktop\POLYTEXMLTASK1\2\nn_1.csv')
    create_an_image_accuracy_to_epochs(r'C:\Users\Gelog\Desktop\POLYTEXMLTASK1\2\nn_0.csv')
    # show_accuracy_epochs(r'C:\Users\Gelog\Desktop\POLYTEXMLTASK1\2\nn_1.csv')
    # part_1(r'C:\Users\Gelog\Desktop\POLYTEXMLTASK1\2\nn_0.csv')
    # part_2()
    # print('========================')
    # part_3()
