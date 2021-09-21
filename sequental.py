import pandas as pd
import time
from sklearn.neural_network import MLPClassifier

# Reading data from file
start = time.time()
train_data = pd.read_csv("mnist_digits_train.csv").values
test_data = pd.read_csv("mnist_digits_test.csv").values
end = time.time()
print("Elapsed time in seconds for reading data is ", (end - start))

# Splitting data into training and test set
x_train = train_data[0:6000, 1:]
y_train = train_data[0:6000, 0]

x_test = test_data[0:, 1:]
y_test = test_data[0:, 0]

# Printing table head
print("{:<10} {:<20} {:<10}".format("test", "training_time", "accuracy"))

# Experimenting with different NN hidden layer sizes and numbers of epochs
for hidden_layer_size in [50, 65, 75, 85, 100]:
    for epochs in [50, 65, 75, 85, 100]:
        # Initializing Neural Network Classifier
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_size, activation="relu",
                            learning_rate_init=0.001, max_iter=epochs, random_state=0)

        # Training Neural Network
        start = time.time()
        clf.fit(x_train, y_train)
        end = time.time()
        training_time = (end - start)

        # Testing the accuracy of the Neural Network on the test set
        p = clf.predict(x_test)
        count = 0
        for i in range(len(x_test)):
            if p[i] == y_test[i]:
                count += 1
        accuracy = (count / len(x_test)) * 100

        print("{:<20} {:<10} {:<20} {:<10}".format(hidden_layer_size, epochs, training_time, accuracy))

# Testing most optimal classifier
for test in range(1, 6):
    clf = MLPClassifier(hidden_layer_sizes=75, activation="relu",
                        learning_rate_init=0.001, max_iter=75, random_state=0)

    # Training Neural Network
    start = time.time()
    clf.fit(x_train, y_train)
    end = time.time()
    training_time = (end - start)

    # Testing the accuracy of the Neural Network on the test set
    p = clf.predict(x_test)
    count = 0
    for i in range(len(x_test)):
        if p[i] == y_test[i]:
            count += 1
    accuracy = (count / len(x_test)) * 100

    print("{:<10} {:<20} {:<10}".format(test, training_time, accuracy))
