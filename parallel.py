import copy

import pandas as pd
import numpy as np
import time
from sklearn.neural_network import MLPClassifier
from mpi4py import MPI

# Setting up MPI environment
comm = MPI.COMM_WORLD
n_processes = comm.Get_size()
rank = comm.Get_rank()

# Reading data from file
if rank == 0:
    train_data = list(pd.read_csv("mnist_digits_train.csv").values)
    train_data = train_data[0:6000]
    test_data = list(pd.read_csv("mnist_digits_test.csv").values)
    size_per_proc = len(train_data) // n_processes
    train_data = [[list(train_data[i + j]) for i in range(size_per_proc) if i + j < len(train_data)]
                  for j in range(0, len(train_data), size_per_proc)]
else:
    size_per_proc = None
    train_data = None
    test_data = None

# Distributing training data
size_per_proc = int(comm.bcast(size_per_proc, root=0))

if rank == 0:
    print("Broadcasting size_per_proc = {0} finished".format(size_per_proc), flush=True)

start = None
if rank == 0:
    start = time.time()
comm.Barrier()

train_data = comm.scatter(train_data, root=0)

comm.Barrier()
if rank == 0:
    end = time.time()
    print("Distributing training data finished in {0} seconds".format((end - start)), flush=True)

train_data = np.array(train_data)
comm.Barrier()

# Initializing the Neural Network Classifier
clf = MLPClassifier(hidden_layer_sizes=75, activation="relu",
                    max_iter=75, learning_rate_init=0.001, random_state=0)

# Splitting data into features and target
x_train = train_data[0:, 1:]
y_train = train_data[0:, 0]

# Training the Neural Network
start = None
if rank == 0:
    start = time.time()
comm.Barrier()

clf.fit(x_train, y_train)

comm.Barrier()

# Averaging out the weights in the Neural Network
weights = []
if rank != 0:
    # Slave processes send their weight matrices to the master process.
    comm.send(clf.coefs_, dest=0, tag=11)
    # comm.Barrier()
else:
    # the master process gets the weight matrices from slave processes and computes average
    weights.append(clf.coefs_)
    for i in range(1, n_processes):
        weights.append(comm.recv(source=i, tag=11))

    mainWeights = copy.deepcopy(weights[0])
    for m in range(1, len(weights)):  # n_processes
        for i in range(0, len(mainWeights)):  # 2
            for j in range(0, len(mainWeights[i])):  # 784 / 50
                for k in range(0, len(mainWeights[i][j])):  # 50 / 10
                    mainWeights[i][j][k] += weights[m][i][j][k]

    for i in range(0, len(mainWeights)):  # 2
        for j in range(0, len(mainWeights[i])):  # 784 / 50
            for k in range(0, len(mainWeights[i][j])):  # 50 / 10
                mainWeights[i][j][k] /= n_processes

    clf.coefs_ = mainWeights

comm.Barrier()
if rank == 0:
    end = time.time()
    print("Elapsed training time in seconds is ", (end - start))

comm.Barrier()

# Testing the trained Neural Network
if rank == 0:
    test_data = np.array(test_data)

    x_test = test_data[0:, 1:]
    y_test = test_data[0:, 0]

    p = clf.predict(x_test)
    count = 0
    for i in range(len(x_test)):
        if p[i] == y_test[i]:
            count += 1
    print("Accuracy is: ".format(rank), (count / len(x_test)) * 100)
