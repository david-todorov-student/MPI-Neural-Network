# MPI-Neural-Network
Parallelizing a Neural Network using MPI: Can we increase speed, while keeping the same accuracy

sequential.py is the NN algorithm running on one CPU core. In it, several NN configurations are tried, to see which one is most optimal in terms of training time and accuracy.
parallel.py takes the most optimal NN from sequential.py and runs in in multiple cores. In order to run parallel.py, one needs to navigate to its directory, and type in the following command:

mpiexec -n {number_of_processors} python parallel.py

e.g.
mpiexec -n 2 python parallel.py

Before running either of these files, one needs to run convert.py in order to get the MNIST dataset in csv format.

The paper submited at the MIPRO conference is in the file:
Parallelization_of_a_neural_network_algorithm_for_use_in_handwriting_recognition.pdf
