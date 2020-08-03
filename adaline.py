# P15/101552/2017
# SAMUEL KIPLAGAT RUTTO

# A single layer neural network
import numpy as np

# Initialise weights, bias and learning rates
np.random.seed(42)
weight = np.random.rand(3, 1)  # three vector weight of randomly generated numbers between 0 & 1
bias = np.random.rand(1)
learning_rate = 0.5

# activation function
def activation(x):
    return 1 / (1 + np.exp(-x))


# derivative of activation function to be used in the cost function
def active_der(x):
    return activation(x) * (1 - activation(x))


# Training data set
x_set = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]])
y_set = np.array([[1, 0, 0, 1, 1]])
y_set = y_set.reshape(5, 1)

# Learning algorithm
# error_out = 0
for epochs in range(1000):
    # while error_out >= 1:
    inputs = x_set

    # net input, dot product of weight and input plus bias
    in_weights = np.dot(x_set, weight) + bias
    z = activation(in_weights)

    # Least Mean Squared error cost function
    error_out = ((1 / 2) * (np.power((z - y_set), 2)))
    print(error_out.sum())

    dcost_db = error_out

    # Adjusting the weight and bias
    dpred_dz = active_der(z)
    delta = dcost_db * dpred_dz

    if error_out.sum() >= 1:
        inputs = x_set.T
        weight -= learning_rate * np.dot(inputs, delta)
        print(weight)

    else:
        weight = weight
        print(weight)

    for num in delta:
        bias -= learning_rate * num
        print(bias)


# Use a single instance to test which category the input falls under
single_point = np.array([1, 0, 0])
result = activation(np.dot(single_point, weight) + bias)
print(result)
