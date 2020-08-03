# P15/101552/2017
# SAMUEL KIPLAGAT RUTTO

import numpy as np

# initialise the weights, learning rate and bias
np.random.seed(42)  # Ensures that we get the same random numbers every time
weights = np.random.rand(3, 1)  # Since we have three feature in the input, we have a vector of three weights
bias = np.random.rand(1)
learn_rate = 0.05


# Inputs and expected output
training_set = np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0],
                         [1, 1, 0],
                         [1, 1, 1]])
classes = np.array([[1, 0, 0, 1, 1]])
classes = classes.reshape(5, 1)  # rows by columns


# Sigmoid function is used for the activation function
def activ_func(x):
    return 1/(1+np.exp(-x))


# Derivative of the activation function which is used in finding the derivative of the cost function
def activ_func_deriv(x):
    return activ_func(x) * (1 - activ_func(x))


# Training the neural network
for epoch in range(20000):  # Number of times the algorithm should run in order to minimize the error
    inputs = training_set

    # Feed-forward
    # Dot product of input and weights plus bias
    input_weights = np.dot(training_set, weights) + bias

    # Pass the dot product through the activ_func function
    z = activ_func(input_weights)

    # Back-pass
    # Print error
    error = z - classes
    print(error.sum())

    # Use chain-rule to find derivative of cost function
    dcost_dpred = error
    dpred_dz = activ_func_deriv(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = training_set.T
    weights -= learn_rate * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= learn_rate * num


# Use a single instance to test which category the input falls under
# single_point = np.array([1, 0, 0])
# result = activ_func(np.dot(single_point, weights) + bias)
# print(result)
