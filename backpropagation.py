# P15/101552/2017
# SAMUEL KIPLAGAT RUTTO

# Back-propagation with one hidden layer
import numpy as np

# Initialise weights, bias and learning rates
weight = np.random.rand(3, 1)  # three vector weight of randomly generated numbers between 0 & 1
weight1 = np.random.rand(3, 1)  # weight at hidden layer
bias1 = 0.35
bias2 = 0.5
learning_rate = 0.5


# activation function: sigmoid function
def activation_function(x):
    return 1/(1+np.exp(-x))


# derivative of logistic function to be used in partial derivative of cost function
def activation_func_deriv(x):
    return activation_function(x) * (1 - activation_function(x))


# Training data sets
x_set = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]])
y_set = np.array([[1, 0, 0, 1, 1]])  # expected output
y_set = y_set.reshape(5, 1)

# learning algorithm
for epochs in range(2000):
    inputs = x_set

    # forward pass
    # input layer to hidden layer
    net_input = np.dot(inputs, weight) + bias1
    a = activation_function(net_input)

    # hidden layer to output layer
    net_output = np.dot(inputs, weight1) + bias2
    b = activation_function(net_output)

    # error_out = ((1 / 2) * (np.power((b - y_set), 2)))
    error_out = b - y_set
    print(error_out.sum())

    # back pass
    # Gradients for output layer weights
    dcost_db = b - y_set
    dao_dnet_output = activation_func_deriv(net_output)
    dzo_dweight1 = a

    dcost_weight1 = np.dot(dzo_dweight1.T, dcost_db * dao_dnet_output)

    # Gradient for hidden layer weights
    dcost_dzo = dcost_db * dao_dnet_output

    dzo_da = weight1
    dcost_dah = np.dot(dcost_dzo, dzo_da.T)
    dah_dnet_input = activation_func_deriv(net_input)
    dzh_dweight = x_set
    dcost_weight = np.dot(dzh_dweight.T, dah_dnet_input * dcost_dah)

    weight -= learning_rate * dcost_weight
    weight1 -= learning_rate * dcost_weight1


# Use a single instance to test which category the input falls under
single_point = np.array([1, 0, 0])
result = activation_function(np.dot(single_point, weight) + bias1) + activation_function(np.dot(single_point + bias2))
print(result)






