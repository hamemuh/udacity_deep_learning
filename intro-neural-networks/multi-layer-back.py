import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])  # implies 3 input nodes
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])  # implies 2 hidden nodes

weights_hidden_output = np.array([0.1, -0.3])  # implies 1 output node

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = output - target

# TODO: Calculate error term for output layer
# error_term = (y - y_hat) * f'(h) , where f() is activation function
# already calculated f(h) in output, derivative of sigmoid(x) is sigmoid(x)(1 - sigmoid(x))
output_error_term = error * output * (1 - output)

# TODO: Calculate error term for hidden layer
hidden_error_term = weights_hidden_output * output_error_term * hidden_layer_output * (1 - hidden_layer_output)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
# hidden error term is a (2,1) matrix, input values are (3,1).
# Therefore we can change x into a column vector for matrix multiplication
delta_w_i_h = learnrate * hidden_error_term * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
