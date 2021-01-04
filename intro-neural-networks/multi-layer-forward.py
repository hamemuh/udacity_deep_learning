import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

# Network size (4x3x2)
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)
print(X)
weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
print(weights_input_to_hidden)
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))
print(weights_hidden_to_output)

# TODO: Make a forward pass through the network

# X size = (1,4), weights_i_h size = (4,3)
# therefore output is (1,3) - i.e. the values of the hidden layer, [h1 h2 h3]
hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

# hidden_layer_out size = (1,3), weights_hidden_to_output = (3,2)
# therefore output is (1,2) - i.e. the values of the output layer, [o1 o2]
output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)