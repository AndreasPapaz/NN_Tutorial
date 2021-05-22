import numpy as np
np.random.seed(0)

x = [
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)


print(output)


class Layer_Dense:
    output = None
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(len(x[0]), 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(x)
print(layer2.output)

layer2.forward(layer1.output)
print(layer2.output)



# inputs = [
#     [1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]
# ]

# weights = [
#     [0.2, 0.8, -0.5, 1.0],
#     [0.5, -0.91, 0.26, -0.5],
#     [-0.26, -0.27, 0.17, 0.87]
# ]
 
# bias = [2, 3, 0.5]

# weights2 = [
#     [0.1, -0.14, 0.5],
#     [-0.5, 0.12, -0.33],
#     [-0.44, 0.73, 0.13]
# ]
 
# bias2 = [-1, 2, -0.5]

# layer1_output = np.dot(inputs, np.array(weights).T) + bias
# layer2_output = np.dot(layer1_output, np.array(weights2).T) + bias2

# print(layer2_output)



# inputs = [1, 2, 3, 2.5]

# wts = [
#     [0.2, 0.8, -0.5, 1.0],
#     [0.5, -0.91, 0.26, -0.5],
#     [-0.26, -0.27, 0.17, 0.87]
# ]

# bias = [2, 3, 0.5]

# some_val = -0.5
# weight = 0.7
# bias = 0.7

# layer_outputs = []
# for neuron_wts, neuron_bias in zip(wts, bias):
#     neuron_output = 0

#     for n_input, wt, in zip(inputs, neuron_wts):
#         neuron_output += n_input * wt

#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

# print(layer_outputs)