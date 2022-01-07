import numpy as np

inputs = [[1,2,3,2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

weights1 = [0.2, .8, -.5, 1]
weights2 = [.5, -.91, .26, -0.5]
weights3 = [-.26, -.27, .17, .87]

weights = [weights1,weights2, weights3]
biases = [2, 3, .5]

#layer_outputs = [] #output of current layer
#zip combines two lists by pairing the nth element of the first list with the nth element second list
#e.g [[weights[0], biases[0]], [weights[0], biases[0]]] or [[[0.2, .8, -.5, 1], 2], [[.5, -.91, .26, -0.5], 3]]
#the following for in loop will iterate through each element of the zipped tuple
# for neuron_weights, neuron_bias in zip(weights, biases):
#   neuron_output = 0 # Output of given neuron
#   for n_input, weight in zip(inputs, neuron_weights):
#     neuron_output += n_input * weight
#   neuron_output += neuron_bias
#   layer_outputs.append(neuron_output)
  
neuron_output = np.dot(inputs, np.array(weights).T) + biases
print(neuron_output)
# print([2]+[5])
# print(layer_outputs)