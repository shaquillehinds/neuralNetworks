#challenge create a layer of 5 neurons

#inputs
inputs =  [1,2,3,2.5]

#neuron 1
n1_weights =   [.2, .8, -.5, 1.0]
n1_bias =  2

#neuron 2
n2_weights = [.5, -.91, .26, -.3]
n2_bias = 3 

#neuron 3
n3_weights = [-.26, -.27, .17, .87]
n3_bias = .5

#neuron 4
n4_weights = [.2, .5, -.26, -.3]
n4_bias = 1

#neuron 5
n5_weights = [.8, -.91, -.27, .5]
n5_bias = 4

#layer
weights = [n1_weights, n2_weights, n3_weights, n4_weights, n5_weights]
biases = [n1_bias, n2_bias, n3_bias, n4_bias, n5_bias]
layer = zip(weights, biases)

layer_output = []

for neuron_weight_set, neuron_bias in layer:
  neuron_output = 0
  for weight, inputt in zip(neuron_weight_set, inputs):
    neuron_output += weight * inputt
  neuron_output += neuron_bias
  layer_output.append(neuron_output)

print(layer_output)