#challenge create a layer of 4 neurons 
layer_input = [3, -.4, 23, 1]
layer_output = []

weights = [
  [5, 35, -2, 3],
  [2, 1, 6, 9],
  [.3, 8, 32, 95],
  [-23, .53, 45, 7]
]
biases = [63, 39, 98,16]

for weight_set, bias in zip(weights, biases):
  neuron_output = 0
  for input, weight in zip(layer_input, weight_set):
    neuron_output+= input * weight
  neuron_output += bias
  layer_output.append(neuron_output)
  
print(layer_output)