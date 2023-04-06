import numpy as np
# what defines a neuron? it's weights and it's bias
# how many values does a neuron output? 1
# how many inputs can a neuron have? any amount
# how do you create a layer? by adding another neuron (weights and bias) to process the same input
# what are weights? weights are individual numerical values that are mapped to input values. 1 weight per input value
# what is a bias? A bias is an offset value that is added to the dot product of weights and inputs. 1 bias per neuron
# how many weight sets can a neuron have? A neuron can only have 1 weight set
# how many weight sets can a layer have? trick question but the number of weight sets will always be equal to the number of nuerons since 1 neuron has 1 weight set
# what is a dot product? a dot product is the product of 2 Arrays
# how is a dot product calculated? Each member of an array is mulitplied by the corresponding index of another array, then all product values are added together to create a sum of all products
# numpy's dot method, how is a dot product calculated if the first Array has more than 1 dimension?
# --First the nested sequence or sequences(the inner array that holds values) has to be the same 1 dimensional shape as the second  array
# --Secondly all of the nested sequences will return a dot product into new Array
# --Example the dot product of [[3, 5]], [8, 6] will be [54], the dot product of [[3, 5], [1, 2]], [8, 6] will be [54, 20]
# --As you can see if a nDArray is multiplied by a 1D Array, each nested sequence will return a value and the value from the entire method will be an array containing dot products
# --The depth of the array will be 1 nested level shallower than the original nD array
# what is a batch? A batch is a layer that processing mulitple input sets at the same time. The layer remains the same in this case but inputs are now given in sets as a matrix/nDArray
# what's the difference between a batch and a layer? A layer is composed of multiple neurons(sets of weights and biases) while a batch is any layer that processing multiple sets of inputs at any given time
# what is a hidden layer? A hidden layer is any layer that is between the input_neurons and the output neurons

# below create 1 neuron with 3 input values

input = [3, 5, 8]

weights = [.5, .2, .7]
bias = 6

# now get the output of that neuron without the help of the numpy library then print the value

dot_product = 0
for input_value, weight in zip(input, weights):
    dot_product += input_value * weight
dot_product += bias
print(dot_product)

# now do the same thing but with the numpy libraries dot method and then print the value
dot_product = np.dot(input, weights) + bias
print(dot_product)

# are the values the same? yes

# let's take it a step further. Below create a layer of 3 neurons

weights = [
    [.5, .2, .7],
    [.025, .12, .99],
    [-.1, -.03, -.46]
]
biases = [6, 4, 3]

# find the output of all three of the neurons in the layer without using the numpy library and print the result

layer_output = []

for weight_set, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for weight, input_value in zip(weight_set, input):
        neuron_output += weight * input_value
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)

# do the same thing but this using numpy's dot method then print the value
layer_output = np.dot(weights, input) + biases
print(layer_output)

# are the values the same? yes, but I had a clash of variables at first and called the wrong variable at one point

# let's take it another step further. Below create a layer 4 neurons that runs in 3 batches and each neuron takes 5 inputs

inputs = [
    [3, 5, 8, 2, 9],
    [-3, -6, -4, -6, -5],
    [.5, .6, .3, .7, .1]
]

weights = [
    [.5, .2, .7, .8, .1],
    [.025, .12, .99, .25, 33],
    [-.1, -.03, -.46, -.36, -.75],
    [-.6, -.26, -.3, -.7, -.12]
]
biases = [5, 3, 6, 7]

# find the batch output of the layer without using numpy and then print the result

layer_batch = []
for input_set in inputs:
    batch_layer_output = []
    for weight_set, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for weight_value, input_value in zip(weight_set, input_set):
            weighted_input = weight_value * input_value
            neuron_output += weighted_input
        neuron_output += neuron_bias
        batch_layer_output.append(neuron_output)
    layer_batch.append(batch_layer_output)

print(layer_batch)

# now the same thin but using numpy's dot product method

layer_batch = np.dot(inputs, np.array(weights).T) + biases

print(layer_batch)

# are the values the same? yes, but I hate python. I was using the variable and I didn't even know and it took forever to debug


# print(np.dot(
#     [
#       [[3, 5],
#        [8, 6]],

#       ],
#     [1, 2]
# ))
