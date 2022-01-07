import numpy as np

# so what makes a batch layer different from a regular layer
# is how many input streams the neurons are processing at given point in time
# here we defined 2 batches of input streams to be processed 'concurrently'
batch_input = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0]
]

# the weights however don't change but each input batch will be processed by each neuron
# and what are neurons? a collection of weights and a bias
####### 4 neurons #######
weights = [
    [.2, .8, -.5, 1.0],
    [.5, -.91, .26, -.3],
    [-.26, -.27, .17, .87],
    [3, 7, 51, 6]
]
biases = [2, 3, .5, .8]
####### 4 neurons #######
# so that means each input batch will be processed by each weight set + respective bias (neuron)
batch_output = np.dot(batch_input, np.array(weights).T) + biases
print(batch_output)
# [[  4.8     1.71    2.385 185.8  ]
# [  8.9    -1.41    0.2     2.8  ]]
# remember each neuron/set of weights + bias will always output 1 value
# so we have a layer of 4 neurons that will return 4 values
# but because we are running a two batches i.e the same layer concurrently
# we have total batch out put of 8 values
# the layer output is still 4 because we have 4 neurons
# but the total batch output is 8 values

# now we have a new batch of data that we can use for the second layer
batch_input2 = batch_output

# let's create our weight sets and biases that define our second layer
weights2 = [
    [6, 3, 6, 7],
    [-.6, -3, .43, .7],
    [.43, .75, -10, .6],
    [2, 1, 5, 4]
]
biases2 = [7, 5, 9, -6]

# because our input is a batch of inputs we will process on this layer twice
batch_output2 = np.dot(batch_input2, np.array(weights2).T) + biases2
print(batch_output2)
# [[1355.84     128.07555   99.9765   760.435  ]
#  [  76.97       5.936     11.4495    22.59   ]]
