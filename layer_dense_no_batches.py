import numpy as np

# first let's create a random seed of 0
np.random.seed(0)

# inital input data is given the variable X
X = [1, 2, 3, 4, 5]

# here we will define our layer class


class Batchless_Layer_Dense:
    # our constructor is where will define our layer
    # a layer definition is simply how many neurons it has(weight sets and biases)
    # and also how many inputs can each neuron receive
    # for the sake of consistency, we will take the number of inputs each neuron will have as the first arg
    # and the number of neurons we will like to have as the second arg
    def __init__(self, n_inputs, n_neurons):
        # because we aren't using batches with this layer
        # our weight shape will be (weight sets, inputs)
        # given us a shape where each row will be a weight set
        # the randn function will return to use a matrix with the specified shape
        # the values in the matrix will ofcourse be random
        self.weights = np.random.rand(n_neurons, n_inputs)
        # now let's create the biases for our neurons
        # we will initialize all of the biases to zero
        # since we only need 1 bias per neuron and we are creating 1 layer, our shape will be a 1D array/vector
        # beware that the zeros method takes an actual shape as it's only argument and not two arguments like randn
        self.biases = np.zeros((1, n_neurons))
    # now let's create our forward pass method, this method is simply taking an input Array and passing it through our layer

    def forward(self, input):
        # we've done this before many times, we're going to find the dot product of each neuron and it's respective bias
        # then finally return an array of the neuron's output to use as the layer's output
        # because our weight sets are a 2D array [[],[]], the dot method will return another 2D array
        # if we were dealing with batches that would be fine, but we only want a 1D array since our Layer_Dense
        # was built specifically for 1D arrays
        self.output = (np.dot(self.weights, input) + self.biases)[0]


# Here we are just creating a layer with our consctructor
# X is the amount of inputs to the layer, in this case count would 5
# and we are saying we would like this layer to have 4 neurons
layer1 = Batchless_Layer_Dense(5, 4)
# now we are going to create a second layer and since we want it to receive input data from the first layser
# we will set the number of inputs to 4, since the first layer has 4 neurons so the layer output will be 4
# I decided to make this layer have 3 neurons this time
layer2 = Batchless_Layer_Dense(4, 3)

# now let's pass our initial data through the first layer
layer1.forward(X)
# let's see what comes out of this layer
print(layer1.output)

# now let's pass the output from layer1 to layer2
layer2.forward(layer1.output)
# let's see what comes out of layer2
print(layer2.output)
