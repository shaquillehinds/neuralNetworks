import numpy as np

np.random.seed(0)

# in machine learning we define the input data as X
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]


class Layer_Dense:
    # To initialize a layer we basically have to initalialize neurons
    # As we know, neurons are just basically weight sets with a bias
    def __init__(self, n_inputs, n_neurons):
        # our weights are initialized as random values
        # we keep our weights small because weights as we know multiply input values
        # too big of weights can cause exponential outputs of high numerical values
        # the randn method will generate for us random values inside of matrix shape that we give it
        # the shape is the number of neuron inputs by the number of neurons we want to be in the layer
        # for example let's say a neuron takes 4 inputs and they are three of those neurons in the layer
        # the shape would be (4, 3) which would give us 4 columns and 3 rows
        # the reason we give the inputs as the first value is becuase we want the column depth/number of weights in a column
        # to match the number of inputs in a row
        # this way we won't have to transpose when using the np.dot method
        # so to reiterate, we are using the number inputs to set depth of the weight set columns so we don't have to transpose
        # finally we can set the number of weight sets as the second shape value
        self.weights = .1 * np.random.randn(n_inputs, n_neurons)
        # our biases are usually initialized as zeros
        # the zeros method will create an array of zeros for us with any shape we give it
        # since we're dealing with biases of 1 layer, we only need a 1D array of containing a value for each neuron
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# our input size 4 so we set the first value to 4 and we want 5 neurons
layer1 = Layer_Dense(4, 5)
# becuase layer1 has 5 neurons, will need to have an input size of 5 for this layers neurons
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer1.output)
print(layer2.output)
