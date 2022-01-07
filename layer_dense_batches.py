import numpy as np

# let's create our input data
X = [
    [1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]
]

# our random seed
np.random.seed(0)

# let's define our layer class


class Layer_Dense:
    # our constructor taking the number of inputs in each input set and the number nuerons we want in our layer
    def __init__(self, n_inputs, n_neurons):
        # let's define our weight sets, this time we want our sets to be the number of columns in the matrix
        # the reason is so we don't have to transpose when we multiply the input (4, 3) by the weight_sets (3, n_neurons)
        # remember, when multiplying matrices, the length of the rows in the first matrix must be same length as the columns in the second matrix
        # so if I have a matrix shapes (4,5), the length of the 4 rows is 5, then the second matrix needs a column length of 5 (5, any_number)
        self.weights = np.random.randn(n_inputs, n_neurons)
        # here set our biases, since this only 1 layer our shape will be a vector and the length will be the number of neurons
        self.biases = np.zeros((1, n_neurons))
    # let's create our forward pass method

    def forward(self, input):
        self.output = np.dot(input, self.weights) + self.biases


layer1 = Layer_Dense(3, 5)
layer2 = Layer_Dense(5, 3)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
