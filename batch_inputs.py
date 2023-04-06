import numpy as np

inputs = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

weights = [
    [.2, .8, -.5, 1.0],
    [.5, -.91, .26, -.5],
    [-.26, -.27, .17, .87]
]
biases = [2, 3, .5]

# to run our neurons on batch inputs, we must first turn our weights list into a 2D array column
# the numpy method array turns our list of lists into a 2D array first
# w_arr stands for weights array
w_arr = np.array(weights)
# then we use the array method T (transpose) to turn the row array into a column array
w_arr_col = w_arr.T
# finally we have to make sure that the inputs list is the first argument to the dot method
# when multiplying matrices, each row will run a dot method on all columns individually
# creating a new row of dot productions
# until finally returns a new matrix of batch neurons
# remember this is the same a layer but it's
# concurrently processing 3 batches of inputs
layer1_output = np.dot(inputs, w_arr_col) + biases
# print(weights)
# print(np.array(weights).T)
print(layer1_output)

# here is our second layer
weights2 = [
    [.1, -.14, .5],
    [-.5, .12, -.33],
    [-.44, .73, -.13]
]
biases2 = [-1, 2, - .5]

# let's calculate the batch output
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print(layer2_output)
