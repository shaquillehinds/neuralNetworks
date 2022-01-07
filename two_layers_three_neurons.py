import numpy as np
input = [1,2,3,2.5]

weights = [
  [.2, .8, -.5, 1.0],
  [.5, -.91, .26, -.3],
  [-.26, -.27, .17, .87]
]
biases = [2, 3, .5]

output_data = np.dot(weights, input) + biases

print(output_data) # [4.8   1.71  2.385]

#so we know that neuron is basically a set of weights plus bias that runs an algo on input data
#we also know that a layer is a collection of neurons that run together on the same input data
#so to create another layer we would have to create another collection of neurons that receives input data from another source
#note that what makes a layer different from a previous layer, is the source of the input data
#so the output_data from the previous layer will now be the new layers input data
input2 = output_data

#remember to define our neurons by each one their own weight set and bias
#IMPORTANT - our input data has been reduced from 4 to 3 so we have to make sure our weights match
#each weight set should have the same amout of weights as elements in the input data
weights2 = [
  [2, -5, .2],
  [-32, 8, .32],
  [21, .54, -.15],
]
biases = [-5, .75, 4]

#now we just do the same equation as the previous layer to the new input data using our new set of neurons
output_data2 = np.dot(weights2, input2) + biases
print(output_data2) #[  -3.473   -138.4068   105.36565]
