#now that we understand that a neuron is just a set of weights
#and a bias that uses an algorithm to modify input data 
#we can simplify our code 

#our inputs
inputs = [1,2,3,2.5]

#our neurons data - we have three neurons because we have 
#3 sets of weights and 3 biases
weights = [
  [.2, .8, -.5, 1.0],
  [.5, -.91, .26, -.3],
  [-.26, -.27, .17, .87]
]
biases = [2, 3, .5]

#each neuron has it's own set of weights and biases
#for example neuron1 would be weights[0] and biases[0]
#and it would receive the inputs array
#all the other neurons will also receive the same inputs array
#but use their own set of weights and their own bias

#let's calculate the output for each of these neurons
#first let's match the weights with their respected biases
#and put them in their own array/tuple e.g [weights[0], biases[0]]
weights_and_biases = zip(weights, biases)

#since we are working with multiple neurons, we will have multiple outputs
#let's create an array to hold these layer outputs in an array
layer_output = []

#now that we have them bundled together, we have individual sets of neuron data
#we can loop through each neuron (weights and bias)
#and run calculations on the input data
#luckily for us python can destructure an array or tuple in a for loop
#so we can get the values of our bundled array and assign them to variables
for weight_set, bias in weights_and_biases:
  neuron_output = 0
  #let's bundle the inputs and weight_set so we can 
  #iterate through the values as a pair 
  for input, weight in zip(inputs, weight_set):
    #let's run the neurons algorithm
    neuron_output += input * weight
  #finally add the bias to complete the neurons algo
  neuron_output += bias
  layer_output.append(neuron_output)

#after we finished loop through each neuron (sets of weights and a bias)
#and running each neuron algo on the input data
#we finally have our out
print(layer_output)