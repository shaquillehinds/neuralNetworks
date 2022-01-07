import numpy as np

#in python a group of data in square brackets is called a list
#in numpy, a list would be called an Array 
#in math, a list would be called a Vector
#if we have a list(s) within a list in python, we call it a list of lists
#if we have an Array(s) within an Array in numpy, we call it a nthD Array
#n being the number total number of nested arrays plus the top parent array
#if we have a vector(s) within a vector in math, we call it a matrix
#A shape is basically a description of the number of elements
#within a list, array, vector, list of lists, nthD Arrays and matrices

#e.g the shape of [1,2,3,4] would be (4). 4 being the number of elements 
#the shape of [[1], [2], [3], [4]] would be (4, 1)
#4 being the number of elements in the top level array and 1 being the number of elements
#one nested level below the parent array. This is a 2D Array
#the shapeof [[[1, 2], [3, 4]]] would be (1, 2, 2). The 1 is the number of elements in the top parent
#the second 2 is the number of elements in the child of the parent
#the third 2 is the number of elements in the granchild of the parent
#this is a 3D Array 

inputs = [1,2,3,2.5]
weights = [
  [.2, .8, -.5, 1.0],
  [.5, -.91, .26, -.3],
  [-.26, -.27, .17, .87]
]
biases = [2, 3, .5]

#using numpy's dot method we can get the dot product of our input and weights
#the dot method will multiply the corresponding index/nth element of each arrays 
#and then finally add those multiplied values together and return us the sum
#e.g
print(np.dot(weights[0], inputs)) #output 2.8

#the dot method can also get the dot product of a matrix
#however it has to be passed in as the first argument
#the matrix will then have each of it's vectors/arrays multiplied by the vector passed as the second argument
#each of the matrix's vectors will then return the sum of the multiplied values
#and the final return from the dot method will be a new vector/array
#e.g
print(np.dot(weights, inputs))

#in python we can add lists of numbers together so that the corresponding index of one list
#will be added together with the corresponding index of another list
#so we will add our weighted values to their corresponding bias
layer_output = np.dot(weights, inputs) + biases
print(layer_output) 