#neuron_input names as ni
ni1 = 1
ni2 = 2
ni3 = 3
#these inputs can come from other neurons(layer) or outside data
inputs = [ni1, ni2, ni3]

#weights are used to adjust the magnitude of the inputs
#by multiplying them by a defined value, each neuron input has it's own weight
#think of each weight as knob that adjusts the input value 
#by multiplying the input value by what ever the weight(knob) is set to
ni1_weight = .2
ni2_weight = .8
ni3_weight = -.5
weights = [ni1_weight, ni2_weight, ni3_weight]

#the bias can be thought of as the offset, to fine tune the output
#it adds/subtracts to the total of the inputs after each input has been weighted
bias = 2

#to get the output of a neuron we first multiply each input by it's weight adjuster
ni1_weighted = ni1_weight * ni1
ni2_weighted = ni2_weight * ni2
ni3_weighted = ni3_weight * ni3
#then we add those weight adjust values together
weighted_inputs = ni1_weighted + ni2_weighted + ni3_weighted
#finally we add the bias to fine tune the final output value
output = weighted_inputs + bias
print(output)