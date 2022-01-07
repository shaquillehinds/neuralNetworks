#so a neuron as we've seen in one_neuron
#is simply an algorithm to process input data with the use of
#multipliers called weights and an fine tuner called biases
#but what if we wanted to calculate the output of in the data we received
#using different weight values and bias?
#Then we just need to add another neuron 
#Which is simply an algorithm which processes input data
#with the use of weights and a bias
#To put it simply, what makes each neuron unique is
#it's bias and it's weights
#so to create a new neuron we simply add another set of weights and a bias
#a collection of neurons that receive the same input data is called a layer

ni1 = 1
ni2 = 2
ni3 = 3
ni4 = 2.5
inputs = [ni1, ni2, ni3, ni4]

#first neuron
ni1_weight = .2
ni2_weight = .8
ni3_weight = -.5
ni4_weight = 1.0
weights = [ni1_weight, ni2_weight, ni3_weight, ni4_weight]
bias = 2
ni1_weighted = ni1_weight * ni1
ni2_weighted = ni2_weight * ni2
ni3_weighted = ni3_weight * ni3
ni4_weighted = ni4_weight * ni4
weighted_inputs = ni1_weighted + ni2_weighted + ni3_weighted + ni4_weighted
neuron1_output = weighted_inputs + bias

print(neuron1_output)

#second neuron - as you can see it has it's weights and biases
ni1_weight2 = .5
ni2_weight2 = -0.91
ni3_weight2 = .26
ni4_weight2 = -.5
weights2 = [ni1_weight2, ni2_weight2, ni3_weight2, ni4_weight2]
bias2 = 3
#notice the input remains the same despite the weight being different
ni1_weighted2 = ni1_weight2 * ni1
ni2_weighted2 = ni2_weight2 * ni2
ni3_weighted2 = ni3_weight2 * ni3
ni4_weighted2 = ni4_weight2 * ni4
weighted_inputs2 = ni1_weighted2 + ni2_weighted2 + ni3_weighted2 + ni4_weighted2
neuron2_output = weighted_inputs2 + bias2

print(neuron2_output)

