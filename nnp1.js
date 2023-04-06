const inputs = [1, 2, 3, 2.5];

const weights1 = [0.2, 0.8, -0.5, 1];
const weights2 = [0.5, -0.91, 0.26, -0.5];
const weights3 = [-0.26, -0.27, 0.17, 0.87];
const weights = [weights1, weights2, weights3];

const biases = [2, 3, 0.5];

const layerOutput = [];

for (let [neuronWeights, neuronBias] of zip(weights, biases)) {
  let neuronOutput = 0;
  for (let [neuronWeight, neuronInput] of zip(neuronWeights, inputs)) {
    neuronOutput += neuronInput * neuronWeight;
  }
  neuronOutput += neuronBias;
  layerOutput.push(neuronOutput);
}

console.log(layerOutput);
console.log(dot(weights, inputs));

function zip(list1, list2) {
  let newList = [];
  for (let i = 0; i < list1.length; i++) {
    if (list1[i] !== undefined && list2[i] !== undefined) {
      newList.push([list1[i], list2[i]]);
    }
  }
  return newList;
}

function dot(vector1, vector2) {
  let returnVector = [];
  if (vector1[0][0] !== undefined) {
    vector1.forEach((vector) => returnVector.push(dot(vector, vector2)));
  }
  if (returnVector.length === 0) returnVector = vector1;
  else return returnVector;
  if (returnVector.length !== vector2.length) throw new Error("Invalid shape");
  returnVector = returnVector
    .map((val, i) => val * vector2[i])
    .reduce((reduced, curr) => reduced + curr, 0);
  return returnVector.length <= 1 ? returnVector[0] : returnVector;
}
