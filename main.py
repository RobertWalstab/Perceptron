import numpy as np
from Perceptron import Perceptron
from NeuralNet import NeuralNet

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([2, 2]))
training_inputs.append(np.array([3, 3]))
training_inputs.append(np.array([4, 4]))

training_inputs.append(np.array([-1, -1]))
training_inputs.append(np.array([-2, -2]))
training_inputs.append(np.array([-3, -3]))
training_inputs.append(np.array([-4, -4]))

training_inputs.append(np.array([1, -1]))
training_inputs.append(np.array([2, -2]))
training_inputs.append(np.array([3, -3]))
training_inputs.append(np.array([4, -4]))

training_inputs.append(np.array([-1, 1]))
training_inputs.append(np.array([-2, 2]))
training_inputs.append(np.array([-3, 3]))
training_inputs.append(np.array([-4, 4]))

labels = [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1],
          [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0],
          [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0],
          [0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1]]

perceptron1 = Perceptron(2)
perceptron2 = Perceptron(2)
perceptron3 = Perceptron(2)
perceptron4 = Perceptron(2)
perceptron5 = Perceptron(2)
perceptron6 = Perceptron(2)

nn = NeuralNet()

nn.add_perceptron(perceptron1)
nn.add_perceptron(perceptron2)
nn.add_perceptron(perceptron3)
nn.add_perceptron(perceptron4)
nn.add_perceptron(perceptron5)
nn.add_perceptron(perceptron6)

nn.train(training_inputs, labels)

per = Perceptron(2)

training = []
training.append(np.array([0, 1]))
training.append(np.array([3, 2]))
training.append(np.array([19, 45]))
training.append(np.array([2, 3]))
training.append(np.array([1, 1.1]))

label = [1, 0, 1, 1, 1]

#per.train(training, label)

inputs = np.array([1, 1])
print(nn.predict(inputs))
#=> 1

inputs = np.array([-1, 1])
print(nn.predict(inputs))
#=> 0

