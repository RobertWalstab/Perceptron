import numpy as np
from Perceptron import Perceptron
from NeuralNet import NeuralNet

training_inputs = []

training_inputs = []
training_inputs.append(np.array([1, 1, 1, 1]))
training_inputs.append(np.array([2, 2, 2, 2]))
training_inputs.append(np.array([3, 3, 3, 3]))
training_inputs.append(np.array([4, 4, 4, 4]))

training_inputs.append(np.array([-1, -1, 1, 1]))
training_inputs.append(np.array([-2, -2, 2, 2]))
training_inputs.append(np.array([-3, -3, 3, 3]))
training_inputs.append(np.array([-4, -4, 4, 4]))

training_inputs.append(np.array([1, -1, 1, 1]))
training_inputs.append(np.array([2, -2, 2, 2]))
training_inputs.append(np.array([3, -3, 3, 3]))
training_inputs.append(np.array([4, -4, 4, 4]))

training_inputs.append(np.array([-1, 1, 1, 1]))
training_inputs.append(np.array([-2, 2, 2, 2]))
training_inputs.append(np.array([-3, 3, 3, 3]))
training_inputs.append(np.array([-4, 4, 4, 4]))

training_inputs.append(np.array([1, 1, 1, 1]))
training_inputs.append(np.array([2, 2, 2, 2]))
training_inputs.append(np.array([3, 3, 3, 3]))
training_inputs.append(np.array([4, 4, 4, 4]))

training_inputs.append(np.array([-1, -1, 1, 1]))
training_inputs.append(np.array([-2, -2, 2, 2]))
training_inputs.append(np.array([-3, -3, 3, 3]))
training_inputs.append(np.array([-4, -4, 4, 4]))

training_inputs.append(np.array([1, -1, 1, 1]))
training_inputs.append(np.array([2, -2, 2, 2]))
training_inputs.append(np.array([3, -3, 3, 3]))
training_inputs.append(np.array([4, -4, 4, 4]))

training_inputs.append(np.array([-1, 1, 1, 1]))
training_inputs.append(np.array([-2, 2, 2, 2]))
training_inputs.append(np.array([-3, 3, 3, 3]))
training_inputs.append(np.array([-4, 4, 4, 4]))

training_inputs.append(np.array([1, 2, 1, 2]))
training_inputs.append(np.array([2, 1, 2, 1]))
training_inputs.append(np.array([-1, -2, 2, 1]))
training_inputs.append(np.array([-2, -1, 2, 1]))

training_inputs.append(np.array([3, 2, 3, 2]))
training_inputs.append(np.array([4, 2, 4, 2]))
training_inputs.append(np.array([3, 1, 3, 1]))
training_inputs.append(np.array([4, 1, 4, 1]))

training_inputs.append(np.array([-3, -2, 3, 2]))
training_inputs.append(np.array([-4, -2, 4, 2]))
training_inputs.append(np.array([-3, -1, 3, 1]))
training_inputs.append(np.array([-4, -1, 4, 1]))


# labels = [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1],
#           [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0],
#           [1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 0],
#           [0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 0, 1]]


labels = [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1],
          [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 1, 1],
          [0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1],
          [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 1, 1],
          [0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1],
          [1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
          [1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0],
          [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]]

perceptron1 = Perceptron(4,1)
perceptron2 = Perceptron(4,2)
perceptron3 = Perceptron(4,3)
perceptron4 = Perceptron(4,4)
perceptron5 = Perceptron(4,5)
perceptron6 = Perceptron(4,6)

per1 = Perceptron(2, 1)
per2 = Perceptron(2, 2)

nn2 = NeuralNet()

nn2.add_perceptron(per1)
nn2.add_perceptron(per2)


nn = NeuralNet()

nn.add_perceptron(perceptron1)
nn.add_perceptron(perceptron2)
nn.add_perceptron(perceptron3)
nn.add_perceptron(perceptron4)
nn.add_perceptron(perceptron5)
nn.add_perceptron(perceptron6)

nn.train(training_inputs, labels)

#per = Perceptron(2)

training = []
training.append(np.array([0, 1]))
training.append(np.array([3, 2]))
training.append(np.array([19, 45]))
training.append(np.array([2, 3]))
training.append(np.array([1, 1.1]))

label = [1, 0, 1, 1, 1]

#per.train(training, label)

inputs = np.array([1, 1, 1, 1])
print(nn.predict(inputs))
#=> 1

inputs = np.array([3, 2, 3, 2])
print(nn.predict(inputs))
#=> 0

