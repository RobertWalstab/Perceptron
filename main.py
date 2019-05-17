import numpy as np
from Perceptron import Perceptron
from NeuralNet import NeuralNet
import random
import matplotlib.pyplot as plt
import copy

training_inputs = []

def awgn(input):
    noise = False
    noise_x = 0
    noise_y = 0
    
    if noise:
        noise_x = random.random() - 0.48
        noise_y = random.random() - 0.48
    
    input[0] += noise_x
    input[1] += noise_y
    input[2] = abs(input[0])
    input[3] = abs(input[1])
    return input


training_inputs = []
training_inputs.append(np.array(awgn([1, 1, 1, 1])))
training_inputs.append(np.array(awgn([2, 2, 2, 2])))
training_inputs.append(np.array(awgn([3, 3, 3, 3])))
training_inputs.append(np.array(awgn([4, 4, 4, 4])))

training_inputs.append(np.array(awgn([-1, -1, 1, 1])))
training_inputs.append(np.array(awgn([-2, -2, 2, 2])))
training_inputs.append(np.array(awgn([-3, -3, 3, 3])))
training_inputs.append(np.array(awgn([-4, -4, 4, 4])))

training_inputs.append(np.array(awgn([1, -1, 1, 1])))
training_inputs.append(np.array(awgn([2, -2, 2, 2])))
training_inputs.append(np.array(awgn([3, -3, 3, 3])))
training_inputs.append(np.array(awgn([4, -4, 4, 4])))

training_inputs.append(np.array(awgn([-1, 1, 1, 1])))
training_inputs.append(np.array(awgn([-2, 2, 2, 2])))
training_inputs.append(np.array(awgn([-3, 3, 3, 3])))
training_inputs.append(np.array(awgn([-4, 4, 4, 4])))

training_inputs.append(np.array(awgn([1, 1, 1, 1])))
training_inputs.append(np.array(awgn([2, 2, 2, 2])))
training_inputs.append(np.array(awgn([3, 3, 3, 3])))
training_inputs.append(np.array(awgn([4, 4, 4, 4])))

training_inputs.append(np.array(awgn([-1, -1, 1, 1])))
training_inputs.append(np.array(awgn([-2, -2, 2, 2])))
training_inputs.append(np.array(awgn([-3, -3, 3, 3])))
training_inputs.append(np.array(awgn([-4, -4, 4, 4])))

training_inputs.append(np.array(awgn([1, -1, 1, 1])))
training_inputs.append(np.array(awgn([2, -2, 2, 2])))
training_inputs.append(np.array(awgn([3, -3, 3, 3])))
training_inputs.append(np.array(awgn([4, -4, 4, 4])))

training_inputs.append(np.array(awgn([-1, 1, 1, 1])))
training_inputs.append(np.array(awgn([-2, 2, 2, 2])))
training_inputs.append(np.array(awgn([-3, 3, 3, 3])))
training_inputs.append(np.array(awgn([-4, 4, 4, 4])))


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
          [0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1]
          ]

perceptron1 = Perceptron(4, 1)
perceptron2 = Perceptron(4, 2)
perceptron3 = Perceptron(4, 3)
perceptron4 = Perceptron(4, 4)
perceptron5 = Perceptron(4, 5)
perceptron6 = Perceptron(4, 6)

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

inputs = np.array(awgn([1, 1, 1, 1]))
print(nn.predict(inputs))
#=> 1

inputs = np.array(awgn([-2, 2, 2, 2]))
print(nn.predict(inputs))
#=> 0
x = np.linspace(-4, 4, 100)

#Perceptron 1
b = copy.deepcopy(perceptron1.weights[0])+0.00000000000000000000000000000001
w1 = copy.deepcopy(perceptron1.weights[1])+0.00000000000000000000000000000001
w2 = copy.deepcopy(perceptron1.weights[2])+0.000000000000000000000000000001

print(b, w1, w2)

plt.plot(x, (-(b/w2) / ( b / w1))*x + (-b / w2), label = 'Perceptron 1')

#Perceptron 2
b = copy.deepcopy(perceptron2.weights[0])+0.000000000000000000000000000001
w1 = copy.deepcopy(perceptron2.weights[1])+0.000000000000000000000000000001
w2 = copy.deepcopy(perceptron2.weights[2])+0.000000000000000000000000000001

print(b, w1, w2)

plt.plot(x, (-(b/w2) / (b / w1))*x + (-b / w2), label = 'Perceptron 2')

#Perceptron 3
b = perceptron3.weights[0]
w1 = perceptron3.weights[1]
w2 = perceptron3.weights[2]

print(b, w1, w2)

plt.plot(x, (-(b/w2) / ( b / w1))*x + (-b / w2), label = 'Perceptron 3')

#Perceptron 4
b = perceptron4.weights[0]
w1 = perceptron4.weights[1]
w2 = perceptron4.weights[2]

print(b, w1, w2)

plt.plot(x, (-(b/w2) / ( b / w1))*x + (-b / w2), label = 'Perceptron 4')

#Perceptron 5
b = perceptron5.weights[0]
w1 = perceptron5.weights[1]
w2 = perceptron5.weights[2]

print(b, w1, w2)

plt.plot(x, (-(b/w2) / ( b / w1))*x + (-b / w2), label = 'Perceptron 5')

#Perceptron 6
b = perceptron6.weights[0]
w1 = perceptron6.weights[1]
w2 = perceptron6.weights[2]

print(b, w1, w2)

plt.plot(x, (-(b/w2) / ( b / w1))*x + (-b / w2), label = 'Perceptron 6')


plt.ylim(-4, 4)
plt.legend()
plt.show()
