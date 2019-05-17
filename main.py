import numpy as np
from Perceptron import Perceptron
from NeuralNet import NeuralNet
import random
import matplotlib.pyplot as plt

training_inputs = []
out_x_c = []
out_y_c = []
out_x_f = []
out_y_f = []
in_x = []
in_y = []


def awgn(inp):
    noise = False
    noise_x = 0
    noise_y = 0
    
    if noise:
        noise_x = random.random() - 0.48
        noise_y = random.random() - 0.48
    
    inp[0] += noise_x
    inp[1] += noise_y
    inp.append(abs(inp[0]))
    inp.append(abs(inp[1]))
    return inp


def predict(n, inp, r_w):
    global in_x, in_y, out_x_c, out_y_c, out_x_f, out_y_f
    result = n.predict(inp)
    print("Result", result)
    in_x.append(inp[0])
    in_y.append(inp[1])
    result_coord = convert_to_coordinate(result)
    if result == r_w:
        out_x_c.append(result_coord[0])
        out_y_c.append(result_coord[1])
    else:
        out_x_f.append(result_coord[0])
        out_y_f.append(result_coord[1])
    return


def convert_to_coordinate(inp):
    coord = []
    if inp[0] == 0:
        signx = -1
    else:
        signx = 1
    if inp[1] == 0:
        signy = -1
    else:
        signy = 1
    if inp[2] == 0:
        signx2 = -1
    else:
        signx2 = 1
    if inp[3] == 0:
        signy2 = -1
    else:
        signy2 = 1

    coord.append(signx*(1.5+2*inp[2]))
    coord.append(signy*(1.5+2*inp[3]))
    #coord.append(signx*(1.5+2*inp[2]+signx2*(-0.5+inp[4])))
    #coord.append(signy*(1.5+2*inp[3]+signy2*(-0.5+inp[5])))
    return coord


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


labels = [[1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1],
          [1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 1, 1],
          [0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1],
          ]

perceptron1 = Perceptron(4, 1)
perceptron2 = Perceptron(4, 2)
perceptron3 = Perceptron(4, 3)
perceptron4 = Perceptron(4, 4)
perceptron5 = Perceptron(4, 5)
perceptron6 = Perceptron(4, 6)

nn = NeuralNet()

nn.add_perceptron(perceptron1)
nn.add_perceptron(perceptron2)
nn.add_perceptron(perceptron3)
nn.add_perceptron(perceptron4)
nn.add_perceptron(perceptron5)
nn.add_perceptron(perceptron6)

nn.train(training_inputs, labels)

inputs = np.array(awgn([1, 1]))
print(nn.predict(inputs))
#=> 1

inputs = np.array(awgn([-2, 2]))
print(nn.predict(inputs))
#=> 0
x = np.linspace(-4, 4, 100)

#Perceptron 1
b = perceptron1.weights[0]+0.00000000000000000000000000000001
w1 = perceptron1.weights[1]+0.00000000000000000000000000000001
w2 = perceptron1.weights[2]+0.000000000000000000000000000001

# print(b, w1, w2)

plt.plot(x, (-(b/w2) / ( b / w1))*x + (-b / w2), label='Perceptron 1')

#Perceptron 2
b = perceptron2.weights[0]+0.000000000000000000000000000001
w1 = perceptron2.weights[1]+0.000000000000000000000000000001
w2 = perceptron2.weights[2]+0.000000000000000000000000000001

# print(b, w1, w2)

plt.plot(x, (-(b/w2) / (b / w1))*x + (-b / w2), label='Perceptron 2')

plt.ylim(-4, 4)
plt.legend()
plt.show()

#Perceptron 3
b = perceptron3.weights[0]
w1 = perceptron3.weights[3]
w2 = perceptron3.weights[4]

# print(b, w1, w2)

plt.plot(x, (-(b/w2) / (b / w1))*x + (-b / w2), label='Perceptron 3')

#Perceptron 4
b = perceptron4.weights[0]
w1 = perceptron4.weights[3]
w2 = perceptron4.weights[4]

# print(b, w1, w2)

plt.plot(x, (-(b/w2) / (b / w1))*x + (-b / w2), label='Perceptron 4')

#Perceptron 5
b = perceptron5.weights[0]
w1 = perceptron5.weights[1]
w2 = perceptron5.weights[2]

# print(b, w1, w2)

# plt.plot(x, (-(b/w2) / (b / w1))*x + (-b / w2), label='Perceptron 5')

#Perceptron 6
b = perceptron6.weights[0]
w1 = perceptron6.weights[1]
w2 = perceptron6.weights[2]

# print(b, w1, w2)

# plt.plot(x, (-(b/w2) / (b / w1))*x + (-b / w2), label='Perceptron 6')

plt.legend()
plt.show()

predict(nn, awgn([2, 2]), [1, 1, 0, 0, 0, 0])

plt.plot(out_x_c, out_y_c, 'gx', label='Correct Predictions')
plt.plot(out_x_f, out_y_f, 'r+', label='Incorrect Predictions')
plt.plot(in_x, in_y, 'b.', label='Inputs')


plt.title("Results")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid()
plt.show()
