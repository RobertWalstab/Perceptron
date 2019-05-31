import numpy as np
from Perceptron import Perceptron
from NeuralNet import NeuralNet
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

training_inputs = []
labels = []
out_x = []
out_y = []
in_x_f = []
in_y_f = []
in_x = []
in_y = []
errorrate = []
trainingsize = 10


def awgn(inp):
    noise = True
    noise_x = 0
    noise_y = 0

    if noise:
        noise_x = (random.random() - 0.48)*.8
        noise_y = (random.random() - 0.48)*.8

    inp[0] += noise_x
    inp[1] += noise_y
    inp.append(abs(inp[0]))
    inp.append(abs(inp[1]))
    return inp


def predict(net, inp, r_w):
    global in_x, in_y, in_x_f, in_y_f, out_x, out_y
    result = net.predict(inp)
    print("Result", result)
    result_coord = convert_to_coordinate(result)
    out_x.append(result_coord[0])
    out_y.append(result_coord[1])
    if result == r_w:
        in_x.append(inp[0])
        in_y.append(inp[1])
    else:
        in_x_f.append(inp[0])
        in_y_f.append(inp[1])
    return result


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

    coord.append(signx * (1 + 2 * inp[2]))
    coord.append(signy * (1 + 2 * inp[3]))
    # coord.append(signx*(1.5+2*inp[2]+signx2*(-0.5+inp[4])))
    # coord.append(signy*(1.5+2*inp[3]+signy2*(-0.5+inp[5])))
    return coord


def make_binary(x, y):
    if x >= 0:
        x1 = 0
    else:
        x1 = 1
    if y >= 0:
        y1 = 0
    else:
        y1 = 1
    x = list(bin(abs(x)))
    y = list(bin(abs(y)))
    x[1] = 0
    y[1] = 0
    return [float(x1), float(x[-2]), float(x[-1]), float(y1), float(y[-2]), float(y[-1])]


def plot(per, i1, i2):
    x = np.linspace(-4, 4, 100)
    b = per.weights[0] + 0.00000000000000000000000000000001
    w1 = per.weights[i1] + 0.00000000000000000000000000000001
    w2 = per.weights[i2] + 0.000000000000000000000000000001
    lbl = 'Perceptron '+per.name
    plt.plot(x, (-(b / w2) / (b / w1)) * x + (-b / w2), label=lbl)
    return


def make_data():
    global training_inputs
    global labels
    x = random.randint(-2, 2)
    y = random.randint(-2, 2)
    ax = abs(x)
    ay = abs(y)
    if x >= 0:
        l1 = 1
    else:
        l1 = 0

    if y >= 1.5:
        l2 = 1
    else:
        l2 = 0

    if ax >= 1.5:
        l3 = 1
    else:
        l3 = 0

    if ay >= 1.5:
        l4 = 1
    else:
        l4 = 0

    #print([l1, l2, l3, l4], x, y)
    labels.append([l1, l2, l3, l4])
    training_inputs.append(np.array(make_binary(x, y)))
    return


correct = True
for index in range(5):
    perceptron1 = Perceptron(6, "1")
    perceptron2 = Perceptron(6, "2")
    perceptron3 = Perceptron(6, "3")
    perceptron4 = Perceptron(6, "4")

    nn = NeuralNet()

    nn.add_perceptron(perceptron1)
    nn.add_perceptron(perceptron2)
    nn.add_perceptron(perceptron3)
    nn.add_perceptron(perceptron4)
    for i in range(trainingsize**index):
        make_data()

    nn.train(training_inputs, labels)

    correct = True

    t = np.linspace(0, trainingsize**index, trainingsize**index)
    plt.plot(in_x, in_y, 'g.', label='Correct Predictions')
    plt.plot(out_x, out_y, 'b+', label='Predictions')
    plt.plot(in_x_f, in_y_f, 'r.', label='Incorrect Predictions')

    plt.title("Results with "+str(10**index)+' training-datasets')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid()
    plt.savefig("ressources/Results with "+str(10**index)+' training-datasets.png')
    plt.show()
    plot(perceptron1, 1, 2)
    plot(perceptron2, 1, 2)

    #plt.ylim(-4, 4)
    #plt.legend()
    #plt.show()

    plot(perceptron3, 3, 4)
    plot(perceptron4, 3, 4)

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.legend()
    plt.title('Linear Separation with '+str(10**index)+' training-datasets')
    plt.savefig('ressources/Linear Separation with '+str(10**index)+' training-datasets.png')
    plt.show()
    training_inputs = []
    labels = []
    out_x = []
    out_y = []
    in_x_f = []
    in_y_f = []
    in_x = []
    in_y = []
    print(nn.predict(make_binary(-2, -2)))
t1 = np.linspace(0, 6, 6)

#plt.plot(t1, errorrate)
plt.title('Errorrate')
plt.savefig('ressources/Errorrate.png')
plt.show()
if correct:
    print("No Error Occurred")
else:
    print(len(in_x_f), "Errors")

