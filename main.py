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
    x = (random.random()-.5)*8
    y = (random.random()-.5)*8
    ax = abs(x)
    ay = abs(y)
    if x >= 0:
        l1 = 1
    else:
        l1 = 0

    if y >= 0:
        l2 = 1
    else:
        l2 = 0

    if ax >= 2:
        l3 = 1
    else:
        l3 = 0

    if ay >= 2:
        l4 = 1
    else:
        l4 = 0

    if x >= 0:
        xo = int(x) + 1
    else:
        xo = int(x) - 1

    if y >= 0:
        yo = int(y) + 1
    else:
        yo = int(y) - 1

    print(xo, yo, [l1, l2, l3, l4], x, y)
    labels.append([l1, l2, l3, l4])
    training_inputs.append(np.array([x, y, ax, ay]))
    return


correct = True
for index in range(6):
    perceptron1 = Perceptron(4, "1")
    perceptron2 = Perceptron(4, "2")
    perceptron3 = Perceptron(4, "3")
    perceptron4 = Perceptron(4, "4")

    nn = NeuralNet()

    nn.add_perceptron(perceptron1)
    nn.add_perceptron(perceptron2)
    nn.add_perceptron(perceptron3)
    nn.add_perceptron(perceptron4)
    for i in range(trainingsize**index):
        make_data()

    nn.train(training_inputs, labels)

    correct = True

    for t in range(30):
        for i in range(9):
            if i != 0:
                for j in range(9):
                    if j != 0:
                        if i < 0:
                            xtemp = i - 3.5
                        else:
                            xtemp = i - 4.5
                        if j < 0:
                            ytemp = j - 3.5
                        else:
                            ytemp = j - 4.5
                        axtemp = abs(xtemp)
                        aytemp = abs(ytemp)
                        inputs = np.array(awgn([xtemp, ytemp]))
                        if xtemp >= 0:
                            l1 = 1
                        else:
                            l1 = 0

                        if ytemp >= 0:
                            l2 = 1
                        else:
                            l2 = 0

                        if axtemp >= 2:
                            l3 = 1
                        else:
                            l3 = 0

                        if aytemp >= 2:
                            l4 = 1
                        else:
                            l4 = 0

                        if xtemp >= 0:
                            xo = int(xtemp) + 1
                        else:
                            xo = int(xtemp) - 1

                        if ytemp >= 0:
                            yo = int(ytemp)
                        else:
                            yo = int(ytemp)

                        prediction = predict(nn, inputs, [l1, l2, l3, l4])

                        if prediction != [l1, l2, l3, l4]:
                            correct = False

                        print(xtemp, ytemp, prediction, prediction == [l1, l2, l3, l4])
    errorrate.append(len(in_x_f)/(30*64))
    t = np.linspace(0, trainingsize**index, trainingsize**index)
    '''
    for p in nn.perceptrons:
        plt.plot(p.errors, 'b.')
        plt.title('Perceptron ' + p.name)
        plt.show()
    '''
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

    plt.ylim(-4, 4)
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

t1 = np.linspace(0, 6, 6)
plt.plot(t1, errorrate)
plt.title('Errorrate')
plt.savefig('ressources/Errorrate.png')
plt.show()
if correct:
    print("No Error Occurred")
else:
    print(len(in_x_f), "Errors")

