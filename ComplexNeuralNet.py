from Perceptron import Perceptron
import numpy as np


class ComplexNeuralNet:

    def __init__(self):
        self.perceptrons = []
        self.output = []
        self.trained = []

    def add_perceptron(self, p: Perceptron):
        self.perceptrons.append(p)

    def train(self, inputs, labels):
        index = 0
        for inp in inputs:
            self.trained = []
            for p in self.perceptrons:
                if int(p.name) < 3:
                    #print(inp, labels[index][self.perceptrons.index(p)])
                    p.train(inp, labels[index][self.perceptrons.index(p)])
                else:
                    if int(p.name) < 7:
                        self.trained.append(p.train(inp, labels[index][self.perceptrons.index(p)]))
                    else:
                        p.train(np.array(self.trained), labels[index][self.perceptrons.index(p)])
        index += 1

    def predict(self, inputs):
        out = []
        predicted = []
        for p in self.perceptrons:
            if int(p.name) < 3:
                out.append(p.predict(inputs))
            else:
                if int(p.name) < 7:
                    predicted.append(p.predict(inputs))
                else:
                    out.append(p.predict(predicted))
        return out


