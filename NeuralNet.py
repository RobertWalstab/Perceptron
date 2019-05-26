from Perceptron import Perceptron
import numpy as np


class NeuralNet:

    def __init__(self):
        self.perceptrons = []

    def add_perceptron(self, p: Perceptron):
        self.perceptrons.append(p)

    def train(self, inputs, labels):
        lbl = []
        for p in self.perceptrons:
            for l in labels:
                lbl.append(l[self.perceptrons.index(p)])
            p.train(inputs, np.array(lbl))
            lbl = []

    def predict(self, inputs):
        out = []
        for p in self.perceptrons:
            out.append(p.predict(inputs))
        return out
