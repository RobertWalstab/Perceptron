from Perceptron import Perceptron
import numpy as np

class NeuralNet:

    def __init__(self):
        self.perceptrons = []

    def add_perceptron(self, p: Perceptron):
        self.perceptrons.append(p)

    def train(self, inputs, labels):
        for p in self.perceptrons:
            p.train(inputs, np.array(labels[self.perceptrons.index(p)]))

    def predict(self, inputs):
        out = []
        for p in self.perceptrons:
            out.append(p.predict(inputs))
        return out
