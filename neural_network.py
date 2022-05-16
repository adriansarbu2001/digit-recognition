import math
import random
import matplotlib.pyplot as plt


class MyMLPClassifier:
    def __init__(self, no_iterations=100, hidden_layer_sizes=(5,), learning_rate=.1, batch_size=None, verbose=False):
        self.no_iterations = no_iterations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.layer_sizes = None
        self.neurons = []
        self.w = []
        self.loss = []

    def fit(self, inputs, outputs):
        self.layer_sizes = (len(inputs[0]),) + self.hidden_layer_sizes + (len(set(outputs)),)
        self.neurons = [[0 for i in range(self.layer_sizes[j])] for j in range(len(self.layer_sizes))]

        no_w = 0
        no_w += len(inputs[0]) * self.hidden_layer_sizes[0]
        for i in range(len(self.hidden_layer_sizes) - 1):
            no_w += self.hidden_layer_sizes[i] * self.hidden_layer_sizes[i + 1]
        no_w += self.hidden_layer_sizes[-1] * len(set(outputs))

        self.w = [random.uniform(-1, 1) for _ in range(no_w)]

        iteration = 1
        while iteration <= self.no_iterations:
            CE_sum = 0.0
            delta_w = [0 for _ in range(no_w)]
            for i in range(len(inputs)):
                self._activate_neurons(inputs[i])
                E = [(1 - self.neurons[-1][j]) * ((1 if outputs[i] == j else 0) - self.neurons[-1][j]) for j in range(len(self.neurons[-1]))]
                CE_sum += sum([(1 if outputs[i] == j else 0) * math.log(self.neurons[-1][j]) for j in range(len(self.neurons[-1]))])
                self._back_propagation(E, delta_w)
                if self.batch_size is not None and i % self.batch_size == 0:
                    self._modify_weights(delta_w)
                    delta_w = [0 for _ in range(no_w)]
            self.loss.append(-CE_sum/len(inputs))
            if self.verbose:
                print("Iteration ", iteration, ", loss = ", self.loss[iteration - 1], sep="")
            self._modify_weights(delta_w)
            iteration += 1

    def _activate_neurons(self, example):
        def activation_function1(x):
            try:
                sig = 1 / (1 + math.exp(-x))
            except OverflowError:
                sig = 1 if x > 0 else 0
            return sig

        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i])):
                if i == 0:
                    self.neurons[i][j] = example[j]
                else:
                    self.neurons[i][j] = 0
                    for k in range(len(self.neurons[i - 1])):
                        w_index = self._compute_index_of_weight(i, j, k)
                        self.neurons[i][j] += self.neurons[i - 1][k] * self.w[w_index]
                    self.neurons[i][j] = activation_function1(self.neurons[i][j])

    def _compute_index_of_weight(self, row, x, x_ant):
        return x * len(self.neurons[row - 1]) + x_ant + sum(
            [self.layer_sizes[i] * self.layer_sizes[i + 1] for i in range(row - 1)])

    def _back_propagation(self, E, delta_w):
        for i in range(len(self.neurons) - 1, 0, -1):
            for j in range(len(self.neurons[i])):
                for k in range(len(self.neurons[i - 1])):
                    w_index = self._compute_index_of_weight(i, j, k)
                    delta_w[w_index] += self.learning_rate * E[j] * self.neurons[i - 1][k]
            E_ant = E.copy()
            E = []
            for j in range(len(self.neurons[i - 1])):
                E.append(sum([self.w[self._compute_index_of_weight(i, r, j)] * E_ant[r] for r in range(len(E_ant))]) *
                         self.neurons[i - 1][j] * (1 - self.neurons[i - 1][j]))

    def _modify_weights(self, delta_w):
        for i in range(len(self.w)):
            self.w[i] += delta_w[i]

    def predict(self, inputs):
        outputs = []
        for example in inputs:
            self._activate_neurons(example)
            o = self.neurons[-1].index(max(self.neurons[-1]))
            outputs.append(o)
        return outputs

    def plot_loss(self):
        plt.plot(self.loss)
        plt.title('LOSS')
        plt.xlabel("EPOCH")
        plt.ylabel("LOSS")
        plt.draw()
        plt.show()
