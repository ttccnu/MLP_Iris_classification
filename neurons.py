import math


class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.total_net_input())
        return self.output

    def total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # sigmoid
    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def pd_input(self, target_output):
        return self.pd_error_wrt_output(target_output) * self.pd_input_wrt_input()

    def error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def pd_error_wrt_output(self, target_output):     # 求偏导 partial derivative
        return -(target_output - self.output)

    def pd_input_wrt_input(self):
        return self.output * (1 - self.output)

    def pd_input_wrt_weight(self, index):
        return self.inputs[index]


