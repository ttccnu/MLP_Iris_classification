from neurons import Neuron


class Layers:
    def __init__(self, num_neurons, bias):
        self.bias = bias
        self.neurons = []
        for _ in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def forward_propagation(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

