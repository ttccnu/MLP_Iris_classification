import random
from layers import Layers


class MLP:
    def __init__(self, num_inputs, num_hidden, num_outputs, learn_rate, h_w=None, h_b=None,
                 o_w=None, o_b=None):
        self.num_inputs = num_inputs
        self.hidden_layer = Layers(num_hidden, h_b)
        self.output_layer = Layers(num_outputs, o_b)

        self.init_weights_i2h(h_w)
        self.init_weights_h2o(o_w)
        self.learn_rate = learn_rate

    def init_weights_i2h(self, hidden_layer_weights):
        weight_num = 0
        for k in range(len(self.hidden_layer.neurons)):
            for _ in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[k].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[k].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_h2o(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.forward_propagation(inputs)
        return self.output_layer.forward_propagation(hidden_layer_outputs)

    def train(self, training_inputs, training_outputs):
        self.forward(training_inputs)

        pd_output_neuron = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            pd_output_neuron[o] = self.output_layer.neurons[
                o].pd_input(training_outputs[o])

        pd_hidden_neuron = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_output_neuron[o] * \
                                                    self.output_layer.neurons[o].weights[h]

            pd_hidden_neuron[h] = d_error_wrt_hidden_neuron_output * \
                                                             self.hidden_layer.neurons[
                                                                 h].pd_input_wrt_input()

        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                pd_error_wrt_weight = pd_output_neuron[o] * self.output_layer.neurons[
                    o].pd_input_wrt_weight(w_ho)

                self.output_layer.neurons[o].weights[w_ho] -= self.learn_rate * pd_error_wrt_weight

        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                pd_error_wrt_weight = pd_hidden_neuron[h] * self.hidden_layer.neurons[
                    h].pd_input_wrt_weight(w_ih)
                self.hidden_layer.neurons[h].weights[w_ih] -= self.learn_rate * pd_error_wrt_weight

    def total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].error(training_outputs[o])
        return total_error

