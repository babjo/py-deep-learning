from MatrixMN import MatrixMN
import random
import math


class NeuralNetwork(object):

    def __init__(self, _num_input, _num_output, _num_hidden_layers):

        # The number of activation values of each layer. This includes bias.
        self.num_layer_acts_ = [None] * (_num_hidden_layers + 2)

        # layer 0 is input layer, +1 is for bias
        self.num_layer_acts_[0] = _num_input + 1

        # default value
        for l in range(1, _num_hidden_layers + 1):
            self.num_layer_acts_[l] = _num_input + 1

        # last layer is output layer. Add +1 for bias as well in case this NN is combined with others.
        self.num_layer_acts_[_num_hidden_layers + 1] = _num_output + 1

        # -1 is for bias
        self.num_input_ = self.num_layer_acts_[0] - 1

        # -1 is for bias
        self.num_output_ = self.num_layer_acts_[_num_hidden_layers + 1] - 1

        # hidden layers + 1 input layer + 1 output layer
        self.num_all_layers_ = _num_hidden_layers + 2

        # constant bias
        self.bias_ = 1

        # learning rate
        self.alpha_ = 0.15

        # layer_neuron_act_[0] = input layer, layer_neuron_act_[num_all_layers_-1] = output_layer, layer_neuron_act_[ix_layer][ix_neuron] = activation value
        # initialize all layers with bias
        self.layer_neuron_act_ = [[0]*(self.num_layer_acts_[i]-1) + [self.bias_] for i in range(self.num_all_layers_)]

        # gradient values for back propagation
        # initialize to store gradient of layers
        self.layer_neuron_grad_ = [[0]*self.num_layer_acts_[i] for i in range(self.num_all_layers_)]

        # weights_[0] is between layer 0 and layer 1.
        # Note -1. Weight matrices are between layers.
        self.weights_ = [MatrixMN() for unused in range(self.num_all_layers_-1)]
        for l in range(len(self.weights_)):

            # row x column = (dimension of next layer -1 for bias) x (dimension of prev layer - this includes bias)
            # -1 is for bias.y = W[x b] ^ T. Don't subtract 1 if you want [y b]^T = W [x b]^T.
            self.weights_[l].initialize(len(self.layer_neuron_act_[l+1])-1, len(self.layer_neuron_act_[l]))

            # random initialization
            for ix in range(self.weights_[l].num_rows_ * self.weights_[l].num_cols_):
                self.weights_[l].values_[ix] = random.random()

        # TODO: Temporary array to store weight matrices from previous step for momentum term.

    def getSigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def getSigmoidGradFromY(self, y):  # not from x. y = getSigmoid(x).
        return (1.0 - y) * y

    def getRELU(self, x):
        return max([0.0, x])

    def getRELUGradFromY(self, x):  # RELU Grad from X == RELU Grad from Y
        if x > 0.0:
            return 1.0
        else:
            return 0.0

    def getLRELU(self, x):
        if x > 0.0:
            return x
        else:
            return 0.01 * x

    def getLRELUGradFromY(self, x):  # RELU Grad from X == RELU Grad from Y
        if x > 0.0:
            return 1.0
        else:
            return 0.01

    def applySigmoidToVector(self, vector):
        # don't apply activation function to bias
        for d in range(len(vector)-1):
            vector[d] = self.getSigmoid(vector[d])

    def applyRELUToVector(self, vector):
        # don't apply activation function to bias
        for d in range(len(vector)-1):
            vector[d] = self.getRELU(vector[d])

    def applyLRELUToVector(self, vector):
        # don't apply activation function to bias
        for d in range(len(vector)-1):
            vector[d] = self.getLRELU(vector[d])

    def propForward(self):
        for l in range(len(self.weights_)):
            # The last component of layer_neuron_act_[l + 1], bias, shouldn't be updated.
            self.weights_[l].multiply(self.layer_neuron_act_[l], self.layer_neuron_act_[l+1])
            # activate
            self.applyRELUToVector(self.layer_neuron_act_[l+1])

    # backward propagation
    def propBackward(self, target):

        # calculate gradients of output layer
        l = len(self.layer_neuron_grad_)-1

        # skip last component (bias)
        for d in range(len(self.layer_neuron_grad_[l])-1):
            output_value = self.layer_neuron_act_[l][d]
            self.layer_neuron_grad_[l][d] = (target[d] - output_value) * self.getRELUGradFromY(output_value)

        # calculate gradients of hidden layers
        for l in reversed(range(len(self.weights_))):
            self.weights_[l].multiplyTransposed(self.layer_neuron_grad_[l+1], self.layer_neuron_grad_[l])

            # skip last component (bias)
            for d in range(len(self.layer_neuron_act_[l])-1):
                self.layer_neuron_grad_[l][d] *= self.getRELUGradFromY(self.layer_neuron_act_[l][d])

        # update weights after all gradients are calculated
        for l in reversed(range(len(self.weights_))):

            # correct weight values of matrix from layer l+1 to l
            self.updateWeight(self.weights_[l], self.layer_neuron_grad_[l+1], self.layer_neuron_act_[l])

    def updateWeight(self, weight_matrix, next_layer_grad, prev_layer_act):
        for row in range(weight_matrix.num_rows_):
            for col in range(weight_matrix.num_cols_):
                delta_w = self.alpha_ * next_layer_grad[row] * prev_layer_act[col]
                weight_matrix.values_[weight_matrix.get1DIndex(row, col)] = weight_matrix.getValue(row, col) + delta_w

    def setInputVector(self, input):

        # use num_input_ in case input vector doesn't include bias
        if len(input) < self.num_input_:
            print "Input dimension is wrong"

        for d in range(self.num_input_):
            self.layer_neuron_act_[0][d] = input[d]

    def copyOutputVector(self, copy_bias=False):
        output_layer_act = self.layer_neuron_act_[len(self.layer_neuron_act_) - 1]
        if copy_bias:
            return [output_layer_act[idx] for idx in range(self.num_output_ + 1)]
        else:
            return [output_layer_act[idx] for idx in range(self.num_output_)]
