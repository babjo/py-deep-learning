#-*- coding: utf-8 -*-

'''
feed-forward
...................................................
....................__________.....................
...................|          |....................
. x (input) -----> |  Neuron  | -----> y (output) .
...................|          |....................
.................↗ |__________|....................
................/..................................
.............. b (bias) ...........................
...................................................
'''

class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def feedForword(self, input):

        # output y = f(\sigma)
        # \sigma = w * input x + b
        # for multiple inputs,
        # \sigma = w0 * input x0 + w1 * input x1 + ... + b

        sigma = self.w * input + self.b
        return self.getAct(sigma)

    def getAct(self, x):

        # for linear or identity activation function
        return x

        # for ReLU activation function
        # return max([0.0, x])

neuron = Neuron(2.0, 1.0)
print 'Input 1.0 -> Output {}'.format(neuron.feedForword(1.0))
print 'Input 2.0 -> Output {}'.format(neuron.feedForword(2.0))
print 'Input 3.0 -> Output {}'.format(neuron.feedForword(3.0))