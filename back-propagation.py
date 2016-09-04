#-*- coding: utf-8 -*-

'''
back-propagation

E = 1/2*(y(target) - y)^2
∂E/∂y = y - y(target)

w(updated) = w - α * (∂E/∂w)
b(updated) = b - α * (∂E/∂b)

α = 0.1 is the learning rate
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
        output = self.getActivation(sigma)

        self.input = input
        self.output = output

        return output

    def getActivation(self, x):
        # for linear or identity activation function
        return x

        # for ReLU activation function
        # return max([0.0, x])

    def getActGrad(self, x):
        # for linear or identity activation function
        return 1.0

        # for ReLU
        # if x > 0.0: return x
        # else: return 0.0

    def propBackword(self, target):

        # alpha
        a = 0.1

        self.w = self.w - a*(self.output - target)*self.getActGrad(self.output)*self.input
        self.b = self.b - a*(self.output - target)*self.getActGrad(self.output)

neuron = Neuron(2.0, 1.0)
for unused in range(1, 100):
    print 'Input 1.0 -> Output {}'.format(neuron.feedForword(1.0))
    neuron.propBackword(4.0)