from MatrixMN import MatrixMN
from NeuralNetwork import NeuralNetwork

x = [0.0, 0.0]
y_target = [0.3]

nn_ = NeuralNetwork(2, 1, 1)
nn_.alpha_ = 0.01

for unused in range(100):
    nn_.setInputVector(x)
    nn_.propForward()

    y_temp = nn_.copyOutputVector()
    print y_temp

    nn_.propBackward(y_target)

'''
a = MatrixMN()
a.initialize(2, 3)
a.values_ = [1, 2, 3, 4, 5, 6]
result = [0, 0, 0]
#a.multiply([1, 2, 3], result)
a.multiplyTransposed([1, 2, 3], result)
a.cout()
print result
'''