import  numpy as np

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

training_input = np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
])

training_output = np.array([[0,1,1,0]]).T

np.random.seed(1)

w = np.random.randn(3,1)* 0.01

for iteration in range(20000):
    x = training_input
    z = np.dot(x,w)
    a = sigmoid(z)
    erro = training_output - a
    adjusments = erro * sigmoid_derivative(a)
    w += np.dot(x.T, adjusments)

print a