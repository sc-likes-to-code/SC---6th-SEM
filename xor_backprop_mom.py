import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

w1 = np.random.rand(2,2)
w2 = np.random.rand(2,1)

v1 = np.zeros_like(w1)
v2 = np.zeros_like(w2)

lr = 0.5
momentum = 0.9

for epoch in range(10000):

    hidden = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(hidden, w2))

    error = y - output

    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(w2.T) * sigmoid_derivative(hidden)

    v2 = momentum * v2 + lr * hidden.T.dot(d_output)
    v1 = momentum * v1 + lr * X.T.dot(d_hidden)

    w2 += v2
    w1 += v1

print("Final Output:")
print(np.round(output))
