import numpy as np
import matplotlib.pyplot as plt

# Define triangular membership function
def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a)/(b - a), (c - x)/(c - b)))

# Universe of discourse
x = np.linspace(0, 10, 500)

# Parameters
a = 2
b = 5
c = 8

# Membership values
y = triangular(x, a, b, c)

# Plot
plt.figure()
plt.plot(x, y)
plt.title("Triangular Membership Function")
plt.xlabel("x")
plt.ylabel("Membership Value")
plt.grid()
plt.show()