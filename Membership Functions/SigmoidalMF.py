# Program to plot Sigmoidal Membership Function

import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid membership function
def sigmoid(x, a, c):
    return 1 / (1 + np.exp(-a * (x - c)))

# Input range
x = np.linspace(-10, 10, 400)

# Parameters
a = 1      # slope (controls steepness)
c = 0      # center

# Compute membership values
y = sigmoid(x, a, c)

# Plotting
plt.figure()
plt.plot(x, y)
plt.title("Sigmoidal Membership Function")
plt.xlabel("Input (x)")
plt.ylabel("Membership Value μ(x)")
plt.grid(True)
plt.show()