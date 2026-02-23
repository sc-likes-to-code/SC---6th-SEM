# Program to plot Generalized Bell Membership Function

import numpy as np
import matplotlib.pyplot as plt

# Define generalized bell membership function
def gbell(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

# Input range
x = np.linspace(-10, 10, 400)

# Parameters
a = 2      # width
b = 2      # slope
c = 0      # center

# Compute membership values
y = gbell(x, a, b, c)

# Plotting
plt.figure()
plt.plot(x, y)
plt.title("Generalized Bell Membership Function")
plt.xlabel("Input (x)")
plt.ylabel("Membership Value μ(x)")
plt.grid(True)
plt.show()