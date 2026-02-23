import numpy as np
import matplotlib.pyplot as plt

# Define trapezoidal membership function
def trapezoidal(x, a, b, c, d):
    return np.maximum(0,
        np.minimum(
            np.minimum((x - a)/(b - a), 1),
            np.minimum((d - x)/(d - c), 1)
        )
    )

# Universe
x = np.linspace(0, 10, 500)

# Parameters
a = 2
b = 4
c = 6
d = 8

# Membership values
y = trapezoidal(x, a, b, c, d)

# Plot
plt.figure()
plt.plot(x, y)
plt.title("Trapezoidal Membership Function")
plt.xlabel("x")
plt.ylabel("Membership Value")
plt.grid()
plt.show()