import numpy as np
import matplotlib.pyplot as plt

# Define Gaussian membership function
def gaussian(x, sigma, c):
    return np.exp(-((x - c)**2) / (2 * sigma**2))

# Universe
x = np.linspace(0, 10, 500)

# Parameters
sigma = 1.5
c = 5

# Membership values
y = gaussian(x, sigma, c)

# Plot
plt.figure()
plt.plot(x, y)
plt.title("Gaussian Membership Function")
plt.xlabel("x")
plt.ylabel("Membership Value")
plt.grid()
plt.show()