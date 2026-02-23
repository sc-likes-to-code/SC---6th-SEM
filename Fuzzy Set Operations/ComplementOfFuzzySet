import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 500)

def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a)/(b - a), (c - x)/(c - b)))

A = triangular(x, 2, 4, 6)

# Complement
complement = 1 - A

plt.figure()

plt.plot(x, A, '--', label='Set A')
plt.plot(x, complement, 'green', linewidth=3, label='Complement of A')

plt.fill_between(x, complement, color='green', alpha=0.2)

plt.title("Complement of Set A")
plt.xlabel("x")
plt.ylabel("Membership Value")
plt.legend()
plt.grid()

plt.show()
