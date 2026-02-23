import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 500)

def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a)/(b - a), (c - x)/(c - b)))

A = triangular(x, 2, 4, 6)
B = triangular(x, 4, 7, 9)

# Intersection (MIN)
intersection = np.minimum(A, B)

plt.figure()

plt.plot(x, A, '--', label='Set A')
plt.plot(x, B, '--', label='Set B')
plt.plot(x, intersection, 'red', linewidth=3, label='Intersection')

plt.fill_between(x, intersection, color='red', alpha=0.3)

plt.title("Intersection of 2 Triangles")
plt.xlabel("x")
plt.ylabel("Membership Value")
plt.legend()
plt.grid()

plt.show()
