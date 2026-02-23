import numpy as np
import matplotlib.pyplot as plt

# Universe of discourse
x = np.linspace(0, 10, 500)

# Triangular membership function
def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a)/(b - a), (c - x)/(c - b)))

# Define sets
A = triangular(x, 2, 4, 6)
B = triangular(x, 4, 7, 9)

# Union (MAX)
union = np.maximum(A, B)

plt.figure()

plt.plot(x, A, '--', label='Set A')
plt.plot(x, B, '--', label='Set B')
plt.plot(x, union, 'black', linewidth=3, label='Union')

plt.fill_between(x, union, color='cyan', alpha=0.3)

plt.title("Union of 2 Triangles")
plt.xlabel("x")
plt.ylabel("Membership Value")
plt.legend()
plt.grid()

plt.show()
