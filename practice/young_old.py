import numpy as np
import matplotlib.pyplot as plt

def bell(x, a, b, c):
    return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

x = np.arange(0, 101, 1)

young = bell(x, 20, 2, 0)
old = bell(x, 30, 3, 100)

more_or_less_young = np.sqrt(young)

not_young = 1 - young
not_old = 1 - old
not_young_and_not_old = np.minimum(not_young, not_old)

too_young = young ** 2
not_too_young = 1 - too_young
young_but_not_too_young = np.minimum(young, not_too_young)

extremely_old = old ** 2

plt.figure(figsize=(10, 6))

plt.plot(x, young, label='Young')
plt.plot(x, old, label='Old')
plt.plot(x, more_or_less_young, label='More or Less Young')
plt.plot(x, not_young_and_not_old, label='Not Young and Not Old')
plt.plot(x, young_but_not_too_young, label='Young but Not Too Young')
plt.plot(x, extremely_old, label='Extremely Old')

plt.xlabel('Age')
plt.ylabel('Membership Value')
plt.title('Fuzzy Membership Function')
plt.legend()
plt.grid(True)

plt.show()