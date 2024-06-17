import numpy as np

a = [np.array([8, 9, 10, 11]), np.array([12, 13, 14])]
b = [np.array([1, 2, 3, 4]), np.array([5, 6, 7])]

c = np.mean([a, b], axis=0)

print(c)

