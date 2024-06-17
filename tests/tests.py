import multiprocessing as mp
import numpy as np

# test = np.array([
#     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
# ])

# def process_element(i, j):
#     print(test[i][j])

# pool = mp.Pool()
# for i in range(test.shape[0]):
#     for j in range(test.shape[1]):
#         pool.apply_async(process_element, args=(i, j))

# pool.close()
# pool.join()


def tt(**a):
    print(a)

tt(a=1, b=2)