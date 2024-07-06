import numpy as np

def get_quadrants(matrix: np.array):
    """
    Returns the four quadrants of a matrix

    Input:
    matrix: np.array: matrix to be split into quadrants

    Output:
    Tuple[np.array, np.array, np.array, np.array]: tuple containing the four quadrants of the matrix
    """

    n = matrix.shape[0]
    half_n = n // 2
    upper_left = matrix[:half_n, :half_n]
    upper_right = matrix[:half_n, half_n:]
    lower_left = matrix[half_n:, :half_n]
    lower_right = matrix[half_n:, half_n:]

    return upper_left, upper_right, lower_left, lower_right

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
print(a)

upper_left, upper_right, lower_left, lower_right = get_quadrants(a)

print(upper_left)
print(upper_right)
print(lower_left)
print(lower_right)