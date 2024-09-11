from typing import Tuple
import numpy as np

def shuffle_visual_field_dataset(
    x_data: np.array, y_data: np.array
) -> Tuple[np.array, np.array]:
    # shuffles dataset comprised of two arrays

    n = len(y_data)
    unique_indices = np.random.permutation(
        n
    )  # generates random permutation of list in range(0, n)
    x_data = x_data[unique_indices]
    y_data = y_data[unique_indices]

    return x_data, y_data


def shuffle_two_visual_fields_dataset(
    x_data_left: np.array, x_data_right: np.array, y_data: np.array
) -> Tuple[np.array, np.array, np.array]:
    # Shuffles dataset comprised of three arrays

    n = len(y_data)
    unique_indices = np.random.permutation(
        n
    )  # generates random permutation of list in range(0, n)
    x_data_left = x_data_left[unique_indices]
    x_data_right = x_data_right[unique_indices]
    y_data = y_data[unique_indices]

    return x_data_left, x_data_right, y_data


def shuffle_and_double_dataset(
    x_data: np.array, y_data: np.array
) -> Tuple[np.array, np.array]:
    x_data1, y_data1 = shuffle_visual_field_dataset(x_data, y_data)
    x_data2, y_data2 = shuffle_visual_field_dataset(x_data, y_data)

    x_data_concatenated = np.concatenate((x_data1, x_data2))
    y_data_concatenated = np.concatenate((y_data1, y_data2))

    return x_data_concatenated, y_data_concatenated


def build_visual_field_data(
    x_data: np.array, y_data: np.array, n: float
) -> Tuple[np.array, np.array]:
    """
    Builds the dataset for a single visual field, choosing random values from the input.

    Input:
    x_data: np.array(np.ndarray): array of two-dimensional arrays corresponding to the pixel values of digits of the MNIST dataset.
    y_data: np.array(int): corresponding value of the digit represented by x_data.
    n: float: size of the final dataset

    Output:
    x_data_right: np.array(np.ndarray): array of two-dimensional arrays corresponding to the pixel values of digits of the MNIST dataset.
    y_data: np.array(int): corresponding value of the digit of the visual field.
    """

    original_size = len(y_data)
    random_indices = np.random.choice(np.arange(0, original_size), n)
    x_data_visual_field = x_data[random_indices]
    y_data_visual_field = y_data[random_indices]

    return x_data_visual_field, y_data_visual_field


def build_double_visual_fields_dataset(
    x_data: np.array,
    y_data: np.array,
    final_size: float = 1,
    proportion_cs: float = 0.5,
    proportion_left: float = 0.5,
    full_attention_value: float = 1,
    reduced_attention_value: float = 0.5,
    ss_attention_value: float = 0.5,
) -> Tuple[np.array, np.array, np.array]:
    """
    Builds an entire double visual fields dataset, comprised of two visual fields, left and right, and an array of the corresponding answer value for both visual fields.

    Input:
    x_data: np.array(np.ndarray): array of two-dimensional arrays corresponding to the pixel values of digits of the MNIST dataset.
    y_data: np.array(int): corresponding value of the digit represented by x_data.
    final_size: float: how many times the final dataset is bigger than the input data. Default is 4.
    proportion_cs: float: proportion of entries in the final dataset that have CS over SS. Default is 0.5.
    proportion_left: float: proportion of entries in the final dataset that have attention on the left visual field. Default is 0.5.
    full_attention_value: float: value of the full attention in CS. Default is 1.
    reduced_attention_value: float: value of the reduced attention in CS. Default is 0.5.
    ss_attention_value: float: value of the attention for SS. Default is 0.5.

    Output:
    x_data_left: np.array(np.ndarray): array of two-dimensional arrays corresponding to the pixel values of digits of the MNIST dataset with a determined attention.
    x_data_right: np.array(np.ndarray): array of two-dimensional arrays corresponding to the pixel values of digits of the MNIST dataset with a determined attention.
    y_data: np.array(int): corresponding value of the digit that has most attention considering both visual fields.
    """

    n = len(y_data) * final_size
    x_data_left, y_data_left = build_visual_field_data(x_data, y_data, n)
    x_data_right, y_data_right = build_visual_field_data(x_data, y_data, n)

    y_data_final = np.zeros(n, dtype=int)
    for i in range(n):
        data_with_cs = np.random.choice(
            [False, True], p=[1 - proportion_cs, proportion_cs]
        )
        data_with_left_attention = np.random.choice(
            [False, True], p=[1 - proportion_left, proportion_left]
        )

        # determines value of attention if dataset entry is CS or SS
        if data_with_cs:
            attention = full_attention_value
            no_attention = reduced_attention_value
        else:
            attention = ss_attention_value
            no_attention = 0

        # determines which visual field has attention
        if data_with_left_attention:
            x_data_left[i] *= attention
            x_data_right[i] *= no_attention
            y_data_final[i] = y_data_left[i]
        else:
            x_data_left[i] *= no_attention
            x_data_right[i] *= attention
            y_data_final[i] = y_data_right[i]

    return x_data_left, x_data_right, y_data_final