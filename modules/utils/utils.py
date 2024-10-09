import tensorflow as tf
import numpy as np
from typing import Tuple, List, Generator
import matplotlib.pyplot as plt
import time
import pickle as pkl
import os

def normalize_ndarray(array: np.ndarray) -> np.ndarray:
    max_value = np.max(array)
    min_value = np.min(array)

    if (max_value - min_value) == 0:
        return array
    else:
        return (array - min_value) / (max_value - min_value)
    
def normalize_dataset(training: np.ndarray, testing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes the training and testing datasets avoiding data leakage.

    Input:
    training: np.ndarray: training dataset
    testing: np.ndarray: testing dataset

    Output:
    Tuple[np.ndarray, np.ndarray]: normalized training and testing datasets
    """
    max_train = np.max(training)
    min_train = np.min(training)

    training_normalized = (training - min_train) / (max_train - min_train)
    testing_normalized = (testing - min_train) / (max_train - min_train)

    return training_normalized, testing_normalized

def split_array(array: np.array, percentage: float) -> Tuple[np.array, np.array]:
    """
    Splits an array in two given a percentage.

    Input:
    array: np.array: data to be split
    percentage: float: percentage of the data to be split in the second part

    Output:
    np.array: the first part of the split
    np.array: the second part of the split
    """
    n = array.shape[0]
    split_index = n - int(n * percentage)
    array1 = array[:split_index]
    array2 = array[split_index:]

    return array1, array2

def show_instance(*x_data: np.array, index: int, print_index: bool = False, y_data: np.array = None) -> None:
    """
    Displays a single instance of the dataset
    """
    num_axes = len(x_data) + (1 if print_index else 0) + (1 if y_data is not None else 0)
    fig, ax = plt.subplots(1, num_axes, figsize=(5, 5))
    
    ax = [ax] if num_axes == 1 else ax
    for i, data in enumerate(x_data):
        ax[i].imshow(data[index], cmap="binary", vmax=1)
        ax[i].set_title((("Left" if i == 0 else "Right")) + " digit" if num_axes > 1 else "Digit")
        ax[i].set_aspect("equal")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    index_ax = len(x_data)
    if print_index:
        ax[index_ax].set_title("Index")
        ax[index_ax].text(0.4, 0.5, f"{index}", fontsize=20)
        ax[index_ax].set_aspect("equal")
        ax[index_ax].axis("off")
    
    y_data_ax = index_ax + 1 if print_index else index_ax
    if y_data is not None:
        ax[y_data_ax].set_title("Digit")
        ax[y_data_ax].text(0.4, 0.5, f"{y_data[index]}", fontsize=20)
        ax[y_data_ax].set_aspect("equal")
        ax[y_data_ax].axis("off")
    
    plt.show()

def show_dataset(*x_data: np.array, y_data: np.array, num_images: int, print_index: bool = False) -> None:
    for i in range(num_images):
        show_instance(*x_data, index=i, print_index=print_index, y_data=y_data)

def get_current_time_string() -> str:
    current_time = time.localtime()
    return f'{current_time.tm_year}-{current_time.tm_mon}-{current_time.tm_mday}_{current_time.tm_hour}-{current_time.tm_min}-{current_time.tm_sec}'

def get_n_digits_indices(y_data, digit: int, n: int) -> Generator[int, None, None]:
    """
    Get the indices of n instances of a given digit in the dataset

    Inputs:
    y_data: np.array: the labels of the digits
    digit: int: the digit to display
    n: int: the number of instances of the digit to retreive
    
    Ouput:
    returns a list containing n indices of the digit in the dataset or an empty list if the digit is not found
    """
    max_size = len(y_data)
    idx = 0
    digits_found = 0
    while idx < max_size and digits_found < n:
        if y_data[idx] == digit:
            digits_found += 1
            yield idx
        idx += 1

def display_n_digits(x_data, y_data, digit: int, n: int) -> None:
    """
    Display n instances of a digit from the dataset and their corresponding indices in the dataset

    Inputs:
    x_data: np.array: the dataset of digits
    y_data: np.array: the labels of the digits
    digit: int: the digit to display
    n: int: the number of instances of the digit to display
    
    Ouput:
    displays n instances of the digit and their corresponding indices in the dataset
    """
    digit_indices = get_n_digits_indices(y_data, digit, n)
    for idx in digit_indices:
        show_instance(x_data, index=idx, print_index=True)

def plot_loss_accuracy(history):
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["loss"], label="Training Accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

def is_dvf_model(model):
    return type(model.input) == list # If input is a list, it is a double visual field model

def is_left_visual_field_model(model):
    return not is_dvf_model(model) and 'left' in model.input.name

def show_dataset_sizes(*data):
    for d in data:
        print(d.shape)

def compute_forward_pass(model: tf.keras.Model, *input_data: np.array) -> List[np.array]:
    '''
        Returns a list with the activations of each layer of the model for the input data.

        Input:
        model: tf.keras.Model: model to be used for prediction
        data: np.array: input data to be used for prediction, one array for each visual field
        
        Output: [np.array, ...]
    
    '''
    layers = model.layers
    activations = {}
    n_computed_layers = 0
    # Compute forward pass for the input layers
    match len(input_data):
        case 1:
            activations[layers[0].name], = input_data
            n_computed_layers = 1

        case 2:
            left_input, right_input = input_data
            activations = {layers[0].name: left_input, layers[1].name: right_input}
            n_computed_layers = 2
        
    # Compute forward pass for the rest of the layers
    for layer in layers[n_computed_layers:]:
        layer_type = layer.__class__.__name__
        layer_name = layer.name
        inbound_layers = layer._inbound_nodes[0].inbound_layers

        if type(inbound_layers) == list: # Inbound layers is a list when the layer has multiple inputs or no inputs
            match len(inbound_layers):
                case 0:
                    raise NotImplementedError('Layer with no inbound layers not implemented')
                case 1:
                    raise Exception('Node.inbound_layers should not return a list with one element')
                case 2:
                    match layer_type:
                        case 'Concatenate':
                            X = np.concatenate((activations[inbound_layers[0].name], activations[inbound_layers[1].name]), axis=0)
                            activations[layer_name] = X
                        case _:
                            raise NotImplementedError(f'Layer type {layer_type} not implemented')            
        else:
            inbound_layer_name = inbound_layers.name
            X = activations[inbound_layer_name]
            match layer_type:
                case 'Flatten':
                    X = X.flatten()
                    activations[layer_name] = X
                case 'Dense':
                    W, B = layer.get_weights()
                    g = layer.activation
                    Z = np.zeros((W.shape[1]))
                    for j in range(len(Z)):
                        for i in range(len(X)):
                            Z[j] += W[i][j] * X[i]
                        Z[j] += B[j]
                    if g.__name__ == 'softmax':
                        F = tf.nn.softmax(Z) # There is a bug in keras 2.15 that makes softmax not work on 1D array
                    else:
                        F = g(Z)
                    activations[layer_name] = F

    activations_numpy = []
    for activation in activations.values():
        try:
            activation = activation.numpy()
        except AttributeError:
            pass
        activations_numpy.append(activation)
        
    return activations_numpy

def save_obj(path: str, obj: any) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with open(path, 'wb') as f:
        pkl.dump(obj, f)

def load_obj(path: str) -> any:
    with open(path, 'rb') as f:
        return pkl.load(f)

def compute_cosine_similarity(vec_a: np.array, vec_b: np.array) -> float:
    """
    Computes the cosine similarity between two vectors

    Input:
    vec_a: np.array: first vector
    vec_b: np.array: second vector

    Output:
    float: cosine similarity between vec_a and vec_b
    """
    dot_product = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0: raise ValueError(f"Undefined cosine similarity value, product of vectors' norms is zero for vec_a={vec_a} and vec_b={vec_b}.")
    cs = dot_product / norm
    return cs

def is_left_visual_field_layer(layer: tf.keras.layers.Layer) -> bool:
    return 'left' in layer.name

def is_left_visual_field_layer_rec(layer: tf.keras.layers.Layer) -> bool:
    if 'input' in layer.name:
        return True if 'left' in layer.name else False
    else:
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        inbound_layer = inbound_layers if type(inbound_layers) != list else inbound_layers[0]
        return is_left_visual_field_layer_rec(inbound_layer)
    
def get_number_of_neurons_in_layer(layer: tf.keras.layers.Layer) -> int:
    """
    Returns the number of neurons present in a layer of a Neural Network
    """
    return layer.output_shape[-1]

def compute_activations(model: tf.keras.Model, *input_data: np.array) -> List[np.array]:
    """
    Generates the individual neuron activation values for an input of a model

    Input:
    model: tf.keras.Model: model to be used for prediction
    input_data: np.array: input data to be used for prediction, one array for each input

    Output:
    List[np.array]: list of activations for each layer of the model
    """

    # Creating intermediate models
    intermediate_layer_models = [
        tf.keras.models.Model(
            inputs=model.input, outputs=layer.output) for layer in model.layers
    ]

    # Reshaping input data to include channel info
    reshaped_input_data = [data.reshape((1, 28, 28)) for data in input_data]

    # Getting activations
    activations = [
        intermediate_layer_model(reshaped_input_data, training=False)[0] for intermediate_layer_model in intermediate_layer_models
    ]   # Model result is returned inside a list, hence we get the first element of that list

    return activations

def compute_digits_model_predicts(model: tf.keras.Model, *data: np.array) -> List[Tuple[int, float]]:
    '''
        Returns a list with the activations of the output layer of the model for each digit.
        The list is sorted in descending order of activation.

        Input:
        model: tf.keras.Model: model to be used for prediction
        *data: np.array: input data to be used for prediction, could be one or two arrays
        
        Output: [(digit, activation), ...]
    
    '''
    activations = compute_activations(model, *data)
    activations = activations[-1]
    digit_activations = []
    for digit, activation in enumerate(activations):
        digit_activations.append((digit, activation))
    
    digit_activations = sorted(digit_activations, key=lambda item: item[1], reverse=True)
        
    return digit_activations

def compute_mean_dynamically(mean_list: List[np.array], new_list: List[np.array], k: int) -> List[np.array]:
    """
    Computes the mean of a list of arrays dynamically using the formula:
    mean = (k * mean + new) / (k + 1)

    Input:
    mean_list: List[np.array]: list of arrays representing the mean to be calculated
    new_list: List[np.array]: list of arrays representing the new values to be added to the mean
    k: int: number of times the mean has been computed

    Output:
    List[np.array]: list of arrays representing the new mean
    """

    new_mean_list = []
    for mean, new in zip(mean_list, new_list):
        mean = (mean * k + new) / (k + 1)
        new_mean_list.append(mean)

    return new_mean_list

def data_generator_dvf(x_data_left, x_data_right, y_data, batch_size):
    """
    Used to generate batches of data for a double visual field model training or testing
    """
    while True:
        for i in range(0, len(y_data), batch_size):
            yield [x_data_left[i:i+batch_size], x_data_right[i:i+batch_size]], y_data[i:i+batch_size]

def data_generator_svf(x_data, y_data, batch_size):
    """
    Used to generate batches of data for a single visual field model training or testing
    """
    while True:
        for i in range(0, len(y_data), batch_size):
            yield x_data[i:i+batch_size], y_data[i:i+batch_size]