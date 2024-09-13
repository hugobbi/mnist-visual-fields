import tensorflow as tf
import numpy as np
import os
import re
from typing import Tuple, List, Generator, Dict
from multiprocessing import Pool
from utils.utils import get_n_digits_indices, compute_activations, compute_forward_pass, compute_mean_dynamically, is_double_visual_field_model, compute_cosine_similarity, is_left_visual_field_layer
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from itertools import combinations

def args_generator(is_drf_model: bool, x_data: np.array, digit_instances: List[int | None], n: int) -> Generator[Tuple[np.array, np.array, Tuple[int, int, int]], None, None]:
    """
    Generates the arguments to be used in prototype calculation

    Input:
    x_data: np.array: input dataset to be used for prototype generation
    digit_instances: List[int | None]: list of indices of the instances of each digit
    n: int: number of instances of a digit to be used to compute the prototype for that digit

    Output:
    Tuple[np.array, Tuple[int, int, bool]]: tuple containing the input data, the digit pair and the input type
    """

    for d, d_instances in enumerate(digit_instances):
        for i in range(n):
            idx = next(d_instances)
            if is_drf_model:
                yield (x_data[idx], (d, i, 0)) # Data for left visual field (input type 0)
                yield (x_data[idx], (d, i, 1)) # Data for right visual field (input type 1)
            else:
                yield (x_data[idx], (d, i, 2)) # Data for single visual field (input type 2)

def generate_prototypes(model: tf.keras.Model, x_data: np.array, y_data: np.array, n: int) -> List[List[np.array]]:
    """
    Returns a list containing the prototypes for each digit in each layer of the model

    Input:
    model: tf.keras.Model: model to be used for prototype generation
    data: np.array: input dataset to be used for prototype generation
    n: int: number of instances of a digit to be used to compute the prototype for that digit

    Output: 
    List[List[np.array]]: list containing the prototypes for each digit in each layer of the model
    """
    
    print("[Cosine similarity] Starting prototype calculation...")
    
    prototypes = {}
    digit_instances_generator = (get_n_digits_indices(y_data, d, n) for d in range(10))
    is_drf_model = is_double_visual_field_model(model)
    args = args_generator(is_drf_model, x_data, digit_instances_generator, n)
    for (data, (d, i, input_type)) in args:
        match input_type:
            case 0:
                activations = compute_activations(model, data, np.zeros_like(data))
                key = f"{d}_l"
            case 1:
                activations = compute_activations(model, np.zeros_like(data), data)
                key = f"{d}_r"
            case 2:
                activations = compute_activations(model, data)
                key = f"{d}"
        prototypes[key] = activations if i == 0 else compute_mean_dynamically(prototypes[key], activations, i)

    # Sort prototypes by visual field. This is done to make cosine similarity matrix calculations easier later on
    if is_drf_model: prototypes = {k: prototypes[k] for k in sorted(prototypes, key=lambda p: p.split('_')[1])}
    print("[Cosine similarity] Done prototype calculation!")

    return prototypes

def compute_activations_mp(args: Tuple[np.array, np.array, Tuple[int, int, int]]) -> Tuple[List[np.array], Tuple[int, int, int]]:
    """
    Computes activations for each arg in multiprocess pool

    Input:
    args: Tuple[np.array, Tuple[int, int, bool]]: tuple containing the input data, the digit pair and the input type

    Output:
    Tuple[List[np.array], Tuple[int, int, int]]: tuple containing list of arrays containing the activations for each layer of the 
    model, the digit pair for the activations and the input type
    """

    data, (digit, i, input_type) = args
    global g_model
    match input_type:
        case 0:
            activations = compute_forward_pass(g_model, data, np.zeros_like(data))
        case 1:
            activations = compute_forward_pass(g_model, np.zeros_like(data), data)
        case 2:
            activations = compute_forward_pass(g_model, data)

    return activations, (digit, i, input_type)

def generate_prototypes_mp(model: tf.keras.Model, x_data: np.array, y_data: np.array, n: int) -> Dict[str, List[np.array]]:
    """
    Returns a list containing the prototypes for each digit in each layer of the model. Uses multiprocessing to speed up
    the calculations

    Input:
    model: tf.keras.Model: model to be used for prototype generation
    data: np.array: input dataset to be used for prototype generation
    n: int: number of instances of a digit to be used to compute the prototype for that digit

    Output: 
    List[List[np.array]]: list containing the prototypes for each digit in each layer of the model
    """
    
    print("[Cosine similarity] Starting multiprocess prototype calculation...")

    prototypes = {}
    digit_instances_generator = (get_n_digits_indices(y_data, d, n) for d in range(10))
    is_drf_model = is_double_visual_field_model(model)
    args_len = 20*n if is_drf_model else 10*n

    # Model needs to be global in order for multiprocessing to work (loading model from file is causing deadlock)
    global g_model
    g_model = model
    # Using generator for the args instead of list to save memory
    args = args_generator(is_drf_model, x_data, digit_instances_generator, n)
    workers = os.cpu_count()
    with Pool(processes=workers) as pool:
        for activations, (d, i, input_type) in pool.imap(compute_activations_mp, args, chunksize=args_len//workers):
            key = f"{d}_l" if input_type == 0 else f"{d}_r" if input_type == 1 else f"{d}"
            prototypes[key] = activations if i == 0 else compute_mean_dynamically(prototypes[key], activations, i)
    del g_model # Save memory

    # Sort prototypes by visual field. This is done to make cosine similarity matrix calculations easier later on
    if is_drf_model: prototypes = {k: prototypes[k] for k in sorted(prototypes, key=lambda p: p.split('_')[1])}
    print("[Cosine similarity] Done multiprocess prototype calculation!")
    
    return prototypes

def key_to_idx(key: str) -> int:
    """
    Converts a key to the corresponding index. The key represents a digit and the visual field (left or right)

    Input:
    key: str: key to be converted

    Output:
    int: index corresponding to the key
    """

    d, vf = key.split('_')
    return int(d) if vf == 'l' else int(d) + 10

def plot_csm(cs_matrices: Dict[str, np.array], layer_name: str, title: str = None, figsize: Tuple[float, float]=(10, 10), not_computed_color: str='black') -> None:
    """
    Plots the cosine similarity matrix for a given layer

    Input:
    cs_matrices: Dict[str, np.array]: dictionary containing the cosine similarity matrices for each layer of the model
    layer_name: str: name of the layer to be plotted
    title: str: title of the plot (default: None)
    figsize: Tuple[float, float]: size of the plot (default: (10, 10))
    not_computed_color: str: color to be used for values that were not computed (default: 'black')
    """

    csm = cs_matrices[layer_name]
    n = csm.shape[0]

    # Masking lower triangle of matrix
    mask = np.tril(np.ones_like(csm, dtype=bool))
    masked_csm = np.ma.masked_array(csm, mask)

    # Creating colormap and setting color for not computed values
    cmap = plt.cm.gray
    cmap.set_bad(color=not_computed_color)

    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(masked_csm, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(cax)

    # Labels
    ax.set_title(title if title else f'{layer_name} CSM')
    ax.set_xlabel('Digit 1')
    ax.set_ylabel('Digit 2')
    
    # Creating grid to separate each matrix entry
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

    # Creating labels for each digit
    if n == 20:
        labels = [f'{i}_l' if i < 10 else f'{i-10}_r' for i in range(n)]
    else:
        labels = [f'{i}' for i in range(n)]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    plt.show()

def get_csm_value(csm: np.array, d1: str, d2: str) -> float:
    """
    Returns the cosine similarity value between two digits in a layer

    Input:
    d1: str: key representing the first digit
    d2: str: key representing the second digit

    Output:
    float: cosine similarity value between the two digits
    """

    idx1 = key_to_idx(d1)
    idx2 = key_to_idx(d2)

    return csm[idx1, idx2]

def plot_csm_interactively(cs_matrices: Dict[str, List[np.array]], figsize: Tuple[float, float]=(10, 10), not_computed_color: str='black') -> None:
    """
    Plots the cosine similarity matrix for a given layer interactively, allowing the user to select the layer to be plotted

    Input:
    cs_matrices: Dict[str, np.array]: dictionary containing the cosine similarity matrices for each layer of the model
    title: str: title of the plot (default: None)
    figsize: Tuple[float, float]: size of the plot (default: (10, 10))
    not_computed_color: str: color to be used for values that were not computed (default: 'black')
    """
    def on_layer_name_change(change):
        layer_name = change['new']
        output.clear_output(wait=True)
        with output:
            plot_csm(cs_matrices=cs_matrices, layer_name=layer_name, figsize=figsize, not_computed_color=not_computed_color)

    layer_names = list(cs_matrices.keys())
    layer_name_selector = widgets.Dropdown(
        options=layer_names,
        value=layer_names[0], # default value is the first layer name
        description='Layer:',
    )

    layer_name_selector.observe(on_layer_name_change, names='value')
    output = widgets.Output()
    display(layer_name_selector, output)

    # Initial plot
    with output:
        plot_csm(cs_matrices=cs_matrices, layer_name=layer_names[0], figsize=figsize, not_computed_color=not_computed_color)

def compute_cosine_similarity_matrix(model: tf.keras.Model, prototypes: Dict[str, List[np.array]]) -> Dict[str, np.array]:
    """
    Computes the cosine similarity matrix for each layer of the model

    Input:
    model: tf.keras.Model: model to be used for cosine similarity calculation
    prototypes: List[List[np.array]]: list containing the prototypes for each digit in each layer of the model

    Output:
    List[np.array]: list containing the cosine similarity matrix for each layer of the model
    """
    is_drf_model = is_double_visual_field_model(model)
    is_drf_layer = False
    cs_matrices = {}
    for i, layer in enumerate(model.layers):
        # Ignore input and flatten layers
        if 'input' in layer.name or 'flatten' in layer.name: continue
        
        # Determine if this and subsequent layers are DRF layers
        if is_drf_model and 'concatenate' in layer.name:
            is_drf_layer = True
        
        if is_drf_layer:
            # Combinations of digits in each visual field
            digit_vf_combinations = combinations(prototypes.keys(), 2)
            cs_matrix = np.zeros((20, 20))
            for d1_vf, d2_vf in digit_vf_combinations:
                idx1, idx2 = key_to_idx(d1_vf), key_to_idx(d2_vf)
                cs_matrix[idx1, idx2] = compute_cosine_similarity(prototypes[d1_vf][i], prototypes[d2_vf][i])
        else:
            # Combinations of digits
            digit_combinations = combinations(range(10), 2)
            cs_matrix = np.zeros((10, 10))
            for (d1, d2) in digit_combinations:
                if is_drf_model:
                    # Determine if layer is left visual field
                    if is_left_visual_field_layer(layer):
                        cs_matrix[d1, d2] = compute_cosine_similarity(prototypes[f'{d1}_l'][i], prototypes[f'{d2}_l'][i]) 
                    else:
                        cs_matrix[d1, d2] = compute_cosine_similarity(prototypes[f'{d1}_r'][i], prototypes[f'{d2}_r'][i])
                else:
                    cs_matrix[d1, d2] = compute_cosine_similarity(prototypes[f'{d1}'][i], prototypes[f'{d2}'][i])

        cs_matrices[layer.name] = cs_matrix.copy()
    
    return cs_matrices

def compute_avg_off_diagonal(matrix: np.array) -> float:
    """
    Computes the average off-diagonal value of a matrix

    Input:
    matrix: np.array: matrix to compute the average off-diagonal value

    Output:
    float: average off-diagonal value of the matrix
    """

    n = matrix.shape[0]
    off_diagonal_elements = matrix[np.triu_indices(n, k=1)] # np.triu_indices returns indices of upper triangle of matrix (k = 1 to exclude main diagonal)
    avg_off_diagonal = np.mean(off_diagonal_elements)
    
    return avg_off_diagonal

def get_quadrants(matrix: np.array) -> Tuple[np.array, np.array, np.array, np.array]:
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

def compute_avg_without_diagonal(matrix: np.array) -> float:
    """
    Computes the average value of a matrix without the diagonal values

    Input:
    matrix: np.array: matrix to compute the operation on

    Output:
    float: average value of the matrix without the diagonal
    """
    n = matrix.shape[0]
    diagonal_elements = np.diag(matrix)
    avg_without_diagonal = (np.sum(matrix) - np.sum(diagonal_elements)) / (n**2 - n)

    return avg_without_diagonal

def compute_orthogonality(csm: np.array, separate_drf: bool = False) -> float | Tuple[float, float, float]:
    """
        Computes the orthogonality of a layer

        Input:
        csm: np.array: cosine similarity matrix for a layer
        separate_drf: bool: flag to separate the orthogonality of the DRF layer into its components (default: False)

        Output:
        float: orthogonality of the layer
    """
    n = csm.shape[0]
    # If the layer is a DRF layer
    if n == 20:
        upper_left, upper_right, lower_left, lower_right = get_quadrants(csm)
        oleft = 1 - compute_avg_off_diagonal(upper_left)
        oright = 1 - compute_avg_off_diagonal(lower_right)
        ocross = np.mean(np.diag(upper_right)) - compute_avg_without_diagonal(upper_right)
        if separate_drf: return oleft, oright, ocross
        o_drf = (oleft + oright + ocross) / 3
        return o_drf
    else:
        o_srf = 1 - compute_avg_off_diagonal(csm)
        return o_srf

def compute_orthogonality_grouping_layers(cs_matrices: Dict[str, np.array], group_by_rf: bool = True, is_srf_model: bool = False, separate_drf: bool = True) -> Dict[str, float | Tuple[float, float, float]]:
    """
    Computes the orthogonality for each group of layers in the model

    Input:
    cs_matrices: Dict[str, np.array]: dictionary containing the cosine similarity matrices for each layer of the model
    group_by_rf: bool: flag to group layers by visual field (left_srf, right_srf, drf or only srf) (default: True)
    is_srf_model: bool: flag to indicate if the model is a single visual field model (default: False)
    separate_drf: bool: flag to separate the orthogonality of the DRF layer into its components (default: False)

    Output:
    Dict[str, float]: dictionary containing the orthogonality for each group of layers in the model
    """

    orthogonalities = {}
    for layer_name, csm in cs_matrices.items():                
        if group_by_rf:
            if is_srf_model:
                key = 'srf'
            else:
                key = 'left_srf' if 'left' in layer_name else 'right_srf' if 'right' in layer_name else 'drf'
        else:
            key = re.sub(r'\d+', '', layer_name)
        orth = compute_orthogonality(csm, separate_drf)
        orthogonalities[key] = [orth] if key not in orthogonalities else orthogonalities[key] + [orth]

    return orthogonalities

def compute_mean_and_deviation(values_list: List[Dict[str, np.array]]) -> Tuple[Dict[str, np.array], Dict[str, np.array]]:
    grouped_values = {}
    for values in values_list:
        for key, value in values.items():
            grouped_values[key] = [value] if key not in grouped_values else grouped_values[key] + [value]

    deviations = {}
    mean_values = {}
    for key in grouped_values.keys():
        deviations[key] = np.std(grouped_values[key], axis=0)
        mean_values[key] = np.mean(grouped_values[key], axis=0)
    
    return mean_values, deviations

def plot_orthogonality(orthogonalities: Dict[str, np.array], deviations: Dict[str, np.array] = None, label: str = '') -> None:
    label = "Orthogonality for each group of layers" if label == '' else label
    alpha = 0.2
    plt.figure()
    for key, orth in orthogonalities.items():
        drf_keys = {'drf', 'dense'}
        x_size = len(orth)
        x_values = np.arange(x_size) if key not in drf_keys else np.arange(x_size-2, 2*x_size-2)
        if len(orth.shape) == 2:
            labels = {0: "oleft", 1: "oright", 2: "ocross"}
            for i in range(3):
                plt.plot(x_values, orth[:,i], label=labels[i])
                if deviations is not None:
                    plt.fill_between(x_values, orth[:,i] - deviations[key][:,i], orth[:,i] + deviations[key][:,i], alpha=alpha)
        else:
            if deviations is not None:
                plt.fill_between(x_values, orth - deviations[key], orth + deviations[key], alpha=alpha)
            plt.plot(x_values, orth, label=key)
    plt.title(label)
    plt.xlabel('Layer instance')
    plt.ylabel('Orthogonality')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

class EpochStopping(tf.keras.callbacks.Callback):
    def __init__(self, max_epochs: int, acc_diff_threshold: float = 0.01, n_accs: int = 5, max_train_acc: float | None = None):
        super(EpochStopping, self).__init__()
        self._prev_accs = []
        self._current_acc = 0
        self._acc_diff_threshold = acc_diff_threshold
        self._n_accs = n_accs
        self._max_train_acc = max_train_acc
        self._max_epochs = max_epochs

        self.epochs = []
        self.previous_epochs = []

    @property
    def num_epochs(self) -> int:
        return len(self._prev_accs)
    
    def reset(self) -> None:
        self.previous_epochs.append(self.num_epochs)
        self._prev_accs = [] 
        self._current_acc = 0
    
    def at_max_train_acc(self) -> bool:
        if self._max_train_acc is None:
            return False
        return self._current_acc >= self._max_train_acc
    
    def get_mean_prev_n_accs(self, n: int) -> float:
        return np.mean(self._prev_accs[-n:])

    def not_changed_within_threshold(self) -> bool:
        if self.num_epochs >= self._n_accs:
            for prev_acc in self._prev_accs[-self._n_accs:]:
                if np.abs(self._current_acc - prev_acc) > self._acc_diff_threshold:
                    return False
            return True
        return False
    
    def should_stop(self) -> bool:
        return self.at_max_train_acc() or self.num_epochs >= self._max_epochs or self.not_changed_within_threshold()

    def on_epoch_end(self, batch, logs={}) -> None:
        self._current_acc = logs.get('accuracy')
        self._prev_accs.append(self._current_acc)
        if self.should_stop():
            self.reset()
            self.model.stop_training = True