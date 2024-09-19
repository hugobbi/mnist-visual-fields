import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import List, Tuple, Optional
from attr import define
from modules.utils.utils import normalize_ndarray, get_current_time_string, compute_activations
import os

@define
class Position:
    """
    Dataclass representing a position with two coordinates
    """

    x: float
    y: float

    def copy(self):
        return Position(self.x, self.y)

    def __str__(self):
        return f"Position({self.x}, {self.y})"

@define
class PlottedNeuron:
    """
    Dataclass representing a neuron in the Neural Network plot
    """

    activation: float
    position: Position
    radius: float

class PlottedLayer(tf.keras.layers.Layer):
    """
    Class representing a layer in the Neural Network plot
    """

    def __init__(self, original_layer, activations: Optional[List[float]] = None, position:  Optional[Position] = None):
        super().__init__()
        # Copying attributes from original layer
        for attr in dir(original_layer):
            if not attr.startswith("__") and not callable(getattr(original_layer, attr)):
                setattr(self, attr, getattr(original_layer, attr))

        self.activations = activations
        self.position = position

    def __attrs_post_init__(self):
        self.neurons = ([])  # this is needed to fix bug where list is not empty at start
        self.num_neurons = self.output_shape[-1]

    def is_output_layer(self):
        return not bool(self._outbound_nodes)

    def set_y_position(self, y_position):
        self.position.y = y_position

@define
class PlottingControl:
    """
    Dataclass representing control variables for plotting the Neural Network
    """

    left_vf: List[PlottedLayer] = []
    right_vf: List[PlottedLayer] = []
    concatenated_vf: List[PlottedLayer] = []
    reference: Optional[List[PlottedLayer]] = None
    left_idx: int = 0
    right_idx: int = 0
    concat_idx: int = 0
    concat_left_idx: int = 0
    concat_right_idx: int = 0
    has_concatenated: bool = False
    is_concatenate_layer: bool = False
    plot_connections_left_vf: bool = True  # used in concatenate layer

class NeuralNetworkPlotter:
    def __init__(
            self, 
            model: tf.keras.Model, 
            max_neurons: int = 300,
            weight_threshold: float = 0.5,
            attribute_lenses: Optional[List[tf.keras.Model]] = None,
            num_attr_lenses_top_activations: int = 3,
            save_plots: bool = True,
            ) -> None:
        
        self.model = model
        self.max_neurons = max_neurons
        self.weight_threshold = weight_threshold
        self.attribute_lenses = attribute_lenses
        self.num_attr_lenses_top_activations = num_attr_lenses_top_activations
        self.save_plots = save_plots
        self._controller = None

    # Single visual field plot function
    def plot(self, data: np.array) -> None:
        pass

    # Double visual field plot function
    def plot(self, 
             left_vf_data: np.array, 
             right_vf_data: np.array,
             model: Optional[tf.keras.Model] = None,
             max_neurons: Optional[int] = None,
             weight_threshold: Optional[float] = None,
             attribute_lenses: Optional[List[tf.keras.Model]] = None,
             num_attr_lenses_top_activations: Optional[int] = None,
             save_plot: Optional[bool] = None) -> None:
        
        """
        Receives a trained model with two visual fields and an input, displaying the entire neural network with its
        activations for that input.

        Observation: the attention value for each visual field is applied in the function input data

        Input: 
        model: tf.keras.Model: model to be used for visualization
        left_vf_data: np.array: left visual field data
        right_vf_data: np.array: right visual field data
        max_neurons: int: maximum number of neurons to be plotted in a layer (default: 300)
        weight_threshold: float: minimum weight value to be plotted, considering normalized values between 0 and 1 (default: 0.5)
        attribute_lenses: Optional[List[tf.keras.Model]]: list of models to be used as attribute lenses for each layer (default: None)
        num_attr_lenses_top_activations: int: number of top digit activations displayed for each attribute lens (default: 3)
        save_plot: bool: if the neural network plot should be saved as an image (default: True)

        Output:
        displays and saves the image in results/images/ if requested
        """
        
        # Setting default values
        if model is None:
            model = self.model
        if max_neurons is None:
            max_neurons = self.max_neurons
        if weight_threshold is None:
            weight_threshold = self.weight_threshold
        if attribute_lenses is None:
            attribute_lenses = self.attribute_lenses
        if num_attr_lenses_top_activations is None:
            num_attr_lenses_top_activations = self.num_attr_lenses_top_activations
        if save_plot is None:
            save_plot = self.save_plots

        # Determining matplotlib figure parameters
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.axis("off")
        top, bottom, left, right = compute_figure_sizes(top=0.98)
        middle = round((top + bottom) / 2, ndigits=4)
        middle_spacing = 0.02

        # Calculating neural network activations
        model_activations = compute_activations(model, left_vf_data, right_vf_data)

        # Determining neural network figure parameters
        connection_opacity = 0.2
        color_connection_opacity = 0.5
        left_vf_position = Position(left, 1.5 * middle)  # initial position of left visual field
        right_vf_position = Position(left, 0.5 * middle)  # initial position of right visual field

        # Plotting layers
        model_number_of_layers = len(model.layers)
        self._controller = PlottingControl()

        for i, layer in enumerate(model.layers):
            match layer.name:
                case s if "input" in s:
                    if "left" in layer.name:
                        input_layer_left = PlottedLayer(layer, activations=model_activations[i])
                        input_layer_left.position = left_vf_position.copy()
                        ab = generate_image_annotation_box(
                            get_image(input_layer_left.activations),
                            input_layer_left.position,
                            cmap="binary",
                            size=calculate_image_size(input_layer_left.num_neurons),
                        )
                        ax.add_artist(ab)
                        self._controller.left_vf.append(input_layer_left)
                        self._controller.left_idx += 1
                    else:
                        pass
                case _:
                    break




def get_image(neural_activation: np.array):
    """
    Transforms the activity of the neurons of a Neural Network into a 2D matrix that can be seen as an image with plt.imshow
    """
    num_neurons = len(neural_activation)
    num_rows = int(np.sqrt(num_neurons))
    num_cols = int(np.ceil(num_neurons / num_rows))

    return np.array(neural_activation).reshape(num_rows, num_cols)

def generate_image_annotation_box(
    image: np.array, position: Position, size: int, cmap="binary", border_width: int = 1, border_color: str = "black"
) -> AnnotationBbox:
    """
    Generates the AnnotationBbox of the grayscale image with a border to be positioned in the figure
    """

    img = OffsetImage(
        image, zoom=size, cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1)
    )
    ab = AnnotationBbox(
        img,
        (position.x, position.y),
        frameon=True,
        pad=0,
        bboxprops=dict(
            edgecolor=border_color, linewidth=border_width, boxstyle="square,pad=0.1"
        ),
    )

    return ab

def get_digit_from_y_spacing(total_position_plotted: float, y_spacing: float) -> int:
    """
    Returns the digit an output neuron represents based on how many neurons have been plotted
    """
    return np.ceil(total_position_plotted / y_spacing)

def calculate_image_size(number_neurons: int) -> float:
    """
    Computes size of image based on the number of neurons
    """
    # Determined in tests
    a = -5/1536
    b = 533/96
    return number_neurons * a + b

def compute_figure_sizes(top: float) -> List[float]:
    """
    Computes sizes of the figure based on the top percentage of the figure
    """
    bottom = 1 - top
    return top, bottom, bottom, top

def generate_output_models(model: tf.keras.Model) -> List[tf.keras.Model]:
    """
    Generates output models for each hidden layer of the model
    """
    # Generate not trainable copy of model
    model_copy = tf.keras.models.clone_model(model)
    model_weights = model.get_weights()
    model_copy.set_weights(model_weights)
    for layer in model_copy.layers:
        layer.trainable = False 
    
    # Generate output models of each hidden layer
    output_models = []
    for layer in model_copy.layers[4:-2]:
        out_model_input = model_copy.input # [left_input, right_input]
        if 'left' in layer.name:
            out_model_input = out_model_input[0]
        elif 'right' in layer.name:
            out_model_input = out_model_input[1]
            
        out_model_output = tf.keras.layers.Dense(10, activation="softmax")(layer.output)
        out_model = tf.keras.models.Model(
        inputs=out_model_input, outputs=out_model_output
        )
        output_models.append(out_model)

    return output_models

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

def display_single_visual_field_mnist_nn_execution(
    model, x_data, y_data, input_digit, max_neurons, weight_plot_threshold
):
    """
    Receives a trained Neural Network model and and input digit, plotting the full neural network execution
    """

    """ Determining matplotlib figure parameters """

    fig = plt.figure(
        figsize=(12, 12)
    )  # TODO: generate size based on max number of neurons on layer
    ax = fig.gca()
    ax.axis("off")
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9

    """ Calculating neural network activations """

    model_activations = compute_activations(
        model, input_digit, x_data, y_data
    )

    """ Determining neural network figure parameters """

    model_number_of_layers = len(model.layers)
    layer_spacing = (right - left) / (model_number_of_layers - 1)  # Space between each layer in the plot
    connection_opacity = 0.2
    color_connection_opacity = 0.5
    image_y_position = 0.5
    current_position = Position(left, image_y_position)  # Initial position

    """ Plotting layers """

    plotted_layers = []
    for i in range(model_number_of_layers):
        model_layer = model.layers[i]
        layer_activations = model_activations[i]  # TODO: Can raise error if number of activations not equal to number of layers

        layer = Layer(model=model_layer, activations=layer_activations)
        plotted_layers.append(layer)  # Adding layer to plotted layers

        # Depending on the type of layer, different  figures will be created to represent it

        # -- Flatten layer --
        if "flatten" in layer.model.name:
            # If it is the input layer
            if i == 0:
                layer.position = current_position.copy()
                ab = generate_image_annotation_box(
                    get_image(layer.activations),
                    layer.position,
                    size=calculate_image_size(layer.num_neurons),
                )
                ax.add_artist(ab)

        #  -- Dense layer --
        elif "dense" in layer.model.name:

            # -- Plotting neurons of dense layer --
            if layer.num_neurons < max_neurons:  # Plotting individual neurons
                current_position.x += layer_spacing
                current_position.y = top
                layer.position = current_position.copy()

                previous_layer = plotted_layers[i - 1]  # Gets previous layer
                layer_weights = normalize_ndarray(layer.model.get_weights()[0])

                neuron_spacing = (top - bottom) / layer.num_neurons  # Space between each neuron in the plot
                layer_activations = (
                    normalize_ndarray(layer_activations)
                    if not layer.is_output_layer()
                    else layer_activations
                )  # Normalize layer activations if it is not output layer
                for idx, neuron_activation in enumerate(layer_activations):
                    # -- Plotting each neuron --
                    neuron = Neuron(
                        neuron_activation,
                        current_position.copy(),
                        radius=neuron_spacing / 4,
                    )
                    layer.neurons.append(neuron)  # Adding neuron to plotted layer
                    neuron_circle = plt.Circle(
                        xy=(current_position.x, current_position.y),
                        radius=neuron.radius,
                        color=plt.cm.viridis(neuron_activation),
                        ec="k",
                    )
                    ax.add_artist(neuron_circle)  # Plots neuron

                    # -- Plotting connections of neurons to previous layer --
                    if (previous_layer.num_neurons < max_neurons):  # Checks if the number of neurons on the previous layer doesn't exceed the maximum to be plotted)
                        layer_weights_neuron = layer_weights[:, idx]
                        for previous_neuron, connection_weight in zip(previous_layer.neurons, layer_weights_neuron):
                            # Checks if weight is above threshold
                            if connection_weight >= weight_plot_threshold:
                                connection = plt.Line2D(
                                    [
                                        current_position.x - neuron.radius,
                                        previous_neuron.position.x + previous_neuron.radius,
                                    ],
                                    [current_position.y, previous_neuron.position.y],
                                    color=plt.cm.viridis(connection_weight),
                                    alpha=color_connection_opacity,
                                )
                                ax.add_artist(connection)  # Plots connection to previous neurons

                    else:  # In case it does, only a single conncetion from each neuron will be shown connecting to the previous layer
                        previous_layer_position = Position(previous_layer.position.x, previous_layer.position.y)  # Saves position of previous layer
                        image_offset = 0.065
                        connection = plt.Line2D(
                            [
                                current_position.x - neuron.radius,
                                previous_layer_position.x + image_offset
                            ],
                            [current_position.y, previous_layer_position.y],
                            color="k",
                            alpha=connection_opacity,
                        )
                        ax.add_artist(connection)  # Plots connection to image

                    # -- Plotting digit and activation if it is the output layer --
                    if i == model_number_of_layers - 1:
                        neuron_digit = get_digit_from_y_spacing(
                            (top - current_position.y), neuron_spacing
                        )
                        text_offset_x = 0.001
                        text_offset_y = 0.005
                        text = plt.Text(
                            current_position.x + neuron.radius + text_offset_x,
                            current_position.y - text_offset_y,
                            f"{neuron_digit}: {neuron_activation:.4f}",
                            fontsize=8,
                            color="k",
                        )
                        ax.add_artist(text)  # Adds corresponding digit and activation of neuron

                    # Changes the current y position to plot the next neuron
                    current_position.y -= neuron_spacing

            else:  # If the number of neurons in the layer exceeds the maximum TODO: Color connections if previous layer has few neurons
                current_position.x += layer_spacing
                current_position.y = image_y_position
                layer.position = current_position.copy()

                # -- Plotting layer as image --
                ab = generate_image_annotation_box(
                    get_image(layer.activations),
                    layer.position,
                    cmap="viridis",
                    size=calculate_image_size(layer.num_neurons),
                )
                ax.add_artist(ab)

                # -- Plotting connections of image to previous layer --
                previous_layer = plotted_layers[i - 1]  # Gets previous layer
                if (previous_layer.num_neurons < max_neurons):  # Checks if the number of neurons on the previous layer doesn't exceed the maximum to be plotted
                    for previous_neuron in previous_layer.neurons:
                        connection = plt.Line2D(
                            [
                                current_position.x,
                                previous_neuron.position.x + previous_neuron.radius,
                            ],
                            [current_position.y, previous_neuron.position.y],
                            color="k",
                            alpha=connection_opacity,
                        )
                        ax.add_artist(connection)  # Plots connection to previous neurons
                else:  # If previous layer exceeds the maximum number of neurons plotted
                    connection = plt.Line2D(
                        [current_position.x, previous_layer.position.x],
                        [current_position.y, previous_layer.position.y],
                        color="k",
                        alpha=connection_opacity,
                    )
                    ax.add_artist(connection)  # Plots connection to previous neurons

    os.makedirs("images", exist_ok=True)
    fig.savefig(f"images/nn_svf_{get_current_time_string()}.png")