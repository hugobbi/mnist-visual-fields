import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import List, Tuple, Optional, Set
from attr import define
from modules.utils.utils import normalize_ndarray, get_current_time_string, compute_activations, compute_digits_model_predicts
from collections.abc import Iterable

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

class PlottedLayer():
    """
    Class representing a layer in the Neural Network plot
    """

    def __init__(self, original_layer: tf.keras.layers.Layer, activations: Optional[List[float]] = None, position:  Optional[Position] = None):
        # Copying necessary attributes from original layer
        self.name = original_layer.name
        self.weights = original_layer.get_weights()
        self.activations = activations
        self.position = position if position is not None else Position(0, 0)
        self.num_neurons = self.__compute_num_neurons(original_layer.output_shape[-1])
        self.neurons = []
        self.type = original_layer.__class__.__name__
        self.is_input_layer = self.type == "InputLayer"
        self.is_output_layer = not bool(original_layer._outbound_nodes)
        self.is_concatenate_layer = self.type == "Concatenate"
        self.previous_layers = self.__get_previous_layers(original_layer)

    # def __attrs_post_init__(self):
    #     self.neurons = ([])  # this is needed to fix bug where list is not empty at start

    def __str__(self):
        return f"PlottedLayer({self.name}, {self.type}, {self.num_neurons}, is_input={self.is_input_layer}, is_output={self.is_output_layer}, is_concat={self.is_concatenate_layer})"
    
    def __compute_num_neurons(self, num_neurons) -> int:
        if (isinstance(num_neurons, Iterable)):
            return (lambda x, y, z: y * z)(*num_neurons)
        return int(num_neurons)
    
    def __get_previous_layers(self, layer: tf.keras.layers.Layer) -> List[tf.keras.layers.Layer]:
        previous_layers = layer._inbound_nodes[0].inbound_layers
        if type(previous_layers) == list:
            return previous_layers
        return [previous_layers]
    
    def get_previously_plotted_layer(self, plotted_layers: List["PlottedLayer"]) -> Optional["PlottedLayer"]:
        pl_queue = self.previous_layers
        while pl_queue:
            pl = pl_queue.pop(0)
            #print(pl.name, set(map(lambda plt_layer: plt_layer.name, plotted_layers)))
            for plt_layer in plotted_layers:
                if plt_layer.name == pl.name:
                    return plt_layer
            pl_previous_layers = self.__get_previous_layers(pl)
            pl_queue.extend(pl_previous_layers)
            
        return None

@define
class PlottingControl:
    """
    Dataclass representing control variables for plotting the Neural Network
    """

    plotted: List[PlottedLayer] = []
    left_vf: List[PlottedLayer] = []
    right_vf: List[PlottedLayer] = []
    concatenated_vf: List[PlottedLayer] = []
    reference: Optional[List[PlottedLayer]] = None
    is_left_vf: bool = False
    idx: int = 0
    left_idx: int = 0
    right_idx: int = 0
    concat_idx: int = 0
    concat_left_idx: int = 0
    concat_right_idx: int = 0
    has_concatenated: bool = False
    plot_connections_left_vf: bool = True  # used in concatenate layer
    vf_top: float = 0
    vf_bottom: float = 0
    SHOULD_PLOT = {"InputLayer", "Dense", "Concatenate", "Conv2D"}

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
        self.__controller = None

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
        top, bottom, left, right = self.__compute_figure_sizes(top=0.98)
        middle = round((top + bottom) / 2, ndigits=4)
        MIDDLE_SPACING = 0.02

        # Calculating neural network activations
        model_activations = compute_activations(model, left_vf_data, right_vf_data)

        # Determining neural network figure parameters
        CONECTION_OPACITY = 0.2
        COLOR_CONECTION_OPACITY = 0.5
        INITIAL_LEFT_VF_POSITION = Position(left, 1.5 * middle) 
        INITIAL_RIGHT_VF_POSITION = Position(left, 0.5 * middle)  
        model_number_of_layers = len(model.layers)

        # Setting up control structure to plot layers
        self.__controller = PlottingControl()
        for i, (layer, activations) in enumerate(zip(model.layers, model_activations)):
            plotted_layer = PlottedLayer(original_layer=layer, activations=activations)
            if plotted_layer.type not in self.__controller.SHOULD_PLOT: continue # only plots layers that should be plotted
            # Determining layer characteristics
            self.__controller.is_left_vf = i % 2 == 0 and not (plotted_layer.is_concatenate_layer or self.__controller.has_concatenated)
            self.__controller.vf_top = top
            self.__controller.vf_bottom = bottom
            # Left VF
            if self.__controller.is_left_vf:
                self.__controller.vf_bottom = middle + MIDDLE_SPACING
            # Right VF
            elif not (plotted_layer.is_concatenate_layer or self.__controller.has_concatenated):
                self.__controller.vf_top = middle - MIDDLE_SPACING
            # Input
            if plotted_layer.is_input_layer:
                plotted_layer.position = INITIAL_LEFT_VF_POSITION if self.__controller.is_left_vf else INITIAL_RIGHT_VF_POSITION
                ab = self.__generate_image_annotation_box(
                            plotted_layer.activations,
                            plotted_layer.position,
                            cmap="binary",
                            size=self.__calculate_image_size(plotted_layer.num_neurons),
                        )
                ax.add_artist(ab)
                self.__controller.plotted.append(plotted_layer)
                continue
            # Concatenate
            if plotted_layer.is_concatenate_layer:
                self.__controller.has_concatenated = True
            # Output
            if plotted_layer.is_output_layer:
                output_offset = 0.2
                self.__controller.vf_top = middle + output_offset
                self.__controller.vf_bottom = middle - output_offset
                        
            # Getting previosuly plotted layer
            layer_spacing = 0.05 if len(self.__controller.plotted) < 4 else 1.3 / (model_number_of_layers-4) # adjust layer spacing based on current layer
            previous_layer = plotted_layer.get_previously_plotted_layer(self.__controller.plotted)
            print(plotted_layer.name, previous_layer.name)
            plotted_layer.position.x = previous_layer.position.x + layer_spacing
            plotted_layer.position.y = self.__controller.vf_top

            if plotted_layer.num_neurons <= max_neurons:
                print(self.__controller.vf_top, self.__controller.vf_bottom)
                neuron_spacing = (self.__controller.vf_top - self.__controller.vf_bottom) / plotted_layer.num_neurons
                neuron_position = plotted_layer.position.copy()
                layer_activations = normalize_ndarray(plotted_layer.activations) if not plotted_layer.is_output_layer else plotted_layer.activations
                for j, neuron_activation in enumerate(layer_activations):
                    neuron = PlottedNeuron(neuron_activation, position=neuron_position.copy(), radius=neuron_spacing/4)
                    plotted_layer.neurons.append(neuron)
                    neuron_circle = plt.Circle(
                        xy=(neuron.position.x, neuron.position.y),
                        radius=neuron.radius,
                        color=plt.cm.viridis(neuron_activation),
                        ec="k",
                    )
                    ax.add_artist(neuron_circle)
                    neuron_position.y -= neuron_spacing
            else:
                ab = self.__generate_image_annotation_box(
                            plotted_layer.activations,
                            plotted_layer.position,
                            cmap="viridis",
                            size=self.__calculate_image_size(plotted_layer.num_neurons),
                        )
                ax.add_artist(ab)
            
            self.__controller.plotted.append(plotted_layer) # add current layer to plotted layers list
        plt.show()

    def __compute_figure_sizes(self, top: float) -> List[float]:
        """
        Computes sizes of the figure based on the top percentage of the figure
        """
        bottom = 1 - top
        return top, bottom, bottom, top
    
    def __generate_image_annotation_box(
        self, image: np.array, position: Position, size: int, cmap="binary", border_width: int = 1, border_color: str = "black"
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
    
    def __get_image(self, neural_activation: np.array):
        """
        Transforms 1D array into 2D array to be plotted as an image
        """
        num_neurons = len(neural_activation)
        num_rows = int(np.sqrt(num_neurons))
        num_cols = int(np.ceil(num_neurons / num_rows))

        return np.array(neural_activation).reshape(num_rows, num_cols)

    def __get_digit_from_y_spacing(self, total_position_plotted: float, y_spacing: float) -> int:
        """
        Returns the digit an output neuron represents based on how many neurons have been plotted
        """
        return np.ceil(total_position_plotted / y_spacing)

    def __calculate_image_size(self, number_neurons: int) -> float:
        """
        Computes size of image based on the number of neurons
        """
        # Determined in tests
        a = -5/1536
        b = 533/96
        return number_neurons * a + b

    def generate_output_models(self, model: Optional[tf.keras.Model] = None) -> List[tf.keras.Model]:
        """
        Generates output models for each hidden layer of the model
        """
        if model is None:
            model = self.model
        
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