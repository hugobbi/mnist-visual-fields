import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import List, Tuple, Optional, Set
from attr import define
import modules.utils.utils as ut
from modules.dataset.dataset import Dataset
from collections.abc import Iterable

@define
class Position:
    """
    Dataclass representing a position with two coordinates
    """

    x: float = 0
    y: float = 0

    def copy(self):
        return Position(self.x, self.y)

    def __str__(self):
        return f"Position({self.x}, {self.y})"

@define
class PlottedNeuron:
    """
    Dataclass representing a neuron in the Neural Network plot
    """

    activation: float = 0
    position: Position = Position()
    radius: float = 0
    spacing: float = 0

class PlottedLayer():
    """
    Class representing a layer in the Neural Network plot
    """

    def __init__(self, original_layer: tf.keras.layers.Layer, activations: Optional[List[float]] = None, position:  Optional[Position] = None):
        # Copying necessary attributes from original layer
        self.name = original_layer.name
        self.weights, self.biases = original_layer.get_weights() if original_layer.get_weights() else (None, None)
        self.activations = activations
        self.position = position if position is not None else Position(0, 0)
        self.num_neurons = self.__compute_num_neurons(original_layer.output_shape)
        self.neurons = []
        self.type = original_layer.__class__.__name__
        self.is_input_layer = self.type == "InputLayer"
        self.is_output_layer = not bool(original_layer._outbound_nodes)
        self.is_concatenate_layer = self.type == "Concatenate"
        self.is_conv_layer = self.type == "Conv2D"
        self.previous_layers = self.__get_previous_layers(original_layer)

    def __str__(self):
        return f"PlottedLayer({self.name}, {self.type}, {self.num_neurons}, is_input={self.is_input_layer}, is_output={self.is_output_layer}, is_concat={self.is_concatenate_layer})"
    
    def __compute_num_neurons(self, num_neurons) -> int:
        if (isinstance(num_neurons, Iterable)):
            if len(num_neurons) == 1: num_neurons = num_neurons[0]
            return int(np.prod([el for el in num_neurons if el is not None]))
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
    is_left_vf: bool = False
    is_right_vf: bool = False
    is_center_vf: bool = False
    has_concatenated: bool = False
    vf_top: float = 0
    vf_center: float = 0
    vf_bottom: float = 0
    using_two_columns: bool = False
    SHOULD_PLOT = {"InputLayer", "Dense", "Concatenate", "Conv2D"}

    @property
    def vf_size(self) -> float:
        return self.vf_top - self.vf_bottom

class AccuracyLogging(tf.keras.callbacks.Callback):
    """
    Callback class to log the accuracy of the attribute lenses during training
    """
    
    def __init__(self, plotter: "NeuralNetworkPlotter", attribute_lens_name: str) -> None:
        super(AccuracyLogging, self).__init__()
        self.plotter = plotter
        self.attribute_lens_name = attribute_lens_name
    
    def on_epoch_end(self, epoch, logs=None):
        self.plotter.attribute_lenses_accuracy.setdefault(self.attribute_lens_name, []).append(logs["accuracy"])

class NeuralNetworkPlotter:
    def __init__(
            self, 
            model: tf.keras.Model, 
            max_neurons: int = 300,
            weight_threshold: float = 0.75,
            attribute_lenses: Optional[List[tf.keras.Model]] = None,
            num_attr_lenses_top_activations: int = 3,
            save_plots: bool = True,
            ) -> None:
        
        self.model = model
        self.max_neurons = max_neurons
        self.weight_threshold = weight_threshold
        self.attribute_lenses = attribute_lenses
        self.attribute_lenses_accuracy = {}
        self.num_attr_lenses_top_activations = num_attr_lenses_top_activations
        self.save_plots = save_plots
        self.__controller = None

    @property
    def is_dvf_plot(self) -> bool:
        return ut.is_dvf_model(self.model)
    
    def plot(self,
             *input_data: np.array,
             model: Optional[tf.keras.Model] = None,
             max_neurons: Optional[int] = None,
             weight_threshold: Optional[float] = None,
             attribute_lenses: Optional[List[tf.keras.Model]] = None,
             num_attr_lenses_top_activations: Optional[int] = None,
             save_plots: Optional[bool] = None) -> None:
        
        """
        Receives a trained model and an input, displaying the entire neural network with its
        activations for that input.

        Input: 
        model: tf.keras.Model: model to be used for visualization
        input_data: np.array: input data to be used for visualization
        max_neurons: int: maximum number of neurons to be plotted in a layer (default: 300)
        weight_threshold: float: minimum weight value to be plotted, considering normalized values between 0 and 1 (default: 0.5)
        attribute_lenses: Optional[List[tf.keras.Model]]: list of models to be used as attribute lenses for each layer (default: None)
        num_attr_lenses_top_activations: int: number of top digit activations displayed for each attribute lens (default: 3)
        save_plot: bool: if the neural network plot should be saved as an image (default: True)

        Output:
        displays and saves the image in results/images/ if requested
        """
        
        # Setting default values
        if model is not None:
            self.model = model
        if max_neurons is not None:
            self.max_neurons = max_neurons 
        if weight_threshold is not None:
            self.weight_threshold = weight_threshold
        if attribute_lenses is not None:
            self.attribute_lenses = attribute_lenses
        if num_attr_lenses_top_activations is not None:
            self.num_attr_lenses_top_activations = num_attr_lenses_top_activations
        if save_plots is not None:
            self.save_plots = save_plots

        # Determining matplotlib figure parameters
        fig, ax = plt.subplots(figsize=(32, 32))
        ax.axis("off")
        top, bottom, left, right = self.__compute_figure_sizes(top=0.98)
        middle = round((top + bottom) / 2, ndigits=4)

        # Calculating activations
        if self.is_dvf_plot:
            left_vf_data, right_vf_data = input_data
            model_activations = ut.compute_activations(self.model, left_vf_data, right_vf_data)
        else:
            input_data = input_data[0]
            model_activations = ut.compute_activations(self.model, input_data)

        # Determining neural network figure parameters
        MIDDLE_SPACING = 0.02
        TOTAL_LAYER_SPACES = self.__compute_total_layer_spaces()
        CONECTION_OPACITY = 0.2
        COLOR_CONECTION_OPACITY = 0.5
        IMAGE_OFFSET = 0.024
        TEXT_X_OFFSET, TEXT_Y_OFFSET = 0.005, 0.003
        LAYER_SPACING_NO_WEIGHTS = 0.1
        LEFT_VF_MIDDLE = 1.5 * middle
        RIGHT_VF_MIDDLE = 0.5 * middle
        INITIAL_LEFT_VF_POSITION = Position(left, LEFT_VF_MIDDLE) 
        INITIAL_RIGHT_VF_POSITION = Position(left, RIGHT_VF_MIDDLE)  
        INITIAL_CENTER_VF_POSITION = Position(left, middle)  
        AL_TEXT_X_OFFSET = -0.02
        AL_TEXT_X_OFFSET_COLUMNS = -0.045
        AL_TEXT_Y_OFFSET_COLUMNS = 0.025
        AL_TEXT_Y_OFFSET = 0.0345
        OUTPUT_OFFSET = 0.2

        # Setting up control structure to plot layers
        self.__controller = PlottingControl()
        for i, (layer, activations) in enumerate(zip(self.model.layers, model_activations)):
            plotted_layer = PlottedLayer(original_layer=layer, activations=activations)
            if plotted_layer.type not in self.__controller.SHOULD_PLOT: continue # only plots layers that should be plotted
            # Determining layer characteristics
            self.__controller.is_center_vf = plotted_layer.is_concatenate_layer or self.__controller.has_concatenated or not self.is_dvf_plot
            self.__controller.is_left_vf = not self.__controller.is_center_vf and i % 2 == 0
            self.__controller.is_right_vf = not self.__controller.is_center_vf and i % 2 != 0 
            self.__controller.vf_top = top
            self.__controller.vf_bottom = bottom
            self.__controller.vf_center = middle
            self.__controller.using_two_columns = False
            # Left VF
            if self.__controller.is_left_vf:
                self.__controller.vf_bottom = middle + MIDDLE_SPACING
                self.__controller.vf_center = LEFT_VF_MIDDLE
            # Right VF
            if self.__controller.is_right_vf:
                self.__controller.vf_top = middle - MIDDLE_SPACING
                self.__controller.vf_center = RIGHT_VF_MIDDLE
            # Input
            if plotted_layer.is_input_layer:
                if self.is_dvf_plot:
                    plotted_layer.position = INITIAL_LEFT_VF_POSITION if self.__controller.is_left_vf else INITIAL_RIGHT_VF_POSITION
                else:
                    plotted_layer.position = INITIAL_CENTER_VF_POSITION
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
                self.__controller.vf_top = middle + OUTPUT_OFFSET
                self.__controller.vf_bottom = middle - OUTPUT_OFFSET
                        
            # Getting previously plotted layer
            previous_layer = plotted_layer.get_previously_plotted_layer(self.__controller.plotted)
            should_plot_weights = plotted_layer.num_neurons <= self.max_neurons and previous_layer.num_neurons <= self.max_neurons and not plotted_layer.is_concatenate_layer
            
            # Determining layer spacing and layer position
            layer_spacing = (right - left - TEXT_X_OFFSET) / TOTAL_LAYER_SPACES if should_plot_weights else LAYER_SPACING_NO_WEIGHTS
            plotted_layer.position.x = previous_layer.position.x + layer_spacing

            # Plotting neurons and weights
            if plotted_layer.num_neurons <= self.max_neurons and not plotted_layer.is_conv_layer:
                # Plotting neurons
                plotted_layer.position.y = self.__controller.vf_top
                neuron_spacing = self.__controller.vf_size / plotted_layer.num_neurons
                neuron_position = plotted_layer.position.copy()
                layer_activations = ut.normalize_ndarray(plotted_layer.activations) if not plotted_layer.is_output_layer else plotted_layer.activations
                for j, neuron_activation in enumerate(layer_activations):
                    neuron = PlottedNeuron(neuron_activation, position=neuron_position.copy(), radius=neuron_spacing/4, spacing=neuron_spacing)
                    plotted_layer.neurons.append(neuron)
                    neuron_circle = plt.Circle(
                        xy=(neuron.position.x, neuron.position.y),
                        radius=neuron.radius,
                        color=plt.cm.viridis(neuron_activation),
                        ec="k",
                    )
                    ax.add_artist(neuron_circle)
                    neuron_position.y -= neuron_spacing if j != plotted_layer.num_neurons - 1 else 0
                    # Plotting neuron acitavtion value of output neurons
                    if plotted_layer.is_output_layer:
                        ax.text(neuron.position.x + neuron.radius + TEXT_X_OFFSET, neuron.position.y - TEXT_Y_OFFSET, f"{j}: {neuron_activation:.4f}", fontsize=12, fontweight="bold")
                    # Plotting weights
                    if should_plot_weights:
                        layer_weights_neuron = ut.normalize_ndarray(plotted_layer.weights[:, j])
                        for previous_neuron, connection_weight in zip(previous_layer.neurons, layer_weights_neuron):
                            if connection_weight < self.weight_threshold: continue
                            connection = plt.Line2D(
                                [neuron.position.x - neuron.radius, 
                                 previous_neuron.position.x + previous_neuron.radius],
                                [neuron.position.y, previous_neuron.position.y],
                                color=plt.cm.viridis(connection_weight),
                                alpha=COLOR_CONECTION_OPACITY
                            )
                            ax.add_artist(connection)
                    # Plotting symbolic connections from current layer to previous layer
                    # elif not plotted_layer.is_concatenate_layer:
                        # connection = plt.Line2D(
                        #     [neuron.position.x - neuron.radius, 
                        #         previous_layer.position.x + IMAGE_OFFSET],
                        #     [neuron.position.y, previous_layer.position.y],
                        #     color='black',
                        #     alpha=CONECTION_OPACITY
                        # )
                        # ax.add_artist(connection)
                    # Plotting symbolic concatenate connecitons
                    elif plotted_layer.is_concatenate_layer:
                        middle_spacing = 0 if j < plotted_layer.num_neurons // 2 else 2 * MIDDLE_SPACING
                        previous_radius = previous_layer.neurons[0].radius
                        previous_spacing = previous_layer.neurons[0].spacing
                        connection = plt.Line2D(
                            [neuron.position.x - neuron.radius, 
                                previous_layer.position.x + previous_radius],
                            [neuron.position.y, previous_layer.position.y - (j * (previous_spacing) + middle_spacing)],
                            color='black',
                            alpha=CONECTION_OPACITY
                        )
                        ax.add_artist(connection)
            else:
                # Plotting convolution layer 
                if plotted_layer.is_conv_layer:
                    # Determine size of feature maps
                    num_feature_maps = plotted_layer.activations.shape[2]
                    fm_size = self.__calculate_image_size(plotted_layer.num_neurons / num_feature_maps)
                    fm_size_plot = fm_size * 0.01
                    fm_spacing = fm_size_plot * 2
                    initial_position = self.__controller.vf_center + (num_feature_maps - 1) * (fm_size_plot)
                    plotted_layer.position.y  = initial_position    
                    
                    # If feature maps exceed vertical space, reduce size and start from top
                    if fm_size_plot * num_feature_maps + (num_feature_maps - 1) * fm_spacing > self.__controller.vf_size:
                        self.__controller.using_two_columns = True
                        plotted_layer.position.x -= fm_size_plot / 2
                        spacing_factor = 2
                        fm_size_plot = self.__controller.vf_size / ((num_feature_maps * 0.5) * (1 + spacing_factor) - spacing_factor) * (1 + num_feature_maps * 0.01)
                        fm_spacing = fm_size_plot * spacing_factor
                        fm_size = fm_size_plot * 100
                        initial_position = self.__controller.vf_center + (num_feature_maps - 1) * (fm_size_plot / 2)
                        plotted_layer.position.y = initial_position
                        AL_TEXT_X_OFFSET_COLUMNS = 0.0011 * num_feature_maps - 0.0663

                    # For each feature map in convolution layer
                    for f in range(num_feature_maps):
                        if f == num_feature_maps // 2 and self.__controller.using_two_columns:
                            plotted_layer.position.x += spacing_factor * fm_size_plot
                            plotted_layer.position.y = initial_position
                        feature_map = plotted_layer.activations[:, :, f]
                        ab = self.__generate_image_annotation_box(
                                feature_map,
                                plotted_layer.position,
                                cmap="binary",
                                size=fm_size
                        )
                        ax.add_artist(ab)
                        plotted_layer.position.y -= fm_spacing if f != num_feature_maps - 1 else 0
                # Plotting layer as image
                else:
                    activations_image = self.__get_image(plotted_layer.activations)
                    ab = self.__generate_image_annotation_box(
                                activations_image,
                                plotted_layer.position,
                                cmap="binary",
                                size=self.__calculate_image_size(plotted_layer.num_neurons),
                            )
                    ax.add_artist(ab)
                    # Plotting symbolic connections from image to previous layer
                    # if previous_layer.num_neurons <= self.max_neurons:
                    #     for previous_neuron in previous_layer.neurons:
                    #         connection = plt.Line2D(
                    #             [plotted_layer.position.x - IMAGE_OFFSET, 
                    #                 previous_neuron.position.x + previous_neuron.radius],
                    #             [plotted_layer.position.y, previous_neuron.position.y],
                    #             color='black',
                    #             alpha=CONECTION_OPACITY
                    #         )
                    #         ax.add_artist(connection)
                    # else:
                    #     # Plotting two symbolic connections on each side of the image
                    #     for l in range(2):
                    #         connection = plt.Line2D(
                    #             [plotted_layer.position.x - IMAGE_OFFSET, 
                    #                 previous_layer.position.x + IMAGE_OFFSET],
                    #             [plotted_layer.position.y + IMAGE_OFFSET*(1-2*l), previous_layer.position.y],
                    #             color='black',
                    #             alpha=CONECTION_OPACITY
                    #         )
                    #         ax.add_artist(connection)
            
            # Plotting attribute lenses
            if self.attribute_lenses is not None:
                al_idx = len(self.__controller.plotted) - (len(self.model.input) if self.is_dvf_plot else 1) # attribute lens index will be all plotted layers except input layers
                if al_idx < len(self.attribute_lenses):
                    attribute_lens = self.attribute_lenses[al_idx]
                    if self.is_dvf_plot:
                        if ut.is_dvf_model(attribute_lens):
                            digit_activations = ut.compute_digits_model_predicts(attribute_lens, left_vf_data, right_vf_data)
                        else:
                            if self.__controller.is_left_vf:
                                digit_activations = ut.compute_digits_model_predicts(attribute_lens, left_vf_data)
                            else:
                                digit_activations = ut.compute_digits_model_predicts(attribute_lens, right_vf_data)
                    else:
                        digit_activations = ut.compute_digits_model_predicts(attribute_lens, input_data)
                    
                    al_text_string = ''
                    for k in range(self.num_attr_lenses_top_activations):
                        eol = '\n' if k != self.num_attr_lenses_top_activations - 1 else ''
                        al_text_string += f'{digit_activations[k][0]}: {digit_activations[k][1]:.4f}{eol}'
                    if plotted_layer.num_neurons <= self.max_neurons:
                        al_text_position = plotted_layer.neurons[-1].position.copy()
                    else:
                        al_text_position = Position(plotted_layer.position.x, plotted_layer.position.y - IMAGE_OFFSET)
                    text = plt.Text(
                        al_text_position.x + (AL_TEXT_X_OFFSET if not self.__controller.using_two_columns else AL_TEXT_X_OFFSET_COLUMNS),
                        al_text_position.y - (AL_TEXT_Y_OFFSET if not self.__controller.using_two_columns else AL_TEXT_Y_OFFSET_COLUMNS),
                        al_text_string,
                        fontsize=12,
                        fontweight='bold',
                        color="k",
                    )
                    ax.add_artist(text)
            
            # Adding current layer to plotted layers list
            self.__controller.plotted.append(plotted_layer)
        
        # Saving plot
        if self.save_plots:
            save_dir = "results/images/"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/nn_plot_{ut.get_current_time_string()}.png")  

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
    
    def __get_image(self, array: np.array):
        """
        Transforms 1D array into 2D array to be plotted as an image
        """
        if len(array.shape) != 1: return array
        num_neurons = len(array)
        num_rows = int(np.sqrt(num_neurons))
        num_cols = int(np.ceil(num_neurons / num_rows))

        return np.array(array).reshape(num_rows, num_cols)

    def __get_digit_from_y_spacing(self, total_position_plotted: float, y_spacing: float) -> int:
        """
        Returns the digit an output neuron represents based on how many neurons have been plotted
        """
        return np.ceil(total_position_plotted / y_spacing)

    def __calculate_image_size(self, number_neurons: int) -> float:
        """
        Computes size of image based on the number of neurons
        """
        return np.log10(number_neurons) # * 1.25
        # Determined in tests
        a = -1/432
        b = 130/27
        return number_neurons * a + b
    
    def __compute_total_layer_spaces(self) -> int:
        """
        Computes the total number of layer spaces a neural network plot will have
        """
        i, j = 0, 0
        has_concatenated = False
        for layer in self.model.layers:
            if layer.__class__.__name__ not in PlottingControl.SHOULD_PLOT: continue    
            i += 1
            if not has_concatenated and self.is_dvf_plot:
                j += 1
                if layer.__class__.__name__ == "Concatenate": has_concatenated = True

        return int(i - j // 2) - 1
    
    def generate_attribute_lenses(self, model: Optional[tf.keras.Model] = None) -> None:
        """
        Generates attribute lenses for each hidden layer of the model
        """
        if model is not None:
            self.model = model
        NOT_ATTRIBUTE_LENSES = {"InputLayer", "Flatten"}
        has_concatenated = False

        # Copying original model and freezing weigths
        model_copy = tf.keras.models.clone_model(self.model)
        model_weights = self.model.get_weights()
        model_copy.set_weights(model_weights)
        for layer in model_copy.layers: layer.trainable = False

        # Generating attribute lenses for each hidden layer
        self.attribute_lenses = []
        for i, layer in enumerate(model_copy.layers[:-2]): # excluding output layer and last hidden layer
            # Layer type controller
            is_conv_layer = False
            if layer.__class__.__name__ in NOT_ATTRIBUTE_LENSES: continue
            if layer.__class__.__name__ == "Conv2D": is_conv_layer = True
            if layer.__class__.__name__ == "Concatenate": has_concatenated = True
            
            # Determining input layer
            if has_concatenated or not self.is_dvf_plot:
                input_layer = model_copy.input
            else:
                input_layer = model_copy.input[0] if i % 2 == 0 else model_copy.input[1] # left or right input
            
            # Determining output layer
            if is_conv_layer:
                flatten = tf.keras.layers.Flatten()(layer.output)
                output = tf.keras.layers.Dense(10, activation="softmax")(flatten)
            else:
                output = tf.keras.layers.Dense(10, activation="softmax")(layer.output)
            
            # Creating model
            attribute_lens = tf.keras.models.Model(
                inputs=input_layer, outputs=output
            )
            self.attribute_lenses.append(attribute_lens)

    def __get_attribute_lens(self, plotted_layer: PlottedLayer) -> tf.keras.Model:
        """
        Matches second to last layer of attribute lense to plotted layer name in order
        to find its corresponing attribute lens model
        """
        for al in self.attribute_lenses:
            if al.layers[-2].name == plotted_layer.name:
                return al
        raise ValueError(f"Attribute lens for layer '{plotted_layer.name}' not found")

    def train_attribute_lenses(
            self, 
            dataset: Dataset, 
            epochs: int, 
            batch_size: int,
            loss: str = "sparse_categorical_crossentropy",
            optimizer: str = "adam",
            metrics: List[str] = ["accuracy"]) -> None:
        """
        Trains attribute lenses for each hidden layer of the model
        """
        if self.attribute_lenses is None:
            raise ValueError("Attribute lenses have not been generated yet")
        for al in self.attribute_lenses:
            if ut.is_dvf_model(al):
                n_train = len(dataset.train_vf.y)
                n_test = len(dataset.test_vf.y)
                training_generator = ut.data_generator_dvf(
                    dataset.train_vf.x_left, dataset.train_vf.x_right, dataset.train_vf.y, batch_size
                )
                testing_generator = ut.data_generator_dvf(
                    dataset.test_vf.x_left, dataset.test_vf.x_right, dataset.test_vf.y, batch_size
                )
            else:
                n_train = len(dataset.train.y)
                n_test = len(dataset.test.y)
                training_generator = ut.data_generator_svf(
                    dataset.train.x, dataset.train.y, batch_size
                )
                testing_generator = ut.data_generator_svf(
                    dataset.test.x, dataset.test.y, batch_size
                )
            al.compile(
                loss=loss,
                optimizer=optimizer, 
                metrics=metrics,
            )
            al.fit(
                training_generator,
                steps_per_epoch=n_train // batch_size,
                epochs=epochs,
                callbacks=[AccuracyLogging(self, al.name)],
                validation_data=testing_generator,
                validation_steps=n_test // batch_size
            )