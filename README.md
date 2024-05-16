# mnist-visual-fields

Code to handle creation and visualization of neural networks using Keras. It was made to work with the MNIST dataset, and the neural networks can have one or two visual fields (inputs). When using one visual field, the network recognizes the digit present in that visual field; when using two visual fields, the network recognizes the digit with higher associated attention value.

## Dependencies
* tensorflow v2.15.0.post1
* keras v2.15.0
* matplotlib v3.8.3
* numpy v1.26.4
* attrs v23.2.0

## Importing MNIST dataset

It is possible to import the MNIST dataset using `keras.datasets.mnist`.

## Creating model or loading it from file and importing MNIST dataset

One use of this program is creating a new model, generating its training and testing data (requires importing MNIST dataset), training it and finally using it for visualization. Another possible use of this program is loading a model from a file, importing the MNIST dataset and then using it for visualization.

In either case, it is necessary to import the MNIST dataset.

## Creating a new model

### Generating training and testing data

It is first necessary to import the MNIST dataset.

#### Single visual field

Using the function `build_visual_field_data`.

#### Double visual field

Using the function `build_double_visual_fields_dataset`.

### Training

Having generated the data and compiled the model, training is done using the `fit` method from `keras.Model`. For the purpose of saving RAM, it is recommended to use the `data_generator` functions (specifying single or double visual field network functions).

#### Saving model to file

It is possible to save the trained model to the `models/` path, using the `keras.Model.save` method.

## Loading model from file

The models are stored in the `models/` directory, having a separate directory for each model and its attribute lenses. It is possible to load it, not having to train a new one, using the `keras.models.load_model` method.

## Visualizing neural network

After importing the MNIST dataset and creating or loading a model, the program is ready to plot the model's execution on some input data.

### Choosing input for visualization

Having chosen one or two digits to use as input data, it is necessary to find their indices in the MNIST dataset (or the dataset containing the digits that will be used). This is done using the `display_n_digits` function. When given a dataset, the labels of that dataset, a digit and an integer `n`, the function displays `n` instances of that digit and their index in the given dataset. This way, it is possible to choose exactly which inputs to use for the neural network, given that the index of these digits is used to reference them in the plotting function.

### Plotting neural network

To plot the neural network, it is used the function `display_double_visual_field_mnist_nn_execution` or `display_single_visual_field_mnist_nn_execution`.  
(observation: the single visual field function is currently outdated)

There are 4 main obligatory arguments:
* model: `keras.Model`: the model (trained or loaded from a file) to be plotted
* data: `np.array`: the dataset from which the input digits for the plot are stored
* left_vf_digit: `Tuple[int, float]`: the index in the dataset for the digit to be used in the left visual field, as well as its attention value
* right_vf_digit: `Tuple[int, float]`: the index in the dataset for the digit to be used in the right visual field, as well as its attention value

Other optional but important arguments:
* output_models: models to be used as attribute lenses for each layer
* k: number of top digit activations displayed for each attribute lens

There are additional fine-tuning arguments, which are explained inside the plotting function.

When the function is executed, an image of the plot will be stored in `images/`.

### Attribute lenses

Attribute lenses are used to inspect which digits are represented inside each hidden layer of a model. Just like the models, they can be created or loaded from a file. Also, when created, they can be saved inside a directory in their model's directory. A list of attrtibute lenses for a model can be passed to the plotting neural network function so that, for each hidden layer plotted, the top `k` digits (on activation value) are shown, as well as their corresponding activation value.

### Custom forward pass function

For every forward pass computed in this program, the `compute_forward_pass` function is used. It can be customized to work in different ways for each time the forward pass is computed.