import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple, List, Dict, Callable
from modules.dataset.dataset import Dataset, Data
from modules.utils.utils import *
from modules.plotting_network.plotting_network import NeuralNetworkPlotter
from modules.metrics.metrics import *

def main():
    mnist_dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
    x_train, x_test = normalize_dataset(x_train, x_test)
    # train = Data(x_train, y_train)
    # test = Data(x_test, y_test)

    # dataset = Dataset(train, test)
    # dataset.build_vf_dataset(proportion_cs=0.5, 
    #                         proportion_left=0.5, 
    #                         full_attention_value=1, 
    #                         reduced_attention_value=0.5, 
    #                         ss_attention_value=0.5)

    # dataset = load_obj("./dataset.dat")

    model_dir = '../models/model_t1_denser_50_256/'
    model_path = model_dir + 'model.keras'
    out_models_dir = model_dir + 'out_models/'

    model = tf.keras.models.load_model(model_path)
    #output_models = [tf.keras.models.load_model(out_models_dir + out_model_filename) for out_model_filename in os.listdir(out_models_dir)]

    plotter = NeuralNetworkPlotter(model)
    plotter.plot(x_test[11], x_test[10] * 0.5)

if __name__ == '__main__':
    main()