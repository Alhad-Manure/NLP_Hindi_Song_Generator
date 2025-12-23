import os
import pickle
from typing import Any, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow import keras

import ptmlib.charts as pch
from ptmlib.time import Stopwatch

HISTORY_FILE_SUFFIX_EXTENSION = '_history.pkl'


def get_file_path(model_file_name: str, model_file_format: str = ""):
    extension = _get_model_file_extension(model_file_format)
    return f'{model_file_name}{extension}'


def _get_model_file_extension(model_file_format: str):
    if model_file_format == "tf_saved_model":
        # no extension means we are using TensorFlow SavedModel format
        extension = ""
    else:
        extension = ".h5"
    return extension


def _default_load_model_function(model_file_name: str, model_file_format: str = ""):
    return keras.models.load_model(get_file_path(model_file_name, model_file_format))


def _default_fit_model_function(model: Any, x: Any, y: Any = None, validation_data: Any = None, epochs: int = 1):
    return model.fit(x, y, validation_data=validation_data, epochs=epochs)


def load_or_fit_model(model: Any, model_file_name: str, x: Any, y: Any = None, validation_data: Any = None,
                      epochs: int = 1, metrics: List[str] = None, images_enabled=True, fig_size: (int, int) = (10, 6),
                      model_file_format: str = "",
                      load_model_function=_default_load_model_function,
                      fit_model_function=_default_fit_model_function):
    file_extension = _get_model_file_extension(model_file_format)

    if os.path.exists(f'{model_file_name}{file_extension}'):
        print(f'Loading existing model file: {model_file_name}{file_extension}')
        model = load_model_function(model_file_name, model_file_format)
        history = load_history_data(model_file_name)
        if images_enabled:
            _show_saved_images(metrics, model_file_name, fig_size)
    else:
        stopwatch = Stopwatch()
        stopwatch.start()
        history = fit_model_function(model, x, y, validation_data, epochs)
        stopwatch.stop()
        print(f'Saving new model file: {model_file_name}{file_extension}')
        model.save(f'{model_file_name}{file_extension}')
        save_history_data(history, model_file_name)
        if images_enabled:
            _show_new_images(history, model_file_name, metrics)

    return model, history


def save_history_data(history: Any, model_file_name: str):
    with open(f'{model_file_name}{HISTORY_FILE_SUFFIX_EXTENSION}', 'wb') as history_file:
        history_params_tuple = (history.history, history.params)
        pickle.dump(history_params_tuple, history_file)


def load_history_data(model_file_name: str):
    if not os.path.exists(f'{model_file_name}{HISTORY_FILE_SUFFIX_EXTENSION}'):
        return None

    # create new history object for return value
    history = keras.callbacks.History()

    with open(f'{model_file_name}{HISTORY_FILE_SUFFIX_EXTENSION}', 'rb') as history_file:
        # load from previously saved history_params_tuple
        h, p = pickle.load(history_file)
        history.history = h
        history.params = p

    return history


def _show_new_images(history: Any, model_file_name: str, metrics: List[str]):
    if metrics is not None:
        for metric in metrics:
            pch.show_history_chart(history, metric, save_fig_enabled=True, file_name_suffix=model_file_name)
    pch.show_history_chart(history, "loss", save_fig_enabled=True, file_name_suffix=model_file_name)


def _show_saved_images(metrics: List[str], model_file_name: str, fig_size: (int, int) = (10, 6)):
    if metrics is not None:
        for metric in metrics:
            if os.path.exists(f'{metric}-{model_file_name}.png'):
                _show_saved_image(f'{metric}-{model_file_name}.png', fig_size)
    if os.path.exists(f'loss-{model_file_name}.png'):
        _show_saved_image(f'loss-{model_file_name}.png', fig_size)


def _show_saved_image(filename: str, fig_size: (int, int) = (10, 6)):
    image_data = mpimg.imread(filename)
    fig = plt.figure(figsize=fig_size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    plt.axis('off')
    plt.imshow(image_data)
    plt.show()
