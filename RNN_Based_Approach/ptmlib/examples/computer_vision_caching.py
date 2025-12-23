# THIS CODE IS A MODULARIZED AND UPDATED VERSION OF CODE FROM THE "DEEPLEARNING.AI TENSORFLOW DEVELOPER" COURSE
# SOURCE:
# https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import ptmlib.model_tools as modt
import ptmlib.charts as pch

class MyCallback(keras.callbacks.Callback):

    def __init__(self, target):
        super().__init__()
        self.target = target

    def on_epoch_end(self, _, logs=None):

        if logs is None:
            logs = {}
        if logs.get("accuracy") > self.target:
            print(f"\nReached {self.target * 100}% accuracy so cancelling training!")
            self.model.stop_training = True


def print_diagnostics() -> None:
    print('TF VERSION:', tf.__version__)
    print('KERAS VERSION:', keras.__version__)


def get_data():
    mnist = keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    np.set_printoptions(linewidth=200)
    plt.imshow(training_images[0])
    plt.show()

    print(training_labels[0])

    # normalize image data to values between 0 and 1
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    return (training_images, training_labels), (test_images, test_labels)


def get_model() -> keras.models.Sequential:
    model = keras.models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dropout(0.2),
        layers.Dense(512, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.summary()

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    # HYPER PARAMS, CONSTANTS, ETC
    hp_epochs = 50
    hp_target = 0.91
    hp_validation_split = 0.2
    model_file_name = "computer_vision_1"

    print_diagnostics()

    (training_images, training_labels), (test_images, test_labels) = get_data()

    model = get_model()

    early_callback = MyCallback(target=hp_target)

    fit_model_function_with_callback = lambda my_model, x, y, validation_data, epochs: my_model.fit(
        x, y, validation_data, epochs=epochs, callbacks=[early_callback], validation_split=hp_validation_split)

    # testing only; a real implementation would specify custom_objects (see README for example)
    load_model_function_custom = lambda file_name, file_format: keras.models.load_model(
        modt.get_file_path(file_name, file_format), custom_objects=None)

    model, history = modt.load_or_fit_model(model, model_file_name, x=training_images, y=training_labels,
                                            epochs=hp_epochs, model_file_format="tf_saved_model",
                                            metrics=["accuracy"],
                                            load_model_function=load_model_function_custom,
                                            fit_model_function=fit_model_function_with_callback)

    model.evaluate(test_images, test_labels)

    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])
    print(max(classifications[0]))

    # ensure history data is still available, even if cached
    pch.show_history_chart(history, "accuracy")  # render again to be sure we have proper history data
    print('type(model):', type(model))
    print('type(history):', type(history))
    print('history.history:', history.history)
    print('history.params:', history.params)


if __name__ == '__main__':
    main()
