"""
This file implements a basic convolutional classifier by subclassing the Keras Model Template. The example builds
up a 2D convolutional classifier model for data inputs of shape (28, 28, 1) and 10 labels, similar to the case
of MNIST.
"""

import os
from pprint import pprint

from dotdict import DotDict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Activation, Conv2D, Conv1D, BatchNormalization

from templates.keras_model_template.keras_model_class import KerasModel

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


class SampleConvModel(KerasModel):

    standard_sample_cnn_params = DotDict(
        # input definition
        vertical_size=28,  # vertical size of image
        horizontal_size=28,  # horizontal size of image
        depth=1,  # number of channels

        # output definition
        output_dim=10,  # count of labels
        output_activation="softmax",  # softmax output activation for classification

        # hidden layers
        conv_dim=2,  # 2D convolutions
        conv_filters=64,  # count of filters per convolution layer
        conv_kernel_size=(3, 3),  # kernel size for convolutions
        conv_blocks_count=3,  # count of convolutional blocks
        conv_activation="relu",  # activation function after convolution blocks
        dense_blocks_count=2,  # count of dense blocks
        dense_units=16,  # count of units per dense layer
        dense_activation="relu",  # activation function after dense blocks

        # loss
        loss="categorical_crossentropy",  # categorical crossentropy loss for classification
    )

    def __init__(self, **params_dict):

        """
        A simple Keras model as example.

        Parameters
        ----------
        params_dict
        """

        super().__init__(**params_dict)

        # update parameters
        self.update_params(self.input_params, self.standard_sample_cnn_params)

        # build model according to specification in params, build method is implemented below
        self._build()

        # compile model according to specification in params, compile method is already implemented in keras model
        self.compile()

    def _build(self) -> None:

        """
        Implements build method from abstract class to actually define a Keras model for member
        training_model.

        Returns
        -------
        None
        """

        # ----------------- input definition -----------------

        if self.params.conv_dim == 2:
            input_shape = (self.params.vertical_size, self.params.horizontal_size, self.params.depth)
        elif self.params.conv_dim == 1:
            input_shape = (self.params.vertical_size, self.params.horizontal_size)
        elif self.params.conv_dim == 0:
            input_shape = (self.params.vertical_size,)
        else:
            raise NotImplementedError(f"Choice {self.params.conv_dim} for conv_dim not implemented.")

        inputs = Input(shape=input_shape, name=f"{self.params.name}_input")

        # ----------------- output definition -----------------

        x = inputs

        # conv blocks
        if self.params.conv_dim in [1, 2]:
            for _ in range(self.params.conv_blocks_count):
                if self.params.conv_dim == 2:
                    x = Conv2D(filters=self.params.conv_filters,
                               kernel_size=self.params.conv_kernel_size)(x)
                elif self.params.conv_dim == 1:
                    x = Conv1D(filters=self.params.conv_filters,
                               kernel_size=self.params.conv_kernel_size[0])(x)

                x = BatchNormalization()(x)
                x = Activation(self.params.conv_activation)(x)

            x = Flatten()(x)

        # dense blocks
        for _ in range(self.params.dense_blocks_count):
            x = Dense(self.params.dense_units)(x)
            x = BatchNormalization()(x)
            x = Activation(self.params.dense_activation)(x)

        x = Dense(self.params.output_dim)(x)
        outputs = Activation(self.params.output_activation)(x)

        # ----------------- model definition -----------------
        # member training_model will be used in training method
        self.training_model = Model(inputs, outputs, name="sample_cnn_model")

        # ----------------- model summary -----------------

        if self.params.model_summary:
            self.training_model.summary()

        # ----------------- loss definition -----------------
        # append loss function so that it can be used in compile and training method of keras model class
        loss = self.get_loss(self.params.loss, reduction="auto")
        self.append_loss(loss, loss_weight=1, loss_name="classification_loss", add_metric=False)


def main(
        # input definition
        vertical_size=28,  # vertical size of image
        horizontal_size=28,  # horizontal size of image
        depth=1,  # number of channels
        # output definition
        output_dim=10,
        # count of data examples
        training_data_count=10000,
        validation_data_count=1000,
        # training params
        epochs=2,
):
    # ------------------------------------ OPTIONS ----------------------------------
    # folder for outputs
    # use sub folder in current working directory
    save_folder = os.path.join(os.getcwd(), "test_model_files")
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    print(save_folder)

    # ------------------------------------ GENERATION OF SYNTHETIC DATA ----------------------------------
    # define pseudo training data for the model

    # ------------- TRAINING DATA ------------
    # input training data
    input_train = np.random.randn(training_data_count, vertical_size, horizontal_size, depth)

    # output training data
    output_train_integers = np.random.randint(0, output_dim, training_data_count)
    output_train = [np.zeros(output_dim) for _ in range(training_data_count)]
    for i in range(training_data_count):
        output_train[i][output_train_integers[i]] = 1
    output_train = np.array(output_train)

    # ------------- VALIDATION DATA ------------
    # input training data
    input_test = np.random.randn(validation_data_count, vertical_size, horizontal_size, depth)

    # output training data
    output_test_integers = np.random.randint(0, output_dim, validation_data_count)
    output_test = [np.zeros(output_dim) for _ in range(validation_data_count)]
    for i in range(validation_data_count):
        output_test[i][output_test_integers[i]] = 1
    output_test = np.array(output_test)

    print("----------- DATA -----------")
    print("training input shape", input_train.shape)
    print("training output shape", output_train.shape)
    print("validation input shape", input_test.shape)
    print("validation output shape", output_test.shape)
    print("\n")

    # ------------------------------------ MODEL GENERATION ----------------------------------

    # build a sample CNN based classifier
    params = DotDict(save_folder=save_folder,
                     name="test_model",

                     # input definition
                     vertical_size=vertical_size,
                     horizontal_size=horizontal_size,
                     depth=depth,

                     # output definition
                     output_dim=output_dim,

                     checkpoint_monitor="val_loss")
    keras_model = SampleConvModel(**params)

    # print model params
    print("----------- MODEL PARAMS -----------")
    pprint(keras_model.params)
    print("\n")

    # loss functions
    print("----------- LOSSES -----------")
    print(keras_model.losses)
    print("\n")

    # model does not have any callbacks after compilation
    print("----------- NO CALLBACKS AFTER COMPILATION -----------")
    print(keras_model.callbacks)
    print("\n")

    # train model
    print("----------- TRAINING -----------")
    keras_model.train(train_input_data=input_train,
                      train_output_data=output_train,
                      test_input_data=input_test,
                      test_output_data=output_test,
                      epochs=epochs
                      )

    # during training, callbacks were created
    print("----------- CALLBACKS CREATED DURING TRAINING -----------")
    print(keras_model.callbacks)
    print("\n")

    # print training results
    print("----------- TRAINING RESULTS -----------")
    pprint(keras_model.last_loss_history)
    print("\n")

    # how to use the model for prediction (.training_model attribute is corresponding keras model)
    print("----------- EXAMPLE PREDICTION -----------")
    example = np.random.randn(1, vertical_size, horizontal_size, depth)
    example_output = keras_model.training_model.predict(example)
    print(example_output.shape)
    print(example_output[0])
    print("\n")

    # how to save models (see parameter 'save_weights_only')
    keras_model.save(save_folder, "model_with_save_name_bla")

    # how to load model (needs ckpt file if 'save_weights_only' == True)
    keras_model.load(save_folder, "model_with_save_name_bla.ckpt")


if __name__ == '__main__':
    main()
