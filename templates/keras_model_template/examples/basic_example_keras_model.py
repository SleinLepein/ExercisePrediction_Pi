"""
This file implements a very basic (and useless) Keras Neural Net Model to illustrate how to work with
the Keras Model Template.
"""

import os
from dotdict import DotDict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization

from templates.keras_model_template.keras_model_class import KerasModel
from templates.keras_model_template.data_generators.standard_data_generator import StandardDataGenerator

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


class BasicKerasModel(KerasModel):
    # standard parameters for this class (name must be different from any such standard parameters in
    # the heritage chain, i.e., at this stage it
    # must not be "standard_params" (used in Parameter Class),
    # must not be "standard_model_params" (used in Model Template) and
    # must not be "standard_params_keras" (used in Keras Model Template)

    standard_params_basic = DotDict(
        # input definition
        input_dim=10,  # number of channels

        # output definition
        output_dim=1,  # count of labels
        output_activation="sigmoid",  # sigmoid output activation for classification

        # loss
        loss="binary_crossentropy",  # binary crossentropy loss for classification
    )

    def __init__(self, **params_dict):
        """
        A basic Keras model as example.

        Parameters
        ----------
        params_dict
        """

        super().__init__(**params_dict)

        # update internal params based on given input params and given standard params
        self.update_params(self.input_params, self.standard_params_basic)

        # build model according to specification in params, build method is implemented below
        self._build()

        # compile model according to specification in params, compile method is already implemented in keras model
        self.compile()

    def _build(self):
        """
        Implements build method from abstract class to actually define a Keras model for member
        training_model.

        Returns
        -------
        None
        """

        # ----------------- input definition -----------------

        inputs = Input(shape=(self.params.input_dim,), name=f"{self.params.name}_input")
        x = inputs

        for _ in range(3):
            x = Dense(10)(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        # ----------------- output definition -----------------
        x = Dense(self.params.output_dim)(x)
        outputs = Activation(self.params.output_activation)(x)

        # ----------------- model definition -----------------
        # member training_model will be used in training method
        self.training_model = Model(inputs, outputs, name="test_model")

        # define loss for training model
        loss = self.get_loss(self.params.loss, reduction="auto")
        self.append_loss(loss, loss_weight=1, loss_name="loss", add_metric=False)

        if self.params.loss == "binary_crossentropy":
            rec_val = tf.keras.metrics.BinaryAccuracy()
        elif self.params.loss == "categorical_crossentropy":
            rec_val = tf.keras.metrics.CategoricalAccuracy()
        else:
            rec_val = loss

        self.append_metric(rec_val, "val_metric")

        if self.params.model_summary:
            self.training_model.summary()


def main(input_dim=10,
         output_dim=1,
         training_data_count=10000,
         validation_data_count=1000,
         epochs=100,
         use_data_generator=True
         ):
    # ------------------------------------ OPTIONS ----------------------------------
    # folder for outputs
    # use sub folder in current working directory
    save_folder = os.path.join(os.getcwd(), "basic_model_files")
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    print(save_folder)

    print("----------- GENERATING SAMPLE TRAINING DATA -----------")
    # ------------- TRAINING DATA ------------
    # input training data
    input_train = np.random.randn(training_data_count, input_dim)

    # output training data
    output_train_integers = np.random.randint(0, output_dim, training_data_count)
    output_train = [np.zeros(output_dim) for _ in range(training_data_count)]
    for i in range(training_data_count):
        output_train[i][output_train_integers[i]] = 1
    output_train = np.array(output_train)

    # ------------- VALIDATION DATA ------------
    # input training data
    input_test = np.random.randn(validation_data_count, input_dim)

    # output training data
    output_test_integers = np.random.randint(0, output_dim, validation_data_count)
    output_test = [np.zeros(output_dim) for _ in range(validation_data_count)]
    for i in range(validation_data_count):
        output_test[i][output_test_integers[i]] = 1
    output_test = np.array(output_test)

    # ------------- MODEL GENERATION ------------
    # build a sample model
    print("----------- BUILDING -----------")
    params = DotDict(save_folder=save_folder,
                     name="test_model",

                     # input definition
                     input_dim=input_dim,

                     # output definition
                     output_dim=output_dim,
                     output_activation="sigmoid",

                     # loss
                     loss="binary_crossentropy",

                     # callbacks
                     checkpoint_monitor="val_binary_accuracy",
                     save_latest=True,

                     early_stopping_monitor="val_binary_accuracy",  # whether to use early stopping
                     early_stopping_patience=5,  # early stopping patience in epochs

                     lr_info=True,
                     tensorboard=True,
                     )

    keras_model = BasicKerasModel(**params)

    # ------------- DATA GENERATOR ------------
    if use_data_generator:
        train_input_data = StandardDataGenerator(inputs=input_train,
                                                 outputs=output_train,
                                                 batch_size=keras_model.params.batch_size)
        train_output_data = None

        test_input_data = StandardDataGenerator(inputs=input_test,
                                                outputs=output_test,
                                                batch_size=keras_model.params.batch_size)
        test_output_data = None
    else:
        train_input_data = input_train
        train_output_data = output_train
        test_input_data = input_test
        test_output_data = output_test

    # train model
    print("----------- TRAINING -----------")
    keras_model.train(train_input_data=train_input_data,
                      train_output_data=train_output_data,
                      test_input_data=test_input_data,
                      test_output_data=test_output_data,
                      epochs=epochs
                      )

    # load model
    print("----------- LOADING -----------")
    # initialize new model
    loaded_keras_model = BasicKerasModel(**params)
    loaded_keras_model.load(folder_path=save_folder, model_name="model_latest.ckpt")

    test = loaded_keras_model.training_model.predict(input_test[0:1])
    print(test)


if __name__ == '__main__':
    main()
