"""
This file implements a very basic (and useless) Keras Neural Net Model to illustrate how to work with
the Keras Model Templates, particularly WindowDataGenerator for sequential problems.
"""

import os
from dotdict import DotDict

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Activation

from templates.keras_model_template.keras_model_class import KerasModel
from templates.keras_model_template.nn.ForwardNet import ForwardNet
from templates.keras_model_template.data_generators.window_data_generator import WindowDataGenerator

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

feature_extractor = DotDict(block_params=(DotDict(layer_params={"filters": 64,
                                                                "kernel_regularizer": {"type": "L2", "c": 0.0001}},
                                                  activation="relu",
                                                  layer_type="Conv1D",
                                                  normalization="batch_normalization",
                                                  ),
                                          DotDict(layer_params={"filters": 64,
                                                                "kernel_regularizer": {"type": "L2", "c": 0.0001}},
                                                  activation="relu",
                                                  layer_type="Conv1D",
                                                  normalization="batch_normalization",
                                                  skip=True
                                                  ),
                                          DotDict(layer_params={"filters": 64,
                                                                "kernel_regularizer": {"type": "L2", "c": 0.0001}},
                                                  activation="relu",
                                                  layer_type="Conv1D",
                                                  normalization="batch_normalization",
                                                  skip=True
                                                  ),
                                          DotDict(layer_params={"units": 32,
                                                                "kernel_regularizer": {"type": "L2", "c": 0.0001}},
                                                  activation="relu",
                                                  layer_type="Dense",
                                                  )
                                          )
                            )


class BasicKerasModel(KerasModel):
    # standard parameters for this class (name must be different from any such standard parameters in
    # the heritage chain, i.e., at this stage it
    # must not be "standard_params" (used in Parameter Class),
    # must not be "standard_model_params" (used in Model Template) and
    # must not be "standard_params_keras" (used in Keras Model Template)

    standard_params_basic = DotDict(
        # input definition
        input_shape=(10, 2),  # main data input shape

        # output definition
        output_dim=1,  # count of labels
        output_activation="sigmoid",  # sigmoid output activation for classification

        # architecture definition
        feature_extractor=feature_extractor,

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

        inputs = [Input(shape=self.params.input_shape, name=f"{self.params.name}_input")]
        x = inputs[0]

        # ----------------- model definition -----------------

        forward_net = ForwardNet(**self.params.feature_extractor)
        x = forward_net.forward(inputs=x)

        # x = Flatten()(x)
        x = Dense(self.params.output_dim)(x)
        outputs = Activation(self.params.output_activation)(x)

        # ----------------- model definition -----------------
        # member training_model will be used in training method
        self.training_model = Model(inputs, outputs, name="test_model")

        # define loss for training model
        loss = self.get_loss(self.params.loss, reduction="auto")
        self.append_loss(loss, loss_weight=1, loss_name="loss", add_metric=False)

        if self.params.loss == "binary_crossentropy":
            val_metric = tf.keras.metrics.BinaryAccuracy()
        elif self.params.loss == "categorical_crossentropy":
            val_metric = tf.keras.metrics.CategoricalAccuracy()
        else:
            val_metric = loss

        self.append_metric(val_metric, "val_metric")

        if self.params.model_summary:
            self.training_model.summary()

        self.save_model_plot_to_file(self.training_model, "training_model", expand_nested=True)


def adjust_shape(array):
    shape = array.shape
    if len(shape) > 1:
        if shape[-1] == 1:
            return np.reshape(array, shape[:-1]), shape[:-1]
        else:
            return array, shape
    else:
        return array, shape


def create_datasets(train_data_shape,
                    cond_data_shape,
                    output_dim,
                    training_data_count):
    train_cond_data_shape = None
    if cond_data_shape is not None:
        train_cond_data_shape = [training_data_count] + list(cond_data_shape)

    input_train = np.random.randn(*train_data_shape)
    cond_input_train = None
    if cond_data_shape is not None:
        cond_input_train = np.random.randn(*train_cond_data_shape)

    # output training data
    output_train_integers = np.random.randint(0, output_dim, training_data_count)
    output_train = [np.zeros(output_dim) for _ in range(training_data_count)]
    for i in range(training_data_count):
        output_train[i][output_train_integers[i]] = 1
    output_train = np.array(output_train)

    return input_train, cond_input_train, output_train


def main(data_shape=(2,),
         # cond_data_shape=(1,),
         cond_data_shape=None,
         window_length=10,
         output_dim=1,
         training_data_count=10000,
         validation_data_count=1000,
         epochs=1,
         batch_size=32,
         train_epoch_size=1000,
         test_epoch_size=200,
         ):
    # ------------------------------------ OPTIONS ----------------------------------
    # folder for outputs
    # use sub folder in current working directory
    save_folder = os.path.join(os.getcwd(), "sequential_forward_model_files")
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    print(save_folder)

    print("----------- GENERATING SAMPLE TRAINING DATA -----------")

    data_window_shape = tuple([window_length] + list(data_shape))
    cond_data_window_shape = None
    if cond_data_shape is not None:
        cond_data_window_shape = tuple([window_length] + list(cond_data_shape))

    train_data_shape = [training_data_count] + list(data_shape)
    validation_data_shape = [validation_data_count] + list(data_shape)

    # ------------- TRAINING DATA ------------
    # input training data
    input_train, cond_input_train, output_train = create_datasets(train_data_shape,
                                                                  cond_data_shape,
                                                                  output_dim,
                                                                  training_data_count)
    input_train, _ = adjust_shape(input_train)

    output_train, _ = adjust_shape(output_train)

    if cond_data_shape is not None:
        cond_input_train, _ = adjust_shape(cond_input_train)

        if data_window_shape[-1] == 1:
            data_window_shape = data_window_shape[:-1]

        if cond_data_window_shape[-1] == 1:
            cond_data_window_shape = cond_data_window_shape[:-1]

    # ------------- VALIDATION DATA ------------
    input_test, cond_input_test, output_test = create_datasets(validation_data_shape,
                                                               cond_data_shape,
                                                               output_dim,
                                                               validation_data_count)

    input_test, _ = adjust_shape(input_test)

    output_test, _ = adjust_shape(output_test)

    if cond_data_shape is not None:
        cond_input_test, _ = adjust_shape(cond_input_test)

    # ------------- MODEL GENERATION ------------
    # build a sample model
    print("----------- BUILDING -----------")
    params = DotDict(save_folder=save_folder,
                     name="test_model",

                     # input definition
                     input_shape=data_window_shape,
                     conditional_shape=cond_data_window_shape,

                     # output definition
                     output_dim=output_dim,
                     output_activation="sigmoid",

                     # loss
                     loss="binary_crossentropy",
                     batch_size=batch_size,

                     # callbacks
                     checkpoint_monitor="binary_accuracy",
                     save_latest=True,

                     early_stopping_monitor="binary_accuracy",  # whether to use early stopping
                     early_stopping_patience=3,  # early stopping patience in epochs

                     lr_info=True,
                     tensorboard=True,
                     )

    keras_model = BasicKerasModel(**params)

    print("data shapes")
    print("\n")

    print("input_train_shape", input_train.shape)
    if cond_input_train is not None:
        print("cond_input_train_shape", cond_input_train.shape)

    print("output_train_shape", output_train.shape)

    print("input_test_shape", input_test.shape)
    if cond_input_test is not None:
        print("cond_input_train_shape", cond_input_test.shape)

    print("output_test_shape", output_test.shape)

    print("data_window_shape", data_window_shape)
    print("cond_data_window_shape", cond_data_window_shape)

    # ------------- DATA GENERATOR ------------
    # here we define a window data generator that randomly draws windows from our datasets
    # our time series data consists of 2 input variables and associated 1 output variable
    # our classification task is to decide if there is the value 1 in the output window array
    # therefore, we transform the window output: 1 if 1 is in window output, else 0
    # and use this transformation as postprocessing in the window data generator

    def transformation_center_input(array):
        return array - array[0]

    def transformation_jump_detection(array):
        return np.array([1 in array]).astype(int)

    if cond_input_train is not None:
        input_postprocessing = [transformation_center_input, None]
    else:
        input_postprocessing = [transformation_center_input]

    output_postprocessing = [transformation_jump_detection]

    # the window data generator expects dictionaries over "data instances" for input and output data
    # (or a suitable data generator)
    # each value is a list of numpy arrays where each numpy array is a synchronous time series dataset
    # for each model input or model output
    # for educational purposes, here we just use the generated data twice
    # note that keys for input and output data must match (maybe this should be changed)

    if cond_input_train is not None:
        all_train_input = {0: [input_train, cond_input_train],
                           1: [input_train[:1000], cond_input_train[:1000]],
                           2: [input_train[1000:3000], cond_input_train[1000:3000]]
                           }
    else:
        all_train_input = {0: [input_train],
                           1: [input_train[:1000]],
                           2: [input_train[1000:3000]]
                           }

    all_train_output = {0: [output_train],
                        1: [output_train[:1000]],
                        2: [output_train[1000:3000]]}

    train_generator_params = DotDict(batch_size=keras_model.params.batch_size,
                                     window_length=window_length,
                                     epoch_size=train_epoch_size,
                                     )

    train_input_data = WindowDataGenerator(input_data=all_train_input,
                                           output_data=all_train_output,
                                           input_postprocessing=input_postprocessing,
                                           output_postprocessing=output_postprocessing,
                                           **train_generator_params)

    if cond_input_test is not None:
        all_test_input = {0: [input_test, cond_input_test],
                          1: [input_test, cond_input_test]}
    else:
        all_test_input = {0: [input_test],
                          1: [input_test]}

    all_test_output = {0: [output_test],
                       1: [output_test]}

    test_generator_params = DotDict(batch_size=keras_model.params.batch_size,
                                    window_length=window_length,
                                    epoch_size=test_epoch_size
                                    )

    test_input_data = WindowDataGenerator(input_data=all_test_input,
                                          output_data=all_test_output,
                                          input_postprocessing=input_postprocessing,
                                          output_postprocessing=output_postprocessing,
                                          **test_generator_params)

    # train model
    print("----------- TRAINING -----------")
    keras_model.train(train_input_data=train_input_data,
                      train_output_data=None,
                      test_input_data=test_input_data,
                      test_output_data=None,
                      epochs=epochs
                      )

    # load model
    print("----------- LOADING -----------")
    # initialize new model
    loaded_keras_model = BasicKerasModel(**params)
    loaded_keras_model.load(folder_path=save_folder, model_name="model_latest.ckpt")

    print("----------- PREDICTING -----------")
    input_batch_example, output_batch_example = test_input_data.__getitem__(0)
    prediction_batch_example = loaded_keras_model.training_model.predict(input_batch_example)

    print(input_batch_example)
    print(output_batch_example)
    print(prediction_batch_example)

    test_generator_params = DotDict(batch_size=1,
                                    window_length=window_length,
                                    epoch_size=test_epoch_size,
                                    shuffle=False
                                    )

    test_input_data = WindowDataGenerator(input_data=all_test_input,
                                          output_data=all_test_output,
                                          input_postprocessing=input_postprocessing,
                                          output_postprocessing=output_postprocessing,
                                          **test_generator_params)

    predictions = loaded_keras_model.training_model.predict(test_input_data)
    print(predictions.shape)
    print(predictions)

    evaluation = loaded_keras_model.training_model.evaluate(test_input_data)
    print(evaluation)


if __name__ == '__main__':
    main()
