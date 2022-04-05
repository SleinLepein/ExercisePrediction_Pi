import numpy as np
import tensorflow as tf
from dotdict import DotDict

from templates.keras_model_template.keras_data_generator_class import KerasDataGenerator
from templates.keras_model_template.helpers.data_handling import get_window_batch_data, transform_along_batch_axis


class WindowDataGenerator(KerasDataGenerator):
    """
    This class defines a window data generator.
    """

    standard_window_params = DotDict(batch_size=32,  # number of windows per batch
                                     window_length=16,  # length of windows
                                     shuffle=True,  # shuffle ordering for training
                                     epoch_size=1000,  # number of batches per epoch

                                     max_accumulations=50000,  # maximal number of choices for datasets per epoch
                                     )

    def __init__(self,
                 input_data,
                 output_data,
                 input_postprocessing=None,
                 output_postprocessing=None,
                 **params_dict):
        """

        Parameters
        ----------
        params_dict
        """

        # ----------------- parameter loading and preparation -----------------
        super().__init__(**params_dict)

        # update default parameters with given parameters
        self.update_params(self.input_params, self.standard_window_params)

        # ----------------- initialize -----------------
        self.batch_size = self.params.batch_size
        self.input_postprocessing = input_postprocessing
        self.output_postprocessing = output_postprocessing

        self.data_generator = None
        self.inputs = None
        self.input_data = None
        self.outputs = None
        self.output_data = None
        self.data_count = None
        self.index_map = {}
        self.index_map_inverse = {}
        self.indexes = None
        self.length = None

        # prepare inputs and outputs
        self._prepare_inputs_outputs(input_data, output_data)

        # ----------------- preparations for next epoch -----------------
        self.on_epoch_end()

    def _prepare_inputs_outputs(self, inputs, outputs):
        """
        Check the shape of inputs and outputs to be able to handle individual behaviours.

        Parameters
        ----------
        inputs
        outputs

        Returns
        -------

        """

        if isinstance(inputs, tf.keras.utils.Sequence):
            self.data_generator = inputs
            self.input_data = None
            self.output_data = None
        else:
            self.data_generator = None

            assert (inputs is not None) & (outputs is not None)

            if not isinstance(inputs, dict):
                self.input_data = {0: inputs}
            else:
                self.input_data = inputs

            if not isinstance(outputs, dict):
                self.output_data = {0: outputs}
            else:
                self.output_data = outputs

    def _accumulate_data(self, min_data_count=1, max_iterations=10):
        """
        Accumulate data from data generator.

        Parameters
        ----------
        min_data_count
        max_iterations

        Returns
        -------

        """
        data_count = 0
        iteration = 0

        inputs = []
        outputs = []

        if self.data_generator is not None:
            data_examples_count = self.data_generator.__len__()
        else:
            data_examples_count = len(self.input_data)

        # first use any dataset at most once but randomly
        if self.params.shuffle:
            random_choices = np.random.choice(data_examples_count, data_examples_count, False)
        else:
            random_choices = range(data_examples_count)

        # accumulate data while there is still unused data and while there is not enough data
        while (data_count < min_data_count) & (iteration < max_iterations):
            if iteration == max_iterations:
                raise Exception(f"Data generator iterated over data sets {max_iterations} times.")

            # choose random item
            if iteration >= data_examples_count:
                choice = np.random.choice(data_examples_count, 1)[0]
            else:
                choice = random_choices[iteration]

            if self.data_generator is not None:
                temp_inputs, temp_outputs = self.data_generator.__getitem__(choice)
            else:
                temp_inputs = self.input_data[choice]
                temp_outputs = self.output_data[choice]

            # convert to lists if necessary and compute new data count
            # inputs
            if not isinstance(temp_inputs, list):
                temp_count = len(temp_inputs)
                temp_inputs = [temp_inputs]
            else:
                temp_count = len(temp_inputs[0])

            # outputs
            if not isinstance(temp_outputs, list):
                temp_outputs = [temp_outputs]

            # append data if large enough for windows of given length
            if temp_count >= self.params.window_length:
                # append new inputs and new outputs to final input and output list
                inputs.append(temp_inputs)
                outputs.append(temp_outputs)

                # update data count
                data_count += temp_count - self.params.window_length + 1

            iteration += 1

        self.inputs = inputs
        self.outputs = outputs
        self.data_count = data_count

    def _determine_index_map(self):

        index = 0
        self.index_map = {}
        for i, sub_inputs in enumerate(self.inputs):
            # sub_inputs is list of input tensors / numpy arrays
            for window in range(len(sub_inputs[0]) - self.params.window_length + 1):
                if index < self.params.epoch_size * self.params.batch_size:
                    self.index_map[index] = (i, window)
                    index += 1
                else:
                    break

        if (not self.params.shuffle) & (self.params.batch_size == 1):
            self.index_map_inverse = dict(zip(list(self.index_map.values()), list(self.index_map.keys())))

    def _determine_length(self):
        """
        Determine the count of total training examples and set corresponding class member length.

        Returns
        -------
        None
        """

        if self.inputs is not None:
            self.length = len(self.index_map)
        else:
            self.length = None

    def __len__(self):
        """
        Compute count of batches per epoch.

        Returns
        -------
        int
            Count of batches per epoch
        """

        return int(np.floor(self.length / self.batch_size))

    def __getitem__(self, index):
        """
        Get batch with specific index out of every batches.

        Parameters
        ----------
        index : int

        Returns
        -------
        tuple
            batch input data and corresponding batch output data
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        index_tuples = [self.index_map[index] for index in indexes]

        # Get batch data
        # window input data
        input_data = get_window_batch_data(self.inputs,
                                           index_tuples,
                                           self.params.window_length)

        # postprocessing
        if self.input_postprocessing is not None:
            for i, sub_data in enumerate(input_data):
                input_data[i] = transform_along_batch_axis(input_data[i], self.input_postprocessing[i])

        # window output data
        output_data = get_window_batch_data(self.outputs,
                                            index_tuples,
                                            self.params.window_length)

        # postprocessing
        if self.output_postprocessing is not None:
            for i, sub_data in enumerate(output_data):
                output_data[i] = transform_along_batch_axis(output_data[i], self.output_postprocessing[i])

        return input_data, output_data

    def on_epoch_end(self):
        """
        Randomize ordering of data indexes for random batches at the end of every epoch.

        Returns
        -------
        None
        """

        # accumulate data
        self._accumulate_data(min_data_count=self.params.epoch_size * self.params.batch_size,
                              max_iterations=self.params.max_accumulations)

        # determine index map and length
        self._determine_index_map()
        self._determine_length()
        self.indexes = np.arange(self.length)

        # randomize
        if self.params.shuffle:
            np.random.shuffle(self.indexes)


def main(data_count=10,
         data_length_range=(500, 2000),
         input_dim=2,
         p=0.001):

    params = DotDict(batch_size=1,
                     window_length=8,
                     epoch_size=5000,
                     shuffle=False,
                     max_accumulations=10000,
                     )

    input_data = {}
    output_data = {}

    for i in range(data_count):
        temp_length = np.random.randint(data_length_range[0], data_length_range[1], 1)[0]
        input_data[i] = np.random.randn(temp_length, input_dim)
        output_data[i] = np.random.choice(2, temp_length, p=[1-p, p])

    print("First rows of second dataset")
    print(input_data[1][:20])

    window_data = WindowDataGenerator(input_data, output_data, **params)

    # first batch
    for i in range(4):
        print(f"window {i} of second dataset")
        index = window_data.index_map_inverse[(1, i)]
        print(index)
        batch_input, batch_output = window_data.__getitem__(index)
        print(batch_input[0])


if __name__ == '__main__':
    main()
