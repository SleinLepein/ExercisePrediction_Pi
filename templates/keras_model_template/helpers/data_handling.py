import numpy as np


def extend_data(first_data, second_data):
    """
    Convenience method to extend a data source by another data source.

    Parameters
    ----------
    first_data
    second_data

    Returns
    -------
    joined data
    """

    if not isinstance(first_data, list):
        first_data = [first_data]
    if not isinstance(second_data, list):
        second_data = [second_data]

    # ----------------- return -----------------

    return first_data + second_data


def get_batch_data(data, indexes):
    """

    Parameters
    ----------
    data
    indexes

    Returns subset of data with given indexes
    -------

    """

    if isinstance(data, list):
        batch_data = []
        for sub_data in data:

            if isinstance(sub_data, list):
                # TODO implement
                raise NotImplementedError("sub_data is of type list which is not implemented, yet")
            else:
                batch_data.append(sub_data[indexes])

    else:
        batch_data = data[indexes]

    return batch_data


def get_window_batch_data(data, indexes, window_length):
    """
        Parameters
        ----------
        data
        indexes
        window_length

        Returns subset of data with given indexes
        -------

        """

    def get_window(single_data, pos, length):

        window = []

        # convert to list if necessary
        if not isinstance(single_data, list):
            single_data = [single_data]

        # for each data branch of this single data example
        for i, temp_data in enumerate(single_data):
            # extract time window
            temp = temp_data[pos:pos + length]
            window.append(temp)

        return window

    if not isinstance(data, list):
        data = [data]

    batch_shapes = []
    for sub_data in data[0]:
        temp_shape = [len(indexes), window_length] + list(sub_data.shape[1:])
        batch_shapes.append(temp_shape)

    batch_data = [np.zeros(batch_shapes[k]) for k in range(len(data[0]))]

    for i, index in enumerate(indexes):
        random_window = get_window(data[index[0]], index[1], window_length)
        for k, branch in enumerate(random_window):
            batch_data[k][i] = branch

    for k, sub_batch in enumerate(batch_data):
        if sub_batch.shape[-1] == 1:
            batch_data[k] = np.reshape(sub_batch, sub_batch.shape[:-1])

    return batch_data


def transform_along_batch_axis(array: np.ndarray,
                               transformation) -> np.ndarray:
    """

    Parameters
    ----------
    array
    transformation : func
        Needs to be a function from np.ndarray to np.ndarray

    Returns
    -------

    """
    if transformation is not None:
        # print(array.shape)
        if len(array) > 0:
            temp = array[0]
            transformed_temp = transformation(temp)
            transformed_shape = transformed_temp.shape

            new_array = np.zeros(tuple([array.shape[0]] + list(transformed_shape)))

            for k in range(array.shape[0]):
                new_array[k] = transformation(array[k])

            return new_array
        else:
            raise ValueError("Input to transformation does not seem to be a suitable array. Check!")
    else:
        return array

