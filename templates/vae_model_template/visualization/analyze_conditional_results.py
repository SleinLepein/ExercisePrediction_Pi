import os
import gc

gc.collect()

from dotdict import DotDict
from copy import copy

import numpy as np

import holoviews as hv

MNIST_COLORS = {"white": "#ffffff",
                "red": "#fc0303",
                "orange": "#ff9900",
                "yellow": "#fffb00",
                "lightgreen": "#81fc87",
                "darkgreen": "#00a349",
                "mint": "#008f70",
                "lightblue": "#0398fc",
                "darkblue": "#0303fc",
                "black": "#000000"}

MNIST_CMAP = list(MNIST_COLORS.values())


def get_one_hot_vector(vector):
    argmax = np.argmax(vector)
    one_hot = np.zeros(vector.shape)
    one_hot[argmax] = 1
    return one_hot


def plot_embedding_whiskers(cond_embedding):
    hv.extension("bokeh")

    try:
        data = []
        dimensions = []

        for i in range(cond_embedding.shape[-1]):
            data.extend(cond_embedding[:, i])
            dimensions.extend([i for _ in range(cond_embedding.shape[0])])

        boxwhisker = hv.BoxWhisker((dimensions, data),
                                   ['Embedding Component'], 'Value').sort()

        boxwhisker.opts(box_color='white',
                        height=300,
                        show_legend=False,
                        whisker_color='gray',
                        width=800,
                        xrotation=90,
                        )

        return boxwhisker
    finally:
        # free memory, hopefully!
        del data, dimensions, boxwhisker
        gc.collect()


def plot_codes_with_cond_embedding(mean_codes,
                                   cond_embedding=None,
                                   colors=None,
                                   alpha=0.3):
    try:
        imgs = []

        min_alpha = 0
        img = None

        plot_counts = int(mean_codes.shape[-1] / 2)

        for i in range(plot_counts):

            temp_temp_img = None

            if cond_embedding is not None:
                for j in range(cond_embedding.shape[-1]):

                    temp_embedding = np.maximum(cond_embedding[:, j], min_alpha)

                    data = zip(mean_codes[:, 2 * i], mean_codes[:, 2 * i + 1], temp_embedding)
                    temp_img = hv.Scatter(data, vdims=["z2", "alpha"])
                    temp_img = temp_img.opts(xlabel=f"latent {2 * i}",
                                             ylabel=f"latent {2 * i + 1}",
                                             alpha=hv.dim("alpha"),
                                             color=colors[j],
                                             bgcolor="darkgrey",
                                             )

                    if j == 0:
                        temp_temp_img = temp_img
                    else:
                        temp_temp_img = temp_temp_img * temp_img

                    del temp_img, temp_embedding
            else:
                data = zip(mean_codes[:, 2 * i], mean_codes[:, 2 * i + 1])
                temp_temp_img = hv.Scatter(data)
                temp_temp_img = temp_temp_img.opts(xlabel=f"latent {2 * i}",
                                                   ylabel=f"latent {2 * i + 1}",
                                                   alpha=alpha,
                                                   color="black",
                                                   bgcolor="darkgrey",
                                                   )

            imgs.append(temp_temp_img)

            if i == 0:
                img = imgs[0]
            else:
                img = img + imgs[-1]

        img = img.cols(4)

        return img
    finally:
        # free memory, hopefully
        del img, imgs, temp_temp_img, data
        gc.collect()


def plot_reconstructions(variational_ae,
                         data,
                         **params):
    standard_params = DotDict(batch_size=32,
                              marker_size=40,
                              alpha=0.9,
                              x_count=10,
                              y_count=10,
                              figsize=np.array([400, 400]),
                              label_fontsize=18,
                              ticks_fontsize=16,
                              title="Reconstructions",
                              title_fontsize=20,
                              #                               save_folder="./",
                              #                               filename="single_reconstructions",
                              hv_extension="bokeh",
                              variational=True,
                              random_samples=False,
                              invert=False,
                              cmap="fire",
                              color_levels=None
                              )

    if params is not None:
        standard_params.update(params)
    params = standard_params

    hv.extension(params.hv_extension)

    try:
        if isinstance(data, list) or isinstance(data, tuple):
            input_data = data[0]
            conditions_data = data[1]
            example = [input_data[:1], conditions_data[:1]]
        else:
            input_data = data
            conditions_data = None
            example = input_data[:1]

        # calculate number of needed examples and select data
        plot_count = params.x_count * params.y_count
        data_count = len(input_data)

        if plot_count < data_count:
            input_data = input_data[:plot_count]
            if conditions_data is not None:
                conditions_data = conditions_data[:plot_count]

        # define blank image
        test_rec = variational_ae.predict(example)
        without_third_channel = len(test_rec.shape) == 3

        if without_third_channel:
            x_size, y_size = test_rec.shape[-2:]
            figure = np.zeros((x_size * params.x_count, y_size * params.y_count))
            figure_rec = np.zeros((x_size * params.x_count, y_size * params.y_count))
            figure_acc = np.zeros((x_size * params.x_count, y_size * params.y_count))
        else:
            x_size, y_size = test_rec.shape[-3:-1]
            figure = np.zeros((x_size * params.x_count, y_size * params.y_count, 3))
            figure_rec = np.zeros((x_size * params.x_count, y_size * params.y_count, 3))
            figure_acc = np.zeros((x_size * params.x_count, y_size * params.y_count, 3))

        # generate reconstructions of examples
        if conditions_data is None:
            reconstructions = variational_ae.predict(input_data, batch_size=params.batch_size)
        else:
            reconstructions = variational_ae.predict([input_data, conditions_data], batch_size=params.batch_size)

        # generate images for originals and reconstructions
        for i in range(params.x_count):
            for j in range(params.y_count):
                if params.random_samples:
                    sample = np.random.randint(len(input_data))
                    x_decoded = reconstructions[sample]
                    true_image = input_data[sample]
                else:
                    data_index = i * params.x_count + j
                    x_decoded = reconstructions[data_index]
                    true_image = input_data[data_index]

                if params.invert:
                    x_decoded = np.ones(x_decoded.shape) - x_decoded
                    true_image = np.ones(true_image.shape) - true_image

                # insert example into figure
                x_start_index = i * x_size
                x_end_index = (i + 1) * x_size
                y_start_index = j * y_size
                y_end_index = (j + 1) * y_size

                if without_third_channel:
                    decoded = x_decoded.reshape(test_rec.shape[1], test_rec.shape[2])

                    figure[x_start_index: x_end_index, y_start_index: y_end_index] \
                        = true_image

                    figure_rec[x_start_index: x_end_index, y_start_index: y_end_index] \
                        = decoded

                    figure_acc[x_start_index: x_end_index, y_start_index: y_end_index] \
                        = np.apply_along_axis(get_one_hot_vector, -1, decoded)
                else:
                    decoded = x_decoded.reshape(test_rec.shape[1], test_rec.shape[2], test_rec.shape[3])

                    figure[x_start_index: x_end_index,
                           y_start_index: y_end_index,
                           0: test_rec.shape[-1]] = true_image

                    figure_rec[x_start_index: x_end_index,
                               y_start_index: y_end_index,
                               0: test_rec.shape[-1]] = decoded

                    figure_acc[x_start_index: x_end_index,
                               y_start_index: y_end_index,
                               0: test_rec.shape[-1]] = np.apply_along_axis(get_one_hot_vector, -1, decoded)

        # image with original data
        img = hv.Image(figure)
        img = img.opts(title="Originals",
                       height=int(params.figsize[0]),
                       width=int(params.figsize[1]),
                       xaxis=None,
                       yaxis=None,
                       fontsize={"xticks": params.ticks_fontsize,
                                 "yticks": params.ticks_fontsize,
                                 "xlabel": params.label_fontsize,
                                 "ylabel": params.label_fontsize,
                                 "title": params.title_fontsize},
                       aspect="equal",
                       cmap=params.cmap,
                       color_levels=params.color_levels,
                       axiswise=True,
                       clim=(0, 1)
                       )

        # image with reconstructed data
        img_2 = hv.Image(figure_rec)
        img_2 = img_2.opts(title="Sample Reconstructions",
                           height=int(params.figsize[0]),
                           width=int(params.figsize[1]),
                           xaxis=None,
                           yaxis=None,
                           fontsize={"xticks": params.ticks_fontsize,
                                     "yticks": params.ticks_fontsize,
                                     "xlabel": params.label_fontsize,
                                     "ylabel": params.label_fontsize,
                                     "title": params.title_fontsize},
                           aspect="equal",
                           cmap=params.cmap,
                           color_levels=params.color_levels,
                           axiswise=True,
                           clim=(0, 1)
                           )

        # image with residuals
        img_3 = hv.Image(figure_acc)
        img_3 = img_3.opts(title="Sample Hard Reconstructions",
                           height=int(params.figsize[0]),
                           width=int(params.figsize[1]),
                           xaxis=None,
                           yaxis=None,
                           fontsize={"xticks": params.ticks_fontsize,
                                     "yticks": params.ticks_fontsize,
                                     "xlabel": params.label_fontsize,
                                     "ylabel": params.label_fontsize,
                                     "title": params.title_fontsize},
                           aspect="equal",
                           cmap=params.cmap,
                           color_levels=params.color_levels,
                           axiswise=True,
                           clim=(0, 1)
                           )

        # image with residuals
        img_4 = hv.Image(figure_acc - figure)
        img_4 = img_4.opts(title="Sample Hard Residuals",
                           height=int(params.figsize[0]),
                           width=int(params.figsize[1]),
                           xaxis=None,
                           yaxis=None,
                           fontsize={"xticks": params.ticks_fontsize,
                                     "yticks": params.ticks_fontsize,
                                     "xlabel": params.label_fontsize,
                                     "ylabel": params.label_fontsize,
                                     "title": params.title_fontsize},
                           aspect="equal",
                           cmap="PuOr",
                           color_levels=params.color_levels,
                           axiswise=True,
                           clim=(-1, 1)
                           )

        img = (img + img_2 + img_3 + img_4).cols(2)
        img.opts(title=params.title)

        return img

    finally:
        # free memory, hopefully!
        try:
            del img
        except UnboundLocalError:
            print("img already closed")

        del img_2, img_3, img_4, figure, figure_rec, reconstructions, true_image
        del decoded, x_decoded, input_data, conditions_data
        gc.collect()


def plot_reconstruction_manifold(decoder,
                                 reference_codes=None,
                                 dim_0=0,
                                 dim_1=1,
                                 conditionals=None,
                                 **params):
    """
    Visualize partial output of a VAEs decoder.

    Parameters
    ----------
    decoder : Keras model
        (flattened) decoder of a VAE
    reference_codes : list of np.array,
        list of reference codes (e.g. latents for some specific data input)
    dim_0 : int
        index of first latent variable for visualization, default: 0
    dim_1 : int
        index of second latent variable for visualization, default : 1
    conditionals : list of np.array
        list with conditionals (e.g. one-hot-encoded labels)

    Returns
    -------
    list of Bokeh images
        visualization of partial reconstruction manifolds

    keyword arguments in params
    -------
    x_range : tuple
        default: (-3, 3)
    y_range : tuple
        default: (-3, 3)
    x_count : int
        default: 30
    y_count : int
        default: 30
    x_size : int
        default: 28
    y_size : int
        default: 28
    figsize : tuple
        default: (800, 800)
    label_fontsize : int
        default: 18
    ticks_fontsize : int
        default: 16
    title : str
        default: "Reconstruction Manifold"
    title_fontsize : int
        default: 20
    cmap : str
        default: "fire"
    color_levels : tuple or None,
        default: None
    invert : bool
        default: False
    save_folder : str
        default: "./"
    filename : str
        default: "reconstructions"
    hv_extension : str
        default: "bokeh",
    save : bool
        default: False
    """

    standard_params = DotDict(x_range=np.array([-3, 3]),
                              y_range=np.array([-3, 3]),
                              x_count=30,
                              y_count=30,
                              x_size=28,
                              y_size=28,
                              figsize=np.array([800, 800]),
                              label_fontsize=18,
                              ticks_fontsize=16,
                              title="Reconstruction Manifold",
                              title_fontsize=20,
                              cmap="fire",
                              color_levels=None,
                              invert=False,
                              save_folder="./",
                              filename="reconstructions",
                              hv_extension="bokeh",
                              save=False,
                              use_numerical_label=False)

    def get_code_for_reference(x, y, dim_x, dim_y, reference):

        """
        Generate new latent code from reference by replacing values at two specified components with specified values.

        Parameters
        ----------
        x : float
            first new value
        y : float
            second new value
        dim_x : int
            index of first component whose value should be overridden by x
        dim_y : int
            index of second component whose value should be overridden by y
        reference : np.array

        Returns
        -------
        np.array
            new latent code
        """

        # -------------- initialization ---------------

        if reference is None:
            assert 0 <= dim_x < 2
            assert 0 <= dim_y < 2
            latent = np.zeros(2)
        else:
            latent = copy(reference)

        # -------------- code construction ---------------

        latent[dim_x] = x
        latent[dim_y] = y

        # -------------- return ---------------

        return latent

    def construct_input(xs, ys, reference, dim_x, dim_y, condition, use_numerical_label=False):

        """
        Construct decoder input data including conditional input by replacing values of two components of a
        reference latent code with predefined values.

        Parameters
        ----------
        xs : np.array
        ys : np.array
        reference : np.array
        dim_x : int
        dim_y : int
        condition : np.array
        use_numerical_label : bool

        Returns
        -------
        np.array
            decoder inputs
        """

        # -------------- initialization ---------------

        assert len(xs) == len(ys)

        if dim_x is None:
            dim_x = 0

        if dim_y is None:
            dim_y = 1

        if reference is not None:
            assert 0 <= dim_x < len(reference)
            assert 0 <= dim_y < len(reference)

        latents = []
        conditions = []

        # -------------- code construction ---------------

        for k, x in enumerate(xs):
            latent = get_code_for_reference(xs[k], ys[k], dim_x, dim_y, reference)
            latents.append(latent)
            if condition is not None:
                conditions.append(condition)

        latents = np.array(latents)
        conditions = np.array(conditions)

        if use_numerical_label:
            conditions = np.expand_dims(np.argmax(conditions, axis=-1), axis=-1)

        # -------------- return ---------------

        if condition is None:
            return latents
        else:
            return [latents, conditions]

    # -------------- initialization ---------------

    if params is not None:
        standard_params.update(params)
    params = standard_params

    hv.extension(params.hv_extension)
    use_numerical_label = params.use_numerical_label

    bounds = (params.x_range[0], params.y_range[0], params.x_range[1], params.y_range[1])

    if conditionals is None:
        conditionals = [np.zeros(10) for _ in range(10)]

        for i in range(len(conditionals)):
            conditionals[i][i] = 1

    if reference_codes is None:
        reference_codes = [None for _ in range(len(conditionals))]

    # -------------- plot definition ---------------

    images = []

    for int_label, digit_conditional in enumerate(conditionals):

        figure = np.zeros((params.x_size * params.x_count, params.y_size * params.y_count))

        grid_x = np.linspace(params.x_range[0], params.x_range[1], params.x_count)
        grid_y = np.linspace(params.y_range[0], params.y_range[1], params.y_count)[::-1]

        label = conditionals[int_label]
        reference_code = reference_codes[int_label]

        # -------------- input code construction and decoder reconstructions ---------------

        inputs_x = []
        inputs_y = []

        for i, yi in enumerate(grid_y):
            for j, xj in enumerate(grid_x):
                inputs_x.append(xj)
                inputs_y.append(yi)

        decoder_inputs = construct_input(inputs_x, inputs_y, reference_code, dim_0, dim_1, label, use_numerical_label)
        reconstructions = decoder.predict(decoder_inputs)

        # -------------- sub image positioning ---------------

        index = 0
        for i, yi in enumerate(grid_y):
            for j, xj in enumerate(grid_x):
                digit = reconstructions[index].reshape(params.x_size, params.y_size)
                figure[i * params.x_size: (i + 1) * params.x_size, j * params.y_size: (j + 1) * params.y_size] = digit

                index += 1

        if params.invert:
            figure = np.ones(figure.shape) - figure

        # -------------- image definition ---------------

        img = hv.Image(figure, bounds=bounds)
        img = img.opts(xlabel=f"latent codes {dim_0}",
                       ylabel=f"latent codes {dim_1}",
                       title=params.title + f" for digit {int_label}",
                       height=int(params.figsize[0]),
                       width=int(params.figsize[1]),
                       fontsize={"xticks": params.ticks_fontsize,
                                 "yticks": params.ticks_fontsize,
                                 "xlabel": params.label_fontsize,
                                 "ylabel": params.label_fontsize,
                                 "title": params.title_fontsize},
                       aspect="equal",
                       cmap=params.cmap,
                       color_levels=params.color_levels)

        images.append(img)

        # -------------- optional save to file ---------------

        if params.save:
            file_path = os.path.join(params.save_folder, params.filename +
                                     f"_{int_label}_latents_{dim_0}_{dim_1}.png")
            hv.save(img, file_path)

    # -------------- return ---------------

    return images
