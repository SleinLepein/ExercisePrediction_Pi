from dotdict import DotDict

standard_params_keras = DotDict(
        # meta
        save_folder="./",  # where to save stuff
        name="model",  # name to be used
        plot_model=True,  # whether to plot models via graphviz (see graphviz documentation)
        model_summary=True,  # whether to print model summary to console

        # optimizer
        optimizer="adam",  # which optimizer to use
        lr=0.75 * 1e-2,  # which starting learning rate to use
        initial_epoch=0,  # which initial epoch to use (used for resuming of training)
        epochs=1,  # how many epochs to train (can also be used in train method)
        batch_size=32,  # which batch size to use

        # callbacks
        lr_info=True,  # whether to print current learning rate at start of every epoch

        save_weights_only=True,  # whether to save model weights only
        save_latest=True,  # whether to save model after every epoch

        checkpoint_monitor=None,  # name of validation metric to be monitored, e.g., "val_loss", "val_acc", ...
        checkpoint_save_best_only=True,  # whether to save only best models during training

        early_stopping_monitor=None,  # whether to use early stopping
        early_stopping_patience=10,  # early stopping patience in epochs

        tensorboard=True,  # whether to use TensorBoard (see TensorBoard documentation)
        tensorboard_update_freq="epoch",  # update frequency
        tensorboard_write_graph=False,  # whether to write model graphs
        tensorboard_write_images=False,  # whether to write TensorBoard predefined visualizations
        tensorboard_histogram_freq=0,  # frequency for histograms
        custom_file_writer=False,  # whether to use a custom file write for own stuff, e.g., custom visualizations

        # learning rate scheduler
        lr_scheduler=None,  # name of learning rate scheduler to be used (more need to be implemented)

        # learning rate reduce on plateau (if not scheduling)
        lr_reducer_monitor=None,  # name of validation metric to be used to reduce learning rate on pleteau
        lr_reducer_factor=0.95,  # reducer factor
        lr_reducer_cooldown=0,  # reducer cooldown in epochs
        lr_reducer_patience=1,  # reducer patience in epochs
        lr_reducer_verbose=1,  # reducer verbosity
        lr_reducer_min_lr=0.5e-6,  # minimal learning rate for reducer

        run_tf_function=False,  # experimental TF run (maybe not needed, see Eager Execution)
    )
