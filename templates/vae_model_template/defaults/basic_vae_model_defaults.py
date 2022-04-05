from dotdict import DotDict

DEFAULT_MODELS_PATH = "../"

standard_params_basic_vae = DotDict(
        # save / load
        save_folder=DEFAULT_MODELS_PATH,  # where to save stuff
        name="vae",  # name to be used

        # model params
        input_shapes=((28, 28, 1),),  # tuple of tuple indicating inout shapes

        # rec loss
        loss="mse",  # reconstruction loss / model family
        loss_weight=1,  # weight of reconstruction loss

        # VAE regularization losses
        use_kl_loss=True,  # whether to use KL loss (always true in VAE)
        # TODO: add KL importance sampling
        # kl_importance_sampling=False,
        kl_loss_weight=1.0,  # weight of KL loss if no annealing is used
        schedule_kl_loss=False,  # whether to use annealing for KL weight
        kl_annealing_params=DotDict(start_epoch=0,
                                    annealing_epochs=5,
                                    start_value=0.0,
                                    end_value=1.0,
                                    method="linear"),  # annealing parameters

        use_tc_loss=False,  # whether to use TC loss (always true in TC-VAE)
        tc_loss_weight=1.0,  # weight of TC loss if no annealing is used
        schedule_tc_loss=False,  # whether to use annealing for TC weight
        tc_annealing_params=DotDict(start_epoch=0,
                                    annealing_epochs=5,
                                    start_value=0.0,
                                    end_value=1.0,
                                    method="linear"),  # annealing parameters

        use_mmd_loss=False,  # whether to use MMD loss (always true in info-VAE)
        mmd_loss_weight=1.0,  # weight of MMD loss if no annealing is used
        schedule_mmd_loss=False,  # whether to use annealing for MMD weight
        mmd_annealing_params=DotDict(start_epoch=0,
                                     annealing_epochs=5,
                                     start_value=0.0,
                                     end_value=1.0,
                                     method="linear"),  # annealing parameters
    )
