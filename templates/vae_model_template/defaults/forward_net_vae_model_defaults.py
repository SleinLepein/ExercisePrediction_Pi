from dotdict import DotDict

DEFAULT_MODELS_PATH = "../"

standard_params_vae = DotDict(
    # save / load
    save_folder=DEFAULT_MODELS_PATH,  # where to save stuff
    name="vae",  # name to be used
    # TODO: check
    model_load_name=None,

    # model params
    input_shapes=((28,),),  # tuple of tuple indicating inout shapes

    feature_extractor=DotDict(final_sampling=False,
                              name_prefix="feat_",
                              block_params=(DotDict(layer_params={"units": 32,
                                                                  "kernel_regularizer": {"type": "L2", "c": 0.0001}},
                                                    activation="relu",
                                                    layer_type="Dense",
                                                    normalization="batch_normalization",
                                                    ),
                                            DotDict(layer_params={"units": 8,
                                                                  "kernel_regularizer": {"type": "L2", "c": 0.0001}},
                                                    activation="relu",
                                                    layer_type="Dense",
                                                    normalization="batch_normalization",
                                                    laterals=DotDict(act=True),  # important
                                                    ),
                                            )
                              ),  # dictionary providing definition of feature extractor
    feature_reconstructors=(DotDict(final_sampling=False,
                                    name_prefix="feat_re_",
                                    block_params=(DotDict(layer_params={"units": 8,
                                                                        "kernel_regularizer": {"type": "L2",
                                                                                               "c": 0.0001}},
                                                          activation="relu",
                                                          layer_type="Dense",
                                                          normalization="batch_normalization",
                                                          ),
                                                  DotDict(layer_params={"units": 8,
                                                                        "kernel_regularizer": {"type": "L2",
                                                                                               "c": 0.0001}},
                                                          activation="relu",
                                                          layer_type="Dense",
                                                          normalization="batch_normalization",
                                                          ),
                                                  )
                                    ),
                            ),  # dictionary providing definition of feature extractor

    latent_extractors=(DotDict(final_sampling=True,
                               name_prefix="lat_",
                               block_params=(DotDict(layer_params={"units": 2,
                                                                   "kernel_regularizer": {"type": "L2", "c": 0.0001}},
                                                     activation="relu",
                                                     layer_type="Dense",
                                                     normalization="batch_normalization",
                                                     ),
                                             ),
                               ),
                       ),  # list of dictionaries providing definitions of lateral models for latents
    latent_reconstructors=(DotDict(final_sampling=False,
                                   name_prefix="lat_re_",
                                   block_params=(DotDict(layer_params={"units": 4,
                                                                       "kernel_regularizer": {"type": "L2",
                                                                                              "c": 0.0001}},
                                                         activation="relu",
                                                         layer_type="Dense",
                                                         normalization="batch_normalization",
                                                         ),
                                                 DotDict(layer_params={"units": 8,
                                                                       "kernel_regularizer": {"type": "L2",
                                                                                              "c": 0.0001}},
                                                         activation="relu",
                                                         layer_type="Dense",
                                                         normalization="batch_normalization",
                                                         ),
                                                 ),
                                   ),
                           ),  # list of dictionaries providing definitions of lateral models for latents
    conditional_extractor=None,  # dictionary with definition of global conditional pre processor

    # TODO: add noise
    # encoder_noise=False,
    # noise_std=0.1,

    output_activation="sigmoid",  # activation for reconstructions

    # losses
    loss="continuous_crossentropy",  # reconstruction loss / model family
    loss_weight=1,  # weight of reconstruction loss

    use_kl_loss=True,  # whether to use KL loss (always true in VAE)

    kl_loss_weight=1.0,  # weight of KL loss if no annealing is used
    schedule_kl_loss=True,  # whether to use annealing for KL weight
    kl_annealing_params=DotDict(start_epoch=0,
                                annealing_epochs=5,
                                start_value=0.0,
                                end_value=1.0,
                                method="linear"),  # annealing parameters

    use_tc_loss=True,  # whether to use KL loss (always true in TC-VAE, TC = Total Correlation)
    tc_loss_weight=1.0,  # weight of TC loss if no annealing is used
    schedule_tc_loss=True,  # whether to use annealing for TC weight
    tc_annealing_params=DotDict(start_epoch=0,
                                annealing_epochs=5,
                                start_value=0.0,
                                end_value=1.0,
                                method="linear"),  # annealing parameters

    use_mmd_loss=True,  # whether to use MMD loss (always true in info-VAE, MMD = Maximum Mean Discrepancy)
    mmd_loss_weight=1.0,  # weight of MMD loss if no annealing is used
    schedule_mmd_loss=True,  # whether to use annealing for MMD weight
    mmd_annealing_params=DotDict(start_epoch=0,
                                 annealing_epochs=5,
                                 start_value=0.0,
                                 end_value=1.0,
                                 method="linear"),  # annealing parameters
)
