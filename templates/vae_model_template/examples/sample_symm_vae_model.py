from dotdict import DotDict

from templates.vae_model_template.symmetric_vae_model_class import SymmetricVAE

fe_params_0 = DotDict(block_params=(DotDict(layer_params={"filters": 64},
                                            activation="relu",
                                            layer_type="Conv2D"),
                                    DotDict(layer_params={"filters": 64},
                                            activation="relu",
                                            laterals=DotDict(act=True),
                                            layer_type="Conv2D"),
                                    DotDict(layer_params={"filters": 64},
                                            activation="relu",
                                            layer_type="Conv2D"),
                                    DotDict(layer_params={"filters": 32},
                                            activation="relu",
                                            laterals=DotDict(act=True),
                                            layer_type="Conv2D"),
                                    DotDict(layer_params={"filters": 32},
                                            activation="relu",
                                            layer_type="Conv2D"),
                                    DotDict(layer_params={"filters": 32},
                                            activation="relu",
                                            laterals=DotDict(act=True),
                                            layer_type="Conv2D"),
                                    )
                      )
laterals_0 = DotDict(final_sampling=True,
                     block_params=(DotDict(layer_params={"filters": 2},
                                           activation="relu",
                                           layer_type="Conv2D"),
                                   DotDict(layer_params={"units": 32},
                                           activation="relu",
                                           layer_type="Dense",
                                           conditional_input=True),
                                   DotDict(layer_params={"units": 2},
                                           layer_type="Dense"),
                                   )
                     )
laterals_1 = DotDict(final_sampling=True,
                     block_params=(DotDict(layer_params={"filters": 2},
                                           activation="relu",
                                           layer_type="Conv2D"),
                                   DotDict(layer_params={"units": 32},
                                           activation="relu",
                                           layer_type="Dense",
                                           conditional_input=True),
                                   DotDict(layer_params={"units": 2},
                                           layer_type="Dense"),
                                   )
                     )
laterals_2 = DotDict(final_sampling=True,
                     block_params=(DotDict(layer_params={"filters": 2},
                                           activation="relu",
                                           layer_type="Conv2D"),
                                   DotDict(layer_params={"units": 32},
                                           activation="relu",
                                           layer_type="Dense",
                                           conditional_input=True),
                                   DotDict(layer_params={"units": 2},
                                           layer_type="Dense"),
                                   )
                     )
svae_params = DotDict(feature_extractor=fe_params_0,
                      laterals=[laterals_0, laterals_1, laterals_2],
                      input_shapes=((28, 28, 1), (10,))
                      )

svae = SymmetricVAE(**svae_params)
