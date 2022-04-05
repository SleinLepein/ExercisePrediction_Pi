from dotdict import DotDict
from templates.keras_model_template.nn.defaults.standard_conv_params import standard_conv_params

# standard params in every iteration
standard_block_params = DotDict(
    layer_params=standard_conv_params,
    activation="relu",
    layer_type="Conv2D",
    normalization="batch_normalization",
    reshape=None,
    laterals=DotDict(weights=False, bn=False, act=False),
    skip=False,
    lateral_only=False,
    conditional_input=False,
    conditional_params=None,
    name_prefix="",
)
