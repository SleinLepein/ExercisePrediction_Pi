from dotdict import DotDict
from templates.keras_model_template.nn.defaults.standard_block_params import standard_block_params

# overall standard params
standard_params = DotDict(
    # definitions per block
    block_params=(standard_block_params,),

    # general definitions
    name_prefix="",
    final_sampling=False,
    interactions=False,
)