import copy
from dotdict import DotDict


def get_updated_params(params: dict,
                       default_params: dict) -> DotDict:

    """
    Update default parameters with new parameters.

    Parameters
    ----------
    params : dict
        given dictionary of parameter keys and parameter values
    default_params : dict
        given dictionary of default parameter keys and parameter values

    Returns
    -------
    DotDict
        dotted dictionary with updated parameter keys and values, i.e., as in default_params if not present in params
    """

    out_params = copy.deepcopy(default_params)
    out_params.update(params)

    # ----------------- return -----------------

    return DotDict(out_params)

