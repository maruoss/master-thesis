

import pdb


def create_foldername(model, dm, to_add: dict ={}, to_exclude: list = [], tag=''):
    """needed in train to convert to_add, model and dm parameters to string repr."""
    name = tag
    for k, v in to_add.items():
        if k not in to_exclude:
            name += k
            name += "_"
            name += str(v)
            name += "."
    for k, v in model.hparams.items():
        if k not in to_exclude:
            name += k
            name += "_"
            name += str(v)
            name += "."
    for k, v in dm.hparams.items():
        if k not in to_exclude:
            name += k
            name += "_"
            name += str(v)
            name += "."
    return name


def string_from_config(config: dict, tag=''):
    """needed in tune to convert search space directory to string repr."""
    name = tag
    name += "config{"
    for i, (k, v) in enumerate(config.items()):
        if i:
            name += "."
        name += k + "_"
        # name += "_"
        name += v.domain_str
    name += "}"

    return name