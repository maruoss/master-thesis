

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


# def string_from_config(config: dict, tag=''):
#     """needed in tune to convert search space directory to string repr."""
#     name = tag
#     name += "config{"
#     for i, (k, v) in enumerate(config.items()):
#         if i:
#             name += "."
#         name += k + "_"
#         # name += "_"
#         if isinstance(v, dict): # for params with gridsearch -> dict of list
#             for kk, vv in v.items():
#                 name += kk
#                 name += "_["
#                 for j, i in enumerate(vv): # vv is a list
#                     if j:
#                         name += ","
#                     name += str(i)
#                 name += "]"
#         else:
#             name += v.domain_str
#     name += "}"

#     return name


def json_from_config(config: dict, tag=''):
    """needed in tune to convert search space directory to serializable dictionary"""
    dict_to_serialize = {}
    for k, v in config.items():
        if isinstance(v, dict): # for params with gridsearch -> dict of list
            dict_to_serialize[k] = v
        else:
            dict_to_serialize[k] = v.domain_str

    return dict_to_serialize


def params_to_dict(model, dm, to_add: dict ={}, to_exclude: list = [], tag=''):
    """needed in train to save model params in dictionary"""
    dic = {}
    for k, v in to_add.items():
        if k not in to_exclude:
            dic[k] = v
    for k, v in model.hparams.items():
        if k not in to_exclude:
            dic[k] = v
    for k, v in dm.hparams.items():
        if k not in to_exclude:
            dic[k] = v

    return dic