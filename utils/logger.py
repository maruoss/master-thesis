

def log_foldername(model, dm, to_add: dict ={}, to_exclude: list = [], tag=''):
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
