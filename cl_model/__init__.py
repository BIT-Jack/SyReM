import os
import importlib
from utils.args_loading import root_dir

def get_all_models():
    return [model.split('.')[0] for model in os.listdir(root_dir + 'cl_model')
            if not model.find('__') > -1 and 'py' in model]

names = {}
for model in get_all_models():
    mod = importlib.import_module('cl_model.' + model)
    class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)

def get_model(args, backbone, loss):
    return names[args.model](backbone, loss, args)
