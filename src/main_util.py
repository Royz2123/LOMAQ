import yaml
import os
import collections
from copy import deepcopy


def get_param(params, arg_name):
    param_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            param_name = _v.split("=")[1]
            del params[_i]
            break
    return param_name


def get_config_from_params(params, arg_name, subfolder):
    config_name = get_param(params, arg_name)
    return get_config_dict(config_name, subfolder)


def get_config_dict(config_name, subfolder=None):
    if subfolder is None:
        path = os.path.join(os.path.dirname(__file__), "config", "{}.yaml".format(config_name))
    else:
        path = os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name))

    config_dict = None
    try:
        with open(path, "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                print("{}.yaml error: {}".format(config_name, exc))
    except Exception as e:
        print("Had some problem opening the config file: {}, {}".format(path, e))
    return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)
