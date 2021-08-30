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


def is_number(s):
    try:
        float(s)
        return True
    except Exception as e:
        return False


def get_current_run_override_config(orig_dict, test_num, num_tests):
    # If this is still a dictionary, we need to go deeper for each new argument
    if type(orig_dict) == dict:
        single_dict = {}
        for k, v in orig_dict.items():
            single_dict[k] = get_current_run_override_config(v, test_num, num_tests)
        return single_dict

    elif type(orig_dict) == list:
        # Assume [min, max] if ints
        if len(orig_dict) == 1:
            return orig_dict[0]

        if len(orig_dict) == 2 and is_number(orig_dict[0]) and is_number(orig_dict[1]):
            delta = orig_dict[1] - orig_dict[0]
            return orig_dict[0] + delta * (test_num / (num_tests - 1))

        if len(orig_dict) == num_tests:
            return orig_dict[test_num]

        else:
            return orig_dict

    # In this case, just assume constant value
    else:
        return orig_dict
