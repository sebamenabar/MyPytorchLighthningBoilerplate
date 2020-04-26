from __future__ import division
from __future__ import print_function

import os
import errno
from itertools import chain, starmap
import functools
import yaml
import json
import numpy as np
from easydict import EasyDict as edict


__C = edict()
__C.cfg_file = ("", edict(help="optional config file", type=str))
__C.gpus = (
    "0",
    edict(
        help="if single number number of gpus to use (-1 for all available), if comma-separated-ints the indices of gpus to use",
        type=str,
        default=0,
    ),
)
__C.num_workers = (0, edict(help="number of dataloader workers", type=int, default=0))
__C.random_seed = (None, edict(type=int, help="random seed, none by default"))
# __C.logdir = (
#     "",
#     edict(help="folder inside of experiments were the run will be saved", type=str),
# )
__C.logtb = (False, edict(help="log on tensorboard", action="store_true"))
__C.logcomet = (False, edict(help="log on comet_ml", action="store_true"))
__C.run_name = (
    "",
    edict(
        help="name of the run of this experiment, will use current datetime as default, experiment will be saved on `experiments/<exp_dir>/<run_name>`",
        type=str,
    ),
)
__C.exp_name = (
    "",
    edict(
        help="experiment name, if empty the run will be saved directly inside of `experiments`", type=str
    ),
)
__C.comet_project_name = ("", edict(type=str))
# __C.cuda = False
__C.eval = (False, edict(help="run evaluation only", action="store_true"))
__C.test = (False, edict(help="run testing onyl", action="store_true"))
__C.checkpoint_path = (None, edict(help="checkpoint path", type=str))

__C.train = edict()
__C.train.bsz = (64, edict(help="train batch size", type=int, default=64))
__C.train.epochs = (10, edict(help="train max number of epochs", type=int, default=10))
__C.train.lr = (1e-4, edict(help="optimizer lr", type=float, default=1e-4))

__C.train.bsz = (64, edict(help="val batch size", type=int, default=64))

# __C.model = edict()
def _to_values_only(values, idx):
    if isinstance(values, edict):
        return edict(**{k: _to_values_only(v, idx) for k, v in values.items()})
    else:
        return values[idx]


cfg = _to_values_only(__C, 0)


def cfg_to_parser_args():
    parser_args = edict()
    for k, v in flatten_json_iterative_solution(_to_values_only(__C, 1)).items():
        k1, _, arg_name = k.rpartition(".")
        if k1 not in parser_args:
            parser_args[k1] = {}
        parser_args[k1][arg_name] = v
    return parser_args


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("{} is not a valid config key".format(k))
            # print("{} is not a valid config key".format(k))
            # continue

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif isinstance(b[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif b[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(
                    ("Type mismatch ({} vs. {}) " "for config key: {}").format(
                        type(b[k]), type(v), k
                    )
                )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    if filename:
        with open(filename, "r") as f:
            _, ext = os.path.splitext(filename)
            if ext == ".yml" or ext == ".yaml":
                file_cfg = edict(yaml.safe_load(f))
            elif ext == ".json":
                file_cfg = edict(json.load(f))

        _merge_a_into_b(file_cfg, __C)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def flatten_json_iterative_solution(dictionary):
    """Flatten a nested json file"""

    def unpack(parent_key, parent_value):
        """Unpack one level of nesting in json file"""
        # Unpack one level only!!!

        if isinstance(parent_value, dict):
            for key, value in parent_value.items():
                temp1 = parent_key + "." + key
                yield temp1, value
        elif isinstance(parent_value, list):
            i = 0
            for value in parent_value:
                temp2 = parent_key + "." + str(i)
                i += 1
                yield temp2, value
        else:
            yield parent_key, parent_value

    # Keep iterating until the termination condition is satisfied
    while True:
        # Keep unpacking the json file until all values are atomic elements (not dictionary or list)
        dictionary = dict(chain.from_iterable(starmap(unpack, dictionary.items())))
        # Terminate condition: not any value in the json file is dictionary or list
        if not any(
            isinstance(value, dict) for value in dictionary.values()
        ) and not any(isinstance(value, list) for value in dictionary.values()):
            break

    return dictionary


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))
