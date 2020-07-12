import torch
from pytorch_lightning.utilities import parsing
from plb_base.base_config import __C, parse_args_and_set_config, edict, _to_values_only

parse_bool = lambda x: bool(parsing.strtobool(x))
cfg = _to_values_only(__C, 0)
