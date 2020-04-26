import os
import os.path as osp
import sys
import shutil
import random
from dateutil import tz
import pytorch_lightning as pl
from datetime import datetime as dt

from config import cfg, mkdir_p


class BasePLModel(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()

        # self.net = Net()
        self.hparams = hparams
        self.cfg = hparams
        # if hparams.resume:
        #     self.load_from_checkpoint(hparams.resume)
        self.cfg = hparams

    # I'm just more used to doing self.cfg than self.hparams
    # and PL automatically logs self.hparams attribute
    # @property
    # def cfg(self):
    #     return self.hparams

    def init_log(self):
        now = dt.now(tz.tzlocal())
        now = now.strftime("%m-%d-%Y-%H-%M-%S")
        log_dir = self.cfg.exp_name
        run_name = self.cfg.run_name
        if run_name == "" or run_name is None:
            run_name = now
        work_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
        exp_dir = osp.join(work_dir, "experiments", log_dir, run_name)
        _exp_dir = exp_dir
        i = 2
        while os.path.exists(_exp_dir):
            _exp_dir = f"{exp_dir}-{i}"
            i += 1
        exp_dir = _exp_dir

        self.exp_dir = exp_dir
        self.checkpoints_dir = osp.join(exp_dir, "checkpoints")
        mkdir_p(self.checkpoints_dir)
        shutil.copytree(
            osp.join(work_dir, "src"),
            osp.join(exp_dir, "src"),
            ignore=shutil.ignore_patterns(".*", "__pycache__", ".DS_Store"),
        )

        # print(exp_dir)

    def forward(self):
        pass
