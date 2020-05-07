A template of everything I use when developing Pytorch NNs

Filestructure
```
    -cfg
        - config_file1.yml
        - ...
    - src
        - config.py
        - base_pl_model.py
        - main.py
    - experiments
        - experiment_name1
            - run_name1
                - cfg.json
                - cfg.yml
                - logfile.log
                - checkpoints
                    - _ckpt_epoch_n.ckpt
                - src
                    - {copy of your src dir}
                - version_0 (default name, haven't changed it yet)
                    - tensorboard_logfile
                    - meta_tags.csv (config in .csv format)
            
            - run_name2
            - ...
        - experiment_name2
        - ...
```

* By default I use tensorboard as logger, you can enable comet by running with the flag `--logcomet`
* Images need to be logged manually. If your model inherited from BasePLModel you can use self.log_figure
* Example of how to use (main.py)

```python
from easydict import EasyDict as edict
from dotenv import load_dotenv

import torch
import torch.nn as nn
import pytorch_lightning as pl
from base_pl_model import BasePLModel
from config import __C, parse_args_and_set_config

### EXAMPLE SPECIFIC
import os
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

load_dotenv()

__C.model = edict(
    in_dim=(784, edict(help="input dimension", type=int)),
    hidden_dim=(1000, edict(help="hidden dim", type=int)),
    out_dim=(10, edict(help="output dimension", type=int)),
)


class Model(BasePLModel):
    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        model_cfg = self.cfg.model
        self.layers = nn.Sequential(
            nn.Linear(784, model_cfg.in_dim, model_cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(784, model_cfg.hidden_dim, model_cfg.out_dim),
        )
        self.loss = nn.CrossEntropyLoss()

    def train_dataloader(self):
        dataset = MNIST(
            os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
        )
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.bsz,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self.use_cuda,
        )
        return loader

    def val_dataloader(self):
        dataset = MNIST(
            os.getcwd(), train=False, download=True, transform=transforms.ToTensor()
        )
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.val_bsz,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=self.use_cuda,
        )
        return loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.layers.parameters(), lr=1e-2,)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.layers(x.flatten(1))
        loss = self.loss(y_hat, y)
        return {
            "loss": loss,
            "progress_bar": {"batch_nb": batch_nb},
            "log": {"batch_nb": batch_nb},
        }

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.layers(x.flatten(1))
        loss = self.loss(y_hat, y)
        return {
            "loss": loss,
            "progress_bar": {"batch_nb": batch_nb},
            "log": {"batch_nb": batch_nb},
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([o["loss"] for o in outputs]).mean()
        log = {"val_loss": val_loss}
        return {"val_loss": val_loss, "log": log, "progress_bar": log}


if __name__ == "__main__":
    args, cfg = parse_args_and_set_config(__C)
    print(cfg)

    model = Model(cfg)
    model.init_log()
    loggers, ckpt_callback = model.make_lightning_loggers_ckpt()
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=loggers,
        checkpoint_callback=ckpt_callback,
        max_epochs=cfg.train.epochs,
        default_root_dir=model.exp_dir,
    )
    if args.eval:
        pass
    elif args.test:
        pass
    else:
        pass
        trainer.fit(model)
```

And run with command `python3 src/main.py --cfg_file cfg/base_config.yml  --exp_name sample --run_name run1 --train.epochs 2 --progress_bar_refresh_rate 10 --overfit_pct 0.1`

A logfolder with checkpoints and tensorboard log with be automatically created. A config file is not required, you could also run `python3 src/main.py --model.hidden_dim 200  --exp_name sample --run_name run1 --train.epochs 2 --progress_bar_refresh_rate 10 --overfit_pct 0.1`