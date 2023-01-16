import pytorch_lightning as pl

from dataset import Datamodule
from model import ConvMixerModule


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=224, help="# learning rate")
parser.add_argument("--data_path", type=str, default="./RealLifeViolence", help="#dataset path")

parser.add_argument("--finetune", type=str, default="", help="#path to model from lear")

cli, _ = parser.parse_known_args()

model= ConvMixerModule().load_from_checkpoint(cli.finetune)
data= Datamodule(path=cli.data_path, resolution= cli.resolution)

trainer = pl.Trainer(
        gpus=1, 
        )

trainer.test(model, data )