import tensorboard 
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import Datamodule
from model import ConvMixerModule

import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="name of the model for recognise differents checkpoints")
parser.add_argument("--gpus", nargs="+", type=int, default=0, help="gpus devices")
parser.add_argument("--epochs", type=int, default=1, help="# epochs")
parser.add_argument("--resolution", type=int, default=224, help="# learning rate")
parser.add_argument("--data_path", type=str, default="./dataset/UCF101", help="#dataset path")
parser.add_argument("--gradient_accumulation", type=int, default= 1, help="#gradien accumulation")
parser.add_argument("--finetune", type=str, default="", help="#path to model from lear")
parser.add_argument("--n_classes", type=int, default=-1, help="#classes")
cli, _ = parser.parse_known_args()



if(cli.finetune=="convmixer"):
    model  = ConvMixerModule()
else:
    model= ConvMixerModule().load_from_checkpoint(cli.finetune)
if (cli.n_classes != -1):
    model.adapt_for_n_classes(cli.n_classes)
data= Datamodule(path=cli.data_path, resolution= cli.resolution)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename=f'{cli.name}'+'-{file}-{epoch}-{val_loss:.2f}-{other_metric:.2f}',
    dirpath=f'checkpoints/{cli.name}',
    save_last=True, 
    every_n_epochs=1, 
    save_top_k=5)

logger = TensorBoardLogger(
            "tb_logs", 
            name=f"{cli.name}")
lr_monitor = LearningRateMonitor(logging_interval='epoch')

if(type(cli.gpus) == int):
    trainer = pl.Trainer(
            gpus=cli.gpus, 
            max_epochs=cli.epochs, 
            callbacks=[lr_monitor],
            accumulate_grad_batches= cli.gradient_accumulation
            )
else:
    ddp= DDPStrategy(
        find_unused_parameters=False,
        process_group_backend="gloo"
        )
    trainer = pl.Trainer(
            max_epochs=cli.epochs, 
            auto_lr_find=False, 
            auto_scale_batch_size=False,
            gpus=cli.gpus, 
            precision=16,
            callbacks=[checkpoint_callback, lr_monitor], 
            logger=logger,
            strategy= ddp,
            accumulate_grad_batches= cli.gradient_accumulation
            )

trainer.fit(model, data )


