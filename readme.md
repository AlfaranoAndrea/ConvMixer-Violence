# CNN-transformer based architecture for Violence Detection: a novel approach using ConvMixer

## main files:
* prepare_dataset.ipynb: a multiprocess implementation for extract frames from dataset and composing them in the super picture configuration 
* dataset.py: contains the Dataset class and the Datamodule
* train.py: contain all the training pipeline, powered by pytorch lightning
* evaluation.py: contain all the evaluation pipeline, powered by pytorch lighting
* config.json: contain the main parameters for the implementation

# model training

## step 1 - make the dataset:
1. download the desired dataset, extract and place it into the Dataset folder
2. on the prepare_dataset.ipynb set source_path and output_path according to the dataset path and execute the notebook

## step 2 - download the convmixer pretrained model:
1. download the desired pretrained Convmixer on imagenet1k (we used convmixer_1024_20_ks9_p14) and place it into checkpoint_convmixer folder

## step 3 - pretrain the model:
1. run train.py with the following comand: python3 train.py --gpus 0 1 --epochs 20 --name 3x3_res224_batchsize144 --gradient_accumulation 3 --finetune convmixer --n_classes 101

## step 4 - test the pretrained model:
1. run evaluation.py giving with --finetune parameter the path to the model. 
For example the following comand: python3 evaluation.py --finetune ./checkpoints/3x3_res224_batchsize144-file=0-epoch=17-val_loss=0.32-other_metric=0.00.ckpt  --data_path ./dataset/UCF101
 
## step 5 - final trainin on the target dataset:
1. run train.py with the following comand: python3 train.py --gpus 0 1 --epochs 20 --name Finetuned_3x3_res224_batchsize144 --gradient_accumulation 3 --finetune ./checkpoints/3x3_res224_batchsize144-file=0-epoch=17-val_loss=0.32-other_metric=0.00.ckpt --n_classes 2 --data_path ./dataset/RealLifeViolence

## step 6 - test the pretrained model:
1. run evaluation.py giving with --finetune parameter the path to the model. 
For example the following comand: python3 evaluation.py --finetune ./checkpoints/3x3_res224_batchsize144-file=0-epoch=17-val_loss=0.32-other_metric=0.00.ckpt --n_classes 2 --data_path ./dataset/RealLifeViolence
