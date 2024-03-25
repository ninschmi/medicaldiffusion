from typing import Any
import gc
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation
from datasets import load_metric
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from torch.utils.data import DataLoader, ConcatDataset
from lightning.pytorch.loggers import TensorBoardLogger
from dataset import FIVESDataset, OpenNeuroDataset
import hydra
from omegaconf import DictConfig
from evaluation.models.segformer import Segformer
    
@hydra.main(config_path='../config', config_name='eval_cfg', version_base=None)
def main(cfg: DictConfig) -> None:
    # PARAMETERS
    #parameters test dataset
    batch_size = 32
    num_workers = 32

    num_classes = 1
    mode = "binary"

    #training parameters
    max_epochs = 2000
    learning_rate = 0.00006 # Paper

    #set CUDA_VISIBLE_DEVICES
    #torch.cuda.set_device(cfg.model.gpus)
    torch.set_float32_matmul_precision('medium')
    gc.collect()
    pl.seed_everything(seed=1234, workers=True)
    dev = torch.cuda.device(0)
    torch.cuda.empty_cache()


    # DATALOADERS
    #load test train and val dataset for fine-tuning; TEST DATASET 160 samples
    if cfg.data.dataset == 'openneuro':
        #OPENNEURO
        test_dataset = OpenNeuroDataset(root_dir='', extended=True) #TODO add path
    elif cfg.data.dataset == 'fives':
        #FIVES
        test_train_dataset = FIVESDataset(root_dir='/home/oliverbr/ninschmi/FIVES_GRAYS_train_val_split/test/train/', extended=True, eval=True)
        test_val_dataset = FIVESDataset(root_dir='/home/oliverbr/ninschmi/FIVES_GRAYS_train_val_split/test/train/valid/', extended=True, eval=True, valid=True)
        #test_train_dataset = FIVESDataset(root_dir='/home/oliverbr/ninschmi/FIVES_GRAYS_all_train/train/', extended=True, eval=True)
        #test_val_dataset = FIVESDataset(root_dir='/home/oliverbr/ninschmi/FIVES_GRAYS_all_train/val', extended=True, eval=True, valid=True)
    test_train_dl = DataLoader(dataset=test_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    test_val_dl = DataLoader(dataset=test_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    filepath = '/home/oliverbr/ninschmi/medicaldiffusion/checkpoints/evaluation/' + cfg.data.dataset + '/pretrain_finetune/'
    #filepath = '/home/oliverbr/ninschmi/medicaldiffusion/checkpoints/evaluation/' + cfg.data.dataset + '/finetune_all_data/'
    
    #MODEL

    #load checkpoint that is to be fine-tuned
    #load pretrained nvidia Segformer
    segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            num_channels=1,
            ignore_mismatched_sizes=True
        )
     
    #fine-tune this on just the TEST TRAIN/VAL data -> finetuned Segformer
    optimizer = torch.optim.Adam(segformer.parameters(), lr=learning_rate)
    #without pretraining
    segformer_finetuned = Segformer(segformer, optimizer, num_classes, mode, filepath=filepath)
    #with pretraining
    #segformer_finetuned = Segformer.load_from_checkpoint('/home/oliverbr/ninschmi/medicaldiffusion/checkpoints/evaluation/fives/pretrain/lightning_logs/version_0/checkpoints/val_loss=0.00-v1.ckpt', \
    #                        model=segformer, optimizer=optimizer, num_classes=num_classes, mode=mode, filepath=filepath)

    logger = TensorBoardLogger(save_dir=filepath)

    cbs = pl.callbacks.ModelCheckpoint(filename = '{val_loss:.2f}',
                                verbose = True, 
                                monitor = 'valid_loss', 
                                mode = 'min',
                                save_top_k = 3)

    logger.log_hyperparams({
    "batch_size": batch_size,
    "num_workers": num_workers,
    "max_epochs": max_epochs,
    "learning_rate": learning_rate,
    })

    #train segformer_finetuned
    trainer = pl.Trainer(callbacks=cbs, devices=1, accelerator='gpu', max_epochs=max_epochs, logger=logger, log_every_n_steps=1)
    trainer.fit(segformer_finetuned, test_train_dl, test_val_dl)


if __name__ == '__main__':
    main()