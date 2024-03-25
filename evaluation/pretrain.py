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
    #load synthetic train and val dataset for pre-training; SYNTHETIC DATASET 2000 samples
    if cfg.data.dataset == 'openneuro':
        #OPENNEURO
        test_dataset = OpenNeuroDataset(root_dir='', extended=True) #TODO add path
    elif cfg.data.dataset == 'fives':
        #FIVES
        test_train_dataset = FIVESDataset(root_dir='/home/oliverbr/ninschmi/synthetic_data/fives/final_extended_2d_vq_gan_wo_tanh_faulty_param/train/', extended=True, eval=True, synthetic=True)
        test_val_dataset = FIVESDataset(root_dir='/home/oliverbr/ninschmi/synthetic_data/fives/final_extended_2d_vq_gan_wo_tanh_faulty_param/valid', extended=True, eval=True, valid=True, synthetic=True)
    syn_train_dl = DataLoader(dataset=test_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    syn_val_dl = DataLoader(dataset=test_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    filepath = '/home/oliverbr/ninschmi/medicaldiffusion/checkpoints/evaluation/' + cfg.data.dataset + '/pretrain/'

    #MODEL

    #load pretrained nvidia Segformer
    segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            num_channels=1,
            ignore_mismatched_sizes=True
        )
    
    #pretrain this on just the synthetic data -> pretrained Segformer
    #optimizer = AdamW(model.parameters(), lr=0.00006)
    optimizer = torch.optim.Adam(segformer.parameters(), lr=learning_rate)
    segformer_pretrained = Segformer(segformer, optimizer, num_classes, mode, filepath=filepath)

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
    
    #train segformer_pretrained
    trainer = pl.Trainer(callbacks=cbs, devices=1, accelerator='gpu', max_epochs=max_epochs, logger=logger, log_every_n_steps=1)
    trainer.fit(segformer_pretrained, syn_train_dl, syn_val_dl)


if __name__ == '__main__':
    main()