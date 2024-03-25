import pytorch_lightning as pl
import gc
from transformers import SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import FIVESDataset, OpenNeuroDataset
from omegaconf import DictConfig
import hydra
from evaluation.models.segformer import Segformer
from tabulate import tabulate
import os
from lightning.pytorch.loggers import TensorBoardLogger

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def evaluate_model(model, test_dataloader, device, reduction=None):
    ## TESTING
    logger = TensorBoardLogger(save_dir=model.filepath)
    trainer = pl.Trainer(devices=1, accelerator="gpu",logger=logger)

    trainer.test(model, test_dataloader)

    outputs = model.on_test_end()
    tp = torch.cat(outputs["tp"])
    fp = torch.cat(outputs["fp"])
    fn = torch.cat(outputs["fn"])
    tn = torch.cat(outputs["tn"])


    metrics = np.round(torch.stack([
                                    torch.mean(smp.metrics.accuracy(tp, fp, fn, tn, reduction=reduction), 0),
                                    torch.mean(smp.metrics.recall(tp, fp, fn, tn, reduction=reduction), 0),
                                    torch.mean(smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction), 0),
                                    torch.mean(smp.metrics.precision(tp, fp, fn, tn, reduction=reduction), 0),
                                    torch.mean(smp.metrics.false_positive_rate(tp, fp, fn, tn, reduction=reduction), 0),
                                    1 - torch.mean(smp.metrics.recall(tp, fp, fn, tn, reduction=reduction), 0),
                                    torch.mean(smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction), 0),
                                    ]).cpu().numpy(), 3)
    
    conf_matrix = {
            'TP': [tp.sum().item()],
            'FP': [fp.sum().item()],
            'FN': [fn.sum().item()],
            'TN': [tn.sum().item()],
            }

    # Annoying thing to add brackets as reduction makes it scalar
    info = {
            'Accuracy': [metrics[0]] if reduction else metrics[0],
            'Recall':   [metrics[1]] if reduction else metrics[1],
            'F1':       [metrics[2]] if reduction else metrics[2],
            'Precision':[metrics[3]] if reduction else metrics[3],
            'FPR':      [metrics[4]] if reduction else metrics[4],
            'FNR':      [metrics[5]] if reduction else metrics[5],
            'IoU':      [metrics[6]] if reduction else metrics[6],
            }
    
    matrix = tabulate(conf_matrix, headers='keys', tablefmt='fancy_grid')
    
    table = tabulate(info, headers='keys', tablefmt='fancy_grid')

    with open(os.path.join(model.filepath, 'confusion.txt'), 'w') as f:
        f.write(matrix)

    with open(os.path.join(model.filepath, 'results.txt'), 'w') as f:
        f.write(table)
    #print(table)


@hydra.main(config_path='../config', config_name='eval_cfg', version_base=None)
def main(cfg: DictConfig) -> None:
    # PARAMETERS
    #parameters test dataset
    test_batch_size = 16
    num_workers = 32

    num_classes = 1
    mode = "binary"

    #set CUDA_VISIBLE_DEVICES
    #torch.cuda.set_device(cfg.model.gpus)
    torch.set_float32_matmul_precision('medium')
    gc.collect()
    pl.seed_everything(seed=1234, workers=True)
    dev = torch.cuda.device(0)
    #torch.cuda.empty_cache()


    # DATALOADERS

    #load test dataset for evaluation; TEST DATASET 40 samples
    if cfg.data.dataset == 'openneuro':
        #OPENNEURO
        test_dataset = OpenNeuroDataset(root_dir='', extended=True) #TODO add path
    elif cfg.data.dataset == 'fives':
        #FIVES
        test_dataset = FIVESDataset(root_dir='/home/oliverbr/ninschmi/FIVES_GRAYS_train_val_split/test/', extended=True, eval=True, valid=True)
    test_dl = DataLoader(dataset=test_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    filepath = os.path.join('/home/oliverbr/ninschmi/evaluation_results/', cfg.data.dataset, 'nvidia_segformer_wo_pt_wo_ft')
    #check path
    os.makedirs(filepath, exist_ok=True)

    # MODELS
    segformers = []
    #load pretrained nvidia Segformer
    segformer_nvidia = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            num_channels=1,
            ignore_mismatched_sizes=True
        )
    
    optimizers = None
    segformers.append(Segformer(segformer_nvidia, optimizers, num_classes, mode, num_to_groups(len(test_dl.dataset), test_dl.batch_size), filepath))
    
    segformer_pretrained = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            num_channels=1,
            ignore_mismatched_sizes=True
        )                         
    segformer_pretrained = Segformer.load_from_checkpoint('/home/oliverbr/ninschmi/medicaldiffusion/checkpoints/evaluation/fives/pretrain/lightning_logs/version_0/checkpoints/val_loss=0.00-v1.ckpt', \
                            model=segformer_pretrained, optimizer=optimizers, num_classes=num_classes, mode=mode, batch_len=num_to_groups(len(test_dl.dataset), test_dl.batch_size), filepath=filepath.replace('nvidia_segformer_wo_pt_wo_ft','pretrained_segformer_w_pt_wo_ft'))
    segformers.append(segformer_pretrained)

    segformer_finetuned = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            num_channels=1,
            ignore_mismatched_sizes=True
        ) 
    
    segformer_finetuned = Segformer.load_from_checkpoint('/home/oliverbr/ninschmi/medicaldiffusion/checkpoints/evaluation/fives/finetune/lightning_logs/version_0/checkpoints/val_loss=0.00-v2.ckpt', \
                            model=segformer_finetuned, optimizer=optimizers, num_classes=num_classes, mode=mode, batch_len=num_to_groups(len(test_dl.dataset), test_dl.batch_size), filepath=filepath.replace('nvidia_segformer_wo_pt_wo_ft','finetuned_segformer_wo_pt_w_ft'))
    segformers.append(segformer_finetuned)

    segformer_pretrained_finetuned = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            num_channels=1,
            ignore_mismatched_sizes=True
        ) 
    
    segformer_pretrained_finetuned = Segformer.load_from_checkpoint('/home/oliverbr/ninschmi/medicaldiffusion/checkpoints/evaluation/fives/pretrain_finetune/lightning_logs/version_0/checkpoints/val_loss=0.00.ckpt', \
                                        model=segformer_pretrained_finetuned, optimizer=optimizers, num_classes=num_classes, mode=mode, batch_len=num_to_groups(len(test_dl.dataset), test_dl.batch_size), filepath=filepath.replace('nvidia_segformer_wo_pt_wo_ft','pretrained_finetuned_segformer_w_pt_w_ft'))

    segformers.append(segformer_pretrained_finetuned)

    # loop trough all fine-tuned models w/wo pretraining
    segformers_finetuned = []
   
    # loop that runs test set (40 samples) on all Segformers (vanilla, pre-trained, fine-tuned w/wo pre-training)
    for segformer in segformers:
        evaluate_model(segformer, test_dl, device=dev, reduction=None)
         #TODO if possible do this for multiple checkpoints


if __name__ == '__main__':
    main()