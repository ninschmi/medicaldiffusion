from typing import Any
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torchvision import transforms as T
import os
import numpy as np

class Segformer(pl.LightningModule):
    def __init__(self, model, optimizer, num_classes, mode, batch_len=None, filepath='') -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.mode = mode
        self.tp = []
        self.fp = []
        self.fn = []
        self.tn = []
        self.batch_len = batch_len
        self.filepath = filepath
        self.log_images = False
        
    def forward(self, images, masks):
        return self.model(images, masks)
    
    def shared_step(self, batch, stage):  
        image, mask = batch['data'], batch['target']

        image = image.float()
        mask = mask.long()
        
        out = self.forward(image, mask.squeeze(dim=1))
        loss, logits = out[0], out[1]

        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=mask.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        binarized = (upsampled_logits.sigmoid() > 0.5).long()
        
        tp, fp, fn, tn = smp.metrics.get_stats(binarized, mask, mode="binary")
        
        log_dict = {f"{stage}_loss" : loss}
        
        # averaged metrics
        for reduction in ["micro", "micro-imagewise"]:
            log_dict.update({
                f"{stage}_{reduction}_iou" : smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction),
                f"{stage}_{reduction}_accuracy" : smp.metrics.accuracy(tp, fp, fn, tn, reduction=reduction),
                f"{stage}_{reduction}_recall" : smp.metrics.recall(tp, fp, fn, tn, reduction=reduction),
                f"{stage}_{reduction}_precision" : smp.metrics.precision(tp, fp, fn, tn, reduction=reduction),
                f"{stage}_{reduction}_F1" : smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction),
                f"{stage}_{reduction}_FPR" : smp.metrics.false_positive_rate(tp, fp, fn, tn, reduction=reduction), 
                f"{stage}_{reduction}_FNR" : 1 - torch.mean(smp.metrics.recall(tp, fp, fn, tn, reduction=reduction), 0),
            })
            
        self.log_dict(log_dict, prog_bar=True, on_epoch=True)

        if stage == "test":
            return {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "images": image,
                "masks": mask,
                "predictions": binarized,
            }
        elif stage == "valid":
            return {
                "loss": loss,
                "images": image,
                "masks": mask,
                "predictions": binarized,
            }
    
        return {
            "loss": loss,
        }
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")     

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, "valid")

        if (batch_idx == 0) and self.log_images:
        
            image = output["images"][1] * 255
            image = T.ToPILImage(mode="L")(image.to(torch.uint8))
            image.save(os.path.join(self.filepath,'image_' + str(self.current_epoch) + '.png'))

            mask = output["masks"][1].to(torch.uint8) * 255
            mask = T.ToPILImage(mode="L")(mask)
            mask.save(os.path.join(self.filepath, 'mask_' + str(self.current_epoch) + '.png'))

            pred = output["predictions"][1].to(torch.uint8) * 255
            pred = T.ToPILImage(mode="L")(pred.to(torch.uint8))
            pred.save(os.path.join(self.filepath, 'bin_mask_' + str(self.current_epoch) + '.png'))
        return output
    
    def test_step(self, batch, batch_idx):
        output = self.shared_step(batch, "test")

        self.tp.append(output["tp"])
        self.fp.append(output["fp"])
        self.fn.append(output["fn"])
        self.tn.append(output["tn"])

        #save test images, masks, predictions

        images = output["images"]
        masks = output["masks"]
        predictions = output["predictions"]

        for i in range((images.shape[0])):
            sample_nr = i+self.batch_len[batch_idx-1] if batch_idx > 0 else i
            image = images[i] * 255
            image = T.ToPILImage(mode="L")(image.to(torch.uint8))
            image.save(os.path.join(self.filepath,'image_' + str(sample_nr) + '.png'))

            mask = masks[i].to(torch.uint8) * 255
            mask = T.ToPILImage(mode="L")(mask)
            mask.save(os.path.join(self.filepath, 'mask_' + str(sample_nr) + '.png'))
            
            pred = predictions[i].to(torch.uint8) * 255
            pred = T.ToPILImage(mode="L")(pred.to(torch.uint8))
            pred.save(os.path.join(self.filepath, 'bin_mask_' + str(sample_nr) + '.png'))
            
        return output
    
    def on_train_start(self) -> None:
        self.filepath = os.path.join(self.filepath, 'images/')
        os.makedirs(self.filepath, exist_ok=True)
        self.log_images = True
    
    def on_test_start(self) -> None:
        # reset metrics
        self.tp = []
        self.fp = []
        self.fn = []
        self.tn = []
    
    def on_test_end(self) -> None:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }
    
    def configure_optimizers(self):
        return self.optimizer