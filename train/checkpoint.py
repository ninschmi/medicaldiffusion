"Adapted from pytorch-lightning: https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/trainer/connectors/checkpoint_connector.py"

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
from torch import Tensor

_PATH = Union[str, Path]


class Checkpoint:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizers: List
        ) -> None:
        self._ckpt_path: Optional[_PATH] = None
        self._loaded_checkpoint: Dict[str, Any] = {}
        self.model = model
        self.optimizers = optimizers

    def resume_start(self, checkpoint_path: Optional[_PATH] = None) -> None:
        self._ckpt_path = checkpoint_path
        if not checkpoint_path:
            print("`checkpoint_path` not specified. Skipping checkpoint loading.")
            return

        #rank_zero_info()
        print(f"Restoring states from the checkpoint path at {checkpoint_path}")
        torch.cuda.empty_cache()
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        self._loaded_checkpoint = torch.load(
                checkpoint_path,
                map_location=lambda storage, loc: storage,  # type: ignore[arg-type] # upstream annotation is not correct
            )

    def resume_end(self) -> None:
        #assert self.trainer.state.fn is not None
        if self._ckpt_path:
            message = "Restored all states or loaded model weights"
            #rank_zero_info(
            print(f"{message} from the checkpoint at {self._ckpt_path}")

        # free memory
        self._loaded_checkpoint = {}
        torch.cuda.empty_cache()

        # wait for all to catch up
        #self.trainer.strategy.barrier("_CheckpointConnector.resume_end")

    #def restore(self, checkpoint_path: Optional[_PATH] = None) -> None:
    #    """Attempt to restore everything at once from a 'PyTorch-Lightning checkpoint' file through file-read and
    #    state-restore, in this priority:
#
    #    1. from HPC weights if found
    #    2. from `checkpoint_path` file if provided
    #    3. don't restore
#
    #    All restored states are listed in return value description of `dump_checkpoint`.
#
    #    Args:
    #        checkpoint_path: Path to a PyTorch Lightning checkpoint file.
#
    #    """
    #    self.resume_start(checkpoint_path)
#
    #    # restore module states
    #    self.restore_datamodule()
    #    self.restore_model()
#
    #    # restore callback states
    #    self.restore_callbacks()
#
    #    # restore training state
    #    self.restore_training_state()
    #    self.resume_end()
#

    def restore_model(self) -> None:
        """Restores a model's weights from a PyTorch Lightning checkpoint.
        """
        if not self._loaded_checkpoint:
            return

        #TODO update batches that stepped when training is resumed from checkpoint
        # self._batches_that_stepped = state_dict.get("_batches_that_stepped", 0)

        # restore model state_dict
        assert self.model is not None
        self.model.load_state_dict(self._loaded_checkpoint["state_dict"])

    def restore_training_state(self) -> None:
        if not self._loaded_checkpoint:
            return

        #assert self.trainer.state.fn is not None
        # restore optimizers (depends on strategy deployed, as originally optimizers are only restored if #if self.trainer.strategy.lightning_restore_optimizer:)
        # validation
        if "optimizer_states" not in self._loaded_checkpoint:
            raise KeyError(
                "Trying to restore optimizer state but checkpoint contains only the model."
                " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
            )
        # restore the optimizers
        optimizer_states = self._loaded_checkpoint["optimizer_states"]
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)

    #def restore_callbacks(self) -> None:
    #    """Restores all callbacks from the pre-loaded checkpoint."""
    #    if not self._loaded_checkpoint:
    #        return
#
    #    trainer = self.trainer
    #    call._call_callbacks_on_load_checkpoint(trainer, self._loaded_checkpoint)
    #    call._call_callbacks_load_state_dict(trainer, self._loaded_checkpoint)
#

    def _restore_modules_and_callbacks(self, checkpoint_path: Optional[_PATH] = None) -> None:
        # restore modules after setup
        self.resume_start(checkpoint_path)
        self.restore_model()
        #TODO: restore callback states
        #self.restore_callbacks()
    
    def dump_checkpoint(self, global_step: int, current_epoch: int, weights_only: bool = False) -> dict:
        """Creating a model checkpoint dictionary object from various component states.

        Args:
            weights_only: saving model weights only
        Return:
            structured dictionary: {
                'epoch':                     training epoch
                'global_step':               training global step
                'callbacks':                 "callback specific state"[] # if not weights_only
                'optimizer_states':          "PT optim's state_dict"[]   # if not weights_only
                'state_dict':                Model's state_dict (e.g. network weights)
                CHECKPOINT_HYPER_PARAMS_NAME:
                CHECKPOINT_HYPER_PARAMS_KEY:
                CHECKPOINT_HYPER_PARAMS_TYPE:
            }

        """
        model = self.model

        checkpoint = {
            # the epoch and global step are saved for compatibility but they are not relevant for restoration
            "epoch": current_epoch,
            "global_step": global_step,
            "state_dict": model.state_dict(),
            #"loops": self._get_loops_state_dict(),
        }

        if not weights_only:
            ## dump callbacks
            #checkpoint["callbacks"] = call._call_callbacks_state_dict(trainer)

            optimizer_states = []
            for i, optimizer in enumerate(self.optimizers):
                # Rely on accelerator to dump optimizer state
                optimizer_state = optimizer.state_dict()
                optimizer_states.append(optimizer_state)

            checkpoint["optimizer_states"] = optimizer_states
            
        #from omegaconf import Container
        ## dump hyper-parameters
        #if model and model.hparams:
        #    if hasattr(model, "_hparams_name"):
        #        checkpoint[model.CHECKPOINT_HYPER_PARAMS_NAME] = model._hparams_name
        #    # dump arguments
        #    if _OMEGACONF_AVAILABLE and isinstance(model.hparams, Container):
        #        checkpoint[model.CHECKPOINT_HYPER_PARAMS_KEY] = model.hparams
        #        checkpoint[model.CHECKPOINT_HYPER_PARAMS_TYPE] = type(model.hparams)
        #    else:
        #        checkpoint[model.CHECKPOINT_HYPER_PARAMS_KEY] = dict(model.hparams)
        return checkpoint
