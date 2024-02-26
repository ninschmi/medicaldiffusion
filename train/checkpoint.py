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

    #def resume_start(self, checkpoint_path: Optional[_PATH] = None) -> None:
    #    """Attempts to pre-load the checkpoint file to memory, with the source path determined in this priority:
#
    #    1. from HPC weights if `checkpoint_path` is ``None`` and on SLURM or passed keyword `"hpc"`.
    #    2. from fault-tolerant auto-saved checkpoint if found
    #    3. from `checkpoint_path` file if provided
    #    4. don't restore
#
    #    """
    #    self._ckpt_path = checkpoint_path
    #    if not checkpoint_path:
    #        print("`checkpoint_path` not specified. Skipping checkpoint loading.")
    #        return
#
    #    #rank_zero_info()
    #    print(f"Restoring states from the checkpoint path at {checkpoint_path}")
    #    loaded_checkpoint = self.trainer.strategy.load_checkpoint(checkpoint_path)
    #    self._loaded_checkpoint = loaded_checkpoint
#
    #def resume_end(self) -> None:
    #    """Signal the connector that all states have resumed and memory for the checkpoint object can be released."""
    #    #assert self.trainer.state.fn is not None
    #    if self._ckpt_path:
    #        message = "Restored all states or loaded model weights"
    #        #rank_zero_info(
    #        print(f"{message} from the checkpoint at {self._ckpt_path}")
#
    #    # free memory
    #    self._loaded_checkpoint = {}
    #    torch.cuda.empty_cache()
#
    #    # wait for all to catch up
    #    #self.trainer.strategy.barrier("_CheckpointConnector.resume_end")
#
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
    #def restore_datamodule(self) -> None:
    #    """Calls hooks on the datamodule to give it a chance to restore its state from the checkpoint."""
    #    if not self._loaded_checkpoint:
    #        return
#
    #    trainer = self.trainer
    #    datamodule = trainer.datamodule
    #    if datamodule is not None and datamodule.__class__.__qualname__ in self._loaded_checkpoint:
    #        call._call_lightning_datamodule_hook(
    #            trainer, "load_state_dict", self._loaded_checkpoint[datamodule.__class__.__qualname__]
    #        )
#
    #def restore_model(self) -> None:
    #    """Restores a model's weights from a PyTorch Lightning checkpoint.
#
    #    Hooks are called first to give the LightningModule a chance to modify the contents, then finally the model gets
    #    updated with the loaded weights.
#
    #    """
    #    if not self._loaded_checkpoint:
    #        return
#
    #    trainer = self.trainer
    #    # hook: give user access to checkpoint if needed.
    #    call._call_lightning_module_hook(trainer, "on_load_checkpoint", self._loaded_checkpoint)
#
    #    # restore model state_dict
    #    trainer.strategy.load_model_state_dict(self._loaded_checkpoint)
#
    #def restore_training_state(self) -> None:
    #    """Restore the trainer state from the pre-loaded checkpoint.
#
    #    This includes the precision settings, loop progress, optimizer states and learning rate scheduler states.
#
    #    """
    #    if not self._loaded_checkpoint:
    #        return
#
    #    # restore precision plugin (scaler etc.)
    #    self.restore_precision_plugin_state()
#
    #    # restore loops and their progress
    #    self.restore_loops()
#
    #    assert self.trainer.state.fn is not None
    #    if self.trainer.state.fn == TrainerFn.FITTING:
    #        # restore optimizers and schedulers state
    #        self.restore_optimizers_and_schedulers()
#
    #def restore_precision_plugin_state(self) -> None:
    #    """Restore the precision plugin state from the pre-loaded checkpoint."""
    #    prec_plugin = self.trainer.precision_plugin
    #    prec_plugin.on_load_checkpoint(self._loaded_checkpoint)
    #    if prec_plugin.__class__.__qualname__ in self._loaded_checkpoint:
    #        prec_plugin.load_state_dict(self._loaded_checkpoint[prec_plugin.__class__.__qualname__])
#
    #    # old checkpoints compatibility
    #    if "native_amp_scaling_state" in self._loaded_checkpoint and isinstance(prec_plugin, MixedPrecisionPlugin):
    #        prec_plugin.load_state_dict(self._loaded_checkpoint["native_amp_scaling_state"])
#
    #def restore_callbacks(self) -> None:
    #    """Restores all callbacks from the pre-loaded checkpoint."""
    #    if not self._loaded_checkpoint:
    #        return
#
    #    trainer = self.trainer
    #    call._call_callbacks_on_load_checkpoint(trainer, self._loaded_checkpoint)
    #    call._call_callbacks_load_state_dict(trainer, self._loaded_checkpoint)
#
    #def restore_loops(self) -> None:
    #    """Restores the loop progress from the pre-loaded checkpoint.
#
    #    Calls hooks on the loops to give it a chance to restore its state from the checkpoint.
#
    #    """
    #    if not self._loaded_checkpoint:
    #        return
#
    #    fit_loop = self.trainer.fit_loop
    #    assert self.trainer.state.fn is not None
    #    state_dict = self._loaded_checkpoint.get("loops")
    #    if state_dict is not None:
    #        if self.trainer.state.fn == TrainerFn.FITTING:
    #            fit_loop.load_state_dict(state_dict["fit_loop"])
    #        elif self.trainer.state.fn == TrainerFn.VALIDATING:
    #            self.trainer.validate_loop.load_state_dict(state_dict["validate_loop"])
    #        elif self.trainer.state.fn == TrainerFn.TESTING:
    #            self.trainer.test_loop.load_state_dict(state_dict["test_loop"])
    #        elif self.trainer.state.fn == TrainerFn.PREDICTING:
    #            self.trainer.predict_loop.load_state_dict(state_dict["predict_loop"])
#
    #    if self.trainer.state.fn != TrainerFn.FITTING:
    #        return
#
    #    # crash if max_epochs is lower then the current epoch from the checkpoint
    #    if (
    #        self.trainer.max_epochs != -1
    #        and self.trainer.max_epochs is not None
    #        and self.trainer.current_epoch > self.trainer.max_epochs
    #    ):
    #        raise MisconfigurationException(
    #            f"You restored a checkpoint with current_epoch={self.trainer.current_epoch},"
    #            f" but you have set Trainer(max_epochs={self.trainer.max_epochs})."
    #        )
#
    #def restore_optimizers_and_schedulers(self) -> None:
    #    """Restores the optimizers and learning rate scheduler states from the pre-loaded checkpoint."""
    #    if not self._loaded_checkpoint:
    #        return
#
    #    if self.trainer.strategy.lightning_restore_optimizer:
    #        # validation
    #        if "optimizer_states" not in self._loaded_checkpoint:
    #            raise KeyError(
    #                "Trying to restore optimizer state but checkpoint contains only the model."
    #                " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
    #            )
    #        self.restore_optimizers()
#
    #    if "lr_schedulers" not in self._loaded_checkpoint:
    #        raise KeyError(
    #            "Trying to restore learning rate scheduler state but checkpoint contains only the model."
    #            " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
    #        )
    #    self.restore_lr_schedulers()
#
    #def restore_optimizers(self) -> None:
    #    """Restores the optimizer states from the pre-loaded checkpoint."""
    #    if not self._loaded_checkpoint:
    #        return
#
    #    # restore the optimizers
    #    self.trainer.strategy.load_optimizer_state_dict(self._loaded_checkpoint)
#
    #def restore_lr_schedulers(self) -> None:
    #    """Restores the learning rate scheduler states from the pre-loaded checkpoint."""
    #    if not self._loaded_checkpoint:
    #        return
#
    #    # restore the lr schedulers
    #    lr_schedulers = self._loaded_checkpoint["lr_schedulers"]
    #    for config, lrs_state in zip(self.trainer.lr_scheduler_configs, lr_schedulers):
    #        config.scheduler.load_state_dict(lrs_state)
#
    #def _restore_modules_and_callbacks(self, checkpoint_path: Optional[_PATH] = None) -> None:
    #    # restore modules after setup
    #    self.resume_start(checkpoint_path)
    #    self.restore_model()
    #    self.restore_datamodule()
    #    if self.trainer.state.fn == TrainerFn.FITTING:
    #        # restore callback states
    #        self.restore_callbacks()
    
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
                LightningDataModule.__class__.__qualname__: pl DataModule's state
            }

        """
        model = self.model
        #datamodule = trainer.datamodule

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
        #for obj in (model, datamodule):
        #    if obj and obj.hparams:
        #        if hasattr(obj, "_hparams_name"):
        #            checkpoint[obj.CHECKPOINT_HYPER_PARAMS_NAME] = obj._hparams_name
        #        # dump arguments
        #        if _OMEGACONF_AVAILABLE and isinstance(obj.hparams, Container):
        #            checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = obj.hparams
        #            checkpoint[obj.CHECKPOINT_HYPER_PARAMS_TYPE] = type(obj.hparams)
        #        else:
        #            checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = dict(obj.hparams)
#
        ## dump stateful datamodule
        #if datamodule is not None:
        #    datamodule_state_dict = call._call_lightning_datamodule_hook(trainer, "state_dict")
        #    if datamodule_state_dict:
        #        checkpoint[datamodule.__class__.__qualname__] = datamodule_state_dict
        return checkpoint
