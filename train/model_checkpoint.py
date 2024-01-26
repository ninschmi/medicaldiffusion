import logging
import os
import re
import shutil
import time
import warnings
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union
from weakref import proxy

import torch
import yaml
from torch import Tensor
from train.logger import TensorBoardLogger
from omegaconf import Container


class ModelCheckpoint():

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_EQUALS_CHAR = "="
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"
    STARTING_VERSION = 1

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        enable_version_counter: bool = True,
        logger: Optional[TensorBoardLogger] = None,
    ):
        self.monitor = monitor
        self.verbose = verbose
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.auto_insert_metric_name = auto_insert_metric_name
        self._save_on_train_epoch_end = save_on_train_epoch_end
        self._enable_version_counter = enable_version_counter
        self._last_global_step_saved = 0  # no need to save when no steps were taken
        self._last_time_checked: Optional[float] = None
        self.current_score: Optional[Tensor] = None
        self.best_k_models: Dict[str, Tensor] = {}
        self.kth_best_model_path = ""
        self.best_model_score: Optional[Tensor] = None
        self.best_model_path = ""
        self.last_model_path = ""
        self._last_checkpoint_saved = ""
        self.logger = logger
        self.num_val_batches
        self.val_check_interval

        self.kth_value: Tensor
        self.dirpath: Optional[Union[str, Path]]
        self.__init_monitor_mode(mode)
        self.__init_ckpt_dir(dirpath, filename)
        self.__init_triggers(every_n_train_steps, every_n_epochs, train_time_interval)
        self.__validate_init_configuration()

        self.setup()

    def setup(self, stage: str) -> None:
        dirpath = self.__resolve_ckpt_dir()
        self.dirpath = dirpath

    def on_train_start(self) -> None:
        self._last_time_checked = time.monotonic()

    def on_train_batch_end(
        self,
        global_step: int,
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        if self._last_global_step_saved == global_step:  # already saved at the last step
            return
        skip_batch = self._every_n_train_steps < 1 or (global_step % self._every_n_train_steps != 0)

        train_time_interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            skip_time = prev_time_check is None or (now - prev_time_check) < train_time_interval.total_seconds()

        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now

        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)

    def on_train_epoch_end(self, global_step, current_epoch) -> None:
        """Save a checkpoint at the end of the training epoch."""
        if not self._last_global_step_saved == global_step  and self._should_save_on_train_epoch_end():
            if self._every_n_epochs >= 1 and (current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def on_validation_end(self, global_step, current_epoch) -> None:
        """Save a checkpoint at the end of the validation stage."""
        if not self._last_global_step_saved == global_step and not self._should_save_on_train_epoch_end():
            if self._every_n_epochs >= 1 and (current_epoch + 1) % self._every_n_epochs == 0:
                self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "monitor": self.monitor,
            "best_model_score": self.best_model_score,
            "best_model_path": self.best_model_path,
            "current_score": self.current_score,
            "dirpath": self.dirpath,
            "best_k_models": self.best_k_models,
            "kth_best_model_path": self.kth_best_model_path,
            "kth_value": self.kth_value,
            "last_model_path": self.last_model_path,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)

        if self.dirpath == dirpath_from_ckpt:
            self.best_model_score = state_dict["best_model_score"]
            self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
            self.kth_value = state_dict.get("kth_value", self.kth_value)
            self.best_k_models = state_dict.get("best_k_models", self.best_k_models)
            self.last_model_path = state_dict.get("last_model_path", self.last_model_path)
        else:
            warnings.warn(
                f"The dirpath has changed from {dirpath_from_ckpt!r} to {self.dirpath!r},"
                " therefore `best_model_score`, `kth_best_model_path`, `kth_value`, `last_model_path` and"
                " `best_k_models` won't be reloaded. Only `best_model_path` will be reloaded."
            )

        self.best_model_path = state_dict["best_model_path"]

    def _save_topk_checkpoint(self, monitor_candidates: Dict[str, Tensor]) -> None:
        if self.save_top_k == 0:
            return

        # validate metric
        if self.monitor is not None:
            if self.monitor not in monitor_candidates:
                print(
                    f"`ModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                    f" metrics: {list(monitor_candidates)}."
                    f" HINT: Did you call `log({self.monitor!r}, value)` in the `LightningModule`?"
                )
            self._save_monitor_checkpoint(trainer, monitor_candidates)
        else:
            self._save_none_monitor_checkpoint(trainer, monitor_candidates)

    def _save_checkpoint(self, filepath: str) -> None:
        #This method needs to be called on all processes in case the selected strategy is handling distributed
        #checkpointing.
        checkpoint = self.dump_checkpoint(self.save_weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath)
        self.strategy.barrier("Trainer.save_checkpoint")



        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    @staticmethod
    def _link_checkpoint(trainer: "pl.Trainer", filepath: str, linkpath: str) -> None:
        if trainer.is_global_zero:
            if os.path.islink(linkpath) or os.path.isfile(linkpath):
                os.remove(linkpath)
            elif os.path.isdir(linkpath):
                shutil.rmtree(linkpath)
            os.symlink(filepath, linkpath)
        trainer.strategy.barrier()

    def _should_save_on_train_epoch_end(self) -> bool:
        if self._save_on_train_epoch_end is not None:
            return self._save_on_train_epoch_end

        # if `check_val_every_n_epoch != 1`, we can't say when the validation dataloader will be loaded
        # so let's not enforce saving at every training epoch end
        if trainer.check_val_every_n_epoch != 1:
            return False

        # no validation means save on train epoch end
        num_val_batches = (
            sum(self.num_val_batches) if isinstance(self.num_val_batches, list) else self.num_val_batches
        )
        if num_val_batches == 0:
            return True

        # if the user runs validation multiple times per training epoch, then we run after validation
        # instead of on train epoch end
        return self.val_check_interval == 1.0

    def __validate_init_configuration(self) -> None:
        if self.save_top_k < -1:
            raise MisconfigurationException(f"Invalid value for save_top_k={self.save_top_k}. Must be >= -1")
        if self._every_n_train_steps < 0:
            raise MisconfigurationException(
                f"Invalid value for every_n_train_steps={self._every_n_train_steps}. Must be >= 0"
            )
        if self._every_n_epochs < 0:
            raise MisconfigurationException(f"Invalid value for every_n_epochs={self._every_n_epochs}. Must be >= 0")

        every_n_train_steps_triggered = self._every_n_train_steps >= 1
        every_n_epochs_triggered = self._every_n_epochs >= 1
        train_time_interval_triggered = self._train_time_interval is not None
        if every_n_train_steps_triggered + every_n_epochs_triggered + train_time_interval_triggered > 1:
            raise MisconfigurationException(
                f"Combination of parameters every_n_train_steps={self._every_n_train_steps}, "
                f"every_n_epochs={self._every_n_epochs} and train_time_interval={self._train_time_interval} "
                "should be mutually exclusive."
            )

        if self.monitor is None and self.save_top_k not in (-1, 0, 1):
            # -1: save all epochs, 0: nothing is saved, 1: save last epoch
            raise MisconfigurationException(
                f"ModelCheckpoint(save_top_k={self.save_top_k}, monitor=None) is not a valid"
                " configuration. No quantity for top_k to track."
            )

    def __init_ckpt_dir(self, dirpath: Optional[_PATH], filename: Optional[str]) -> None:
        self._fs = get_filesystem(dirpath if dirpath else "")

        if dirpath and self._fs.protocol == "file":
            dirpath = os.path.realpath(dirpath)

        self.dirpath = dirpath
        self.filename = filename

    def __init_monitor_mode(self, mode: str) -> None:
        torch_inf = torch.tensor(torch.inf)
        mode_dict = {"min": (torch_inf, "min"), "max": (-torch_inf, "max")}

        if mode not in mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(mode_dict.keys())} but got {mode}")

        self.kth_value, self.mode = mode_dict[mode]

    def __init_triggers(
        self,
        every_n_train_steps: Optional[int],
        every_n_epochs: Optional[int],
        train_time_interval: Optional[timedelta],
    ) -> None:
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_epochs is set
        if every_n_train_steps is None and every_n_epochs is None and train_time_interval is None:
            every_n_epochs = 1
            every_n_train_steps = 0
            log.debug("Both every_n_train_steps and every_n_epochs are not set. Setting every_n_epochs=1")
        else:
            every_n_epochs = every_n_epochs or 0
            every_n_train_steps = every_n_train_steps or 0

        self._train_time_interval: Optional[timedelta] = train_time_interval
        self._every_n_epochs: int = every_n_epochs
        self._every_n_train_steps: int = every_n_train_steps

    @property
    def every_n_epochs(self) -> Optional[int]:
        return self._every_n_epochs

    def check_monitor_top_k(self, trainer: "pl.Trainer", current: Optional[Tensor] = None) -> bool:
        if current is None:
            return False

        if self.save_top_k == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
        should_update_best_and_save = monitor_op(current, self.best_k_models[self.kth_best_model_path])

        # If using multiple devices, make sure all processes are unanimous on the decision.
        should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

        return should_update_best_and_save

    @classmethod
    def _format_checkpoint_name(
        cls,
        filename: Optional[str],
        metrics: Dict[str, Tensor],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)

        # sort keys from longest to shortest to avoid replacing substring
        # eg: if keys are "epoch" and "epoch_test", the latter must be replaced first
        groups = sorted(groups, key=lambda x: len(x), reverse=True)

        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                filename = filename.replace(group, name + cls.CHECKPOINT_EQUALS_CHAR + "{" + name)

            # support for dots: https://stackoverflow.com/a/7934969
            filename = filename.replace(group, f"{{0[{name}]")

            if name not in metrics:
                metrics[name] = torch.tensor(0)
        filename = filename.format(metrics)

        if prefix:
            filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename

    def format_checkpoint_name(
        self, metrics: Dict[str, Tensor], filename: Optional[str] = None, ver: Optional[int] = None
    ) -> str:
        filename = filename or self.filename
        filename = self._format_checkpoint_name(filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name)

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        return os.path.join(self.dirpath, ckpt_name) if self.dirpath else ckpt_name

    def __resolve_ckpt_dir(self) -> Union[str, Path]:
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath

        if len(self.logger) > 0:
            if self.logger.save_dir is not None:
                save_dir = self.logger.save_dir
            else:
                save_dir = self.logger.root_dir
            name = self.logger.name
            version = self.logger.version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(self.logger.root_dir, "checkpoints")

        return ckpt_path

    def _find_last_checkpoints(self, trainer: "pl.Trainer") -> Set[str]:
        # find all checkpoints in the folder
        ckpt_path = self.__resolve_ckpt_dir(trainer)
        if self._fs.exists(ckpt_path):
            return {
                os.path.normpath(p)
                for p in self._fs.ls(ckpt_path, detail=False)
                if self.CHECKPOINT_NAME_LAST in os.path.split(p)[1]
            }
        return set()
    
    def _get_metric_interpolated_filepath_name(
        self, monitor_candidates: Dict[str, Tensor], trainer: "pl.Trainer", del_filepath: Optional[str] = None
    ) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while trainer.strategy.broadcast(self._fs.exists(filepath)) and filepath != del_filepath:
                filepath = self.format_checkpoint_name(monitor_candidates, ver=version_cnt)
                version_cnt += 1

        return filepath

    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]) -> None:
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while trainer.strategy.broadcast(self._fs.exists(filepath)) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        if self._fs.protocol == "file" and self._last_checkpoint_saved and self.save_top_k != 0:
            self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        else:
            self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)

    def _save_monitor_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]) -> None:
        assert self.monitor
        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            assert current is not None
            self._update_best_and_save(current, trainer, monitor_candidates)
        elif self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} was not in top {self.save_top_k}")

    def _save_none_monitor_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]) -> None:
        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer)
        # set the best model path before saving because it will be part of the state.
        previous, self.best_model_path = self.best_model_path, filepath
        self._save_checkpoint(trainer, filepath)

        if self.save_top_k == 1 and previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)

    def _update_best_and_save(
        self, current: Tensor, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        if del_filepath and self._should_remove_checkpoint(trainer, del_filepath, filepath):
            self._remove_checkpoint(trainer, del_filepath)

    def _should_remove_checkpoint(self, trainer: "pl.Trainer", previous: str, current: str) -> bool:
        """Checks if the previous checkpoint should be deleted.

        A checkpoint won't be deleted if any of the cases apply:
        - The previous checkpoint is the same as the current checkpoint
        - The previous checkpoint is not in the current checkpoint directory and the filesystem is local
        - The previous checkpoint is the checkpoint the Trainer resumed from and the filesystem is local

        """
        if previous == current:
            return False
        if self._fs.protocol != "file":
            return True
        previous = Path(previous).absolute()
        resume_path = Path(trainer.ckpt_path).absolute() if trainer.ckpt_path is not None else None
        if resume_path is not None and previous == resume_path:
            return False
        assert self.dirpath is not None
        dirpath = Path(self.dirpath).absolute()
        return dirpath in previous.parents

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Calls the strategy to remove the checkpoint file."""
        trainer.strategy.remove_checkpoint(filepath)

    def dump_checkpoint(self, global_step, current_epoch, weights_only: bool = False) -> dict:
        """Creating a model checkpoint dictionary object from various component states.

        Args:
            weights_only: saving model weights only
        Return:
            structured dictionary: {
                'epoch':                     training epoch
                'global_step':               training global step
                'pytorch-lightning_version': The version of PyTorch Lightning that produced this checkpoint
                'callbacks':                 "callback specific state"[] # if not weights_only
                'optimizer_states':          "PT optim's state_dict"[]   # if not weights_only
                'lr_schedulers':             "PT sched's state_dict"[]   # if not weights_only
                'state_dict':                Model's state_dict (e.g. network weights)
                precision_plugin.__class__.__qualname__:  precision plugin state_dict # if not weights_only
                CHECKPOINT_HYPER_PARAMS_NAME:
                CHECKPOINT_HYPER_PARAMS_KEY:
                CHECKPOINT_HYPER_PARAMS_TYPE:
                something_cool_i_want_to_save: anything you define through model.on_save_checkpoint
                LightningDataModule.__class__.__qualname__: pl DataModule's state
            }

        """
        checkpoint = {
            # the epoch and global step are saved for compatibility but they are not relevant for restoration
            "epoch": current_epoch,
            "global_step": global_step,
        }

        if not weights_only:
            # dump callbacks
            checkpoint["callbacks"] = call._call_callbacks_state_dict(trainer)

            optimizer_states = []
            for i, optimizer in enumerate(trainer.optimizers):
                # Rely on accelerator to dump optimizer state
                optimizer_state = trainer.strategy.optimizer_state(optimizer)
                optimizer_states.append(optimizer_state)

            checkpoint["optimizer_states"] = optimizer_states

            # dump lr schedulers
            lr_schedulers = []
            for config in trainer.lr_scheduler_configs:
                lr_schedulers.append(config.scheduler.state_dict())
            checkpoint["lr_schedulers"] = lr_schedulers

            # precision plugin
            prec_plugin = trainer.precision_plugin
            prec_plugin_state_dict = prec_plugin.state_dict()
            if prec_plugin_state_dict:
                checkpoint[prec_plugin.__class__.__qualname__] = prec_plugin_state_dict
            prec_plugin.on_save_checkpoint(checkpoint)
            

        # dump hyper-parameters
        for obj in (model, datamodule):
            if obj and obj.hparams:
                if hasattr(obj, "_hparams_name"):
                    checkpoint[obj.CHECKPOINT_HYPER_PARAMS_NAME] = obj._hparams_name
                # dump arguments
                if _OMEGACONF_AVAILABLE and isinstance(obj.hparams, Container):
                    checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = obj.hparams
                    checkpoint[obj.CHECKPOINT_HYPER_PARAMS_TYPE] = type(obj.hparams)
                else:
                    checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = dict(obj.hparams)

        # dump stateful datamodule
        if datamodule is not None:
            datamodule_state_dict = call._call_lightning_datamodule_hook(trainer, "state_dict")
            if datamodule_state_dict:
                checkpoint[datamodule.__class__.__qualname__] = datamodule_state_dict

        # on_save_checkpoint hooks
        if not weights_only:
            # if state is returned from callback's on_save_checkpoint
            # it overrides the returned state from callback's state_dict
            # support for returning state in on_save_checkpoint
            # will be removed in v1.8
            call._call_callbacks_on_save_checkpoint(trainer, checkpoint)
        call._call_lightning_module_hook(trainer, "on_save_checkpoint", checkpoint)
        return checkpoint
