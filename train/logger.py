"Adapted from pytorch-lightning: https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/loggers/tensorboard.py"

from enum import Enum
import logging
import numpy as np
import os
from copy import deepcopy
import yaml
from pathlib import Path
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Dict, Mapping, MutableMapping, Optional, Union
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from torch.nn import Module
from omegaconf import Container, OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.errors import UnsupportedValueType, ValidationError
from torch.utils.tensorboard.summary import hparams
from torchmetrics import Metric, MetricCollection
from train.results_torch import _ResultCollection

def _flatten_dict(params: MutableMapping[Any, Any], delimiter: str = "/", parent_key: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in params.items():
        new_key = parent_key + delimiter + str(k) if parent_key else str(k)
        if isinstance(v, Namespace):
            v = vars(v)
        if isinstance(v, MutableMapping):
            result = {**result, **_flatten_dict(v, parent_key=new_key, delimiter=delimiter)}
        else:
            result[new_key] = v
    return result

def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    for k in params:
        # convert relevant np scalars to python types first (instead of str)
        if isinstance(params[k], (np.bool_, np.integer, np.floating)):
            params[k] = params[k].item()
        elif type(params[k]) not in [bool, int, float, str, Tensor]:
            params[k] = str(params[k])
    return params

def save_hparams_to_yaml(config_yaml: Union[str, Path], hparams: Union[dict, Namespace], use_omegaconf: bool = True) -> None:
    if not os.path.isdir(os.path.dirname(config_yaml)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(config_yaml)}.")

    # convert Namespace or AD to dict
    if isinstance(hparams, Namespace):
        hparams = vars(hparams)

    # saving with OmegaConf objects
    if use_omegaconf:
        # deepcopy: hparams from user shouldn't be resolved
        hparams = deepcopy(hparams)
        assert isinstance(hparams, dict) and all(isinstance(x, DictConfig) for x in hparams.values())
        hparams = {k: OmegaConf.to_container(v, resolve=True) for k, v in hparams.items()}
        with open(config_yaml, "w", encoding="utf-8") as fp:
            try:
                OmegaConf.save(hparams, fp)
                return
            except (UnsupportedValueType, ValidationError):
                pass

    if not isinstance(hparams, dict):
        raise TypeError("hparams must be dictionary")

    hparams_allowed = {}
    # drop parameters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        try:
            v = v.name if isinstance(v, Enum) else v
            yaml.dump(v)
        except TypeError:
            print(f"Skipping '{k}' parameter because it is not possible to safely dump to YAML.")
            hparams[k] = type(v).__name__
        else:
            hparams_allowed[k] = v

    # saving the standard way
    with open(config_yaml, "w", newline="") as fp:
        yaml.dump(hparams_allowed, fp)

class TensorBoardLogger():
   
    LOGGER_JOIN_CHAR = "-"
    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(
        self,
        root_dir: Union[str, Path],
        name: Optional[str] = "lightning_logs",
        default_hp_metric: bool = True,
    ):
        self._name = name or ""
        self.root_dir = root_dir
        self._log_dir = self.configure_logger()
        self.logger = SummaryWriter(log_dir=self.log_dir)
        self.default_hp_metric = default_hp_metric
        self.hparams: Union[Dict[str, Any], Namespace] = {}
        self.metrics = {}
        self.train_results = _ResultCollection(training=True)
        self.val_results = _ResultCollection(training=False)
        self._train_state = None
        self.current_epoch = None
    
    def configure_logger(self):
        self._log_dir = os.path.join(self.root_dir, f'lightning_logs')
        self._version = 0
        if os.path.exists(self.log_dir):
            dirs = os.listdir(self.log_dir)
            version_dirs = [d for d in dirs if d.startswith('version_')]
            version_numbers = [int(d.split('_')[1]) for d in version_dirs]
            if len(version_numbers) == 0:
                return os.path.join(self.log_dir, f'version_{self.version}')
            self._version = max(version_numbers, default=0)
            return os.path.join(self.log_dir, f'version_{self.version+1}')
        else:
            print("Missing logger folder: %s", self.log_dir)
            return os.path.join(self.log_dir, f'version_{self.version}')

    def log(
        self,
        name: str,
        value: Union[Metric, Tensor],
        prog_bar: bool = False,
        on_step: Optional[bool] = False,
        on_epoch: Optional[bool] = True,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,   
    ) -> None:  
        #TODO
        #if trainer._logger_connector.should_reset_tensors(self._current_fx_name):
        #    # if we started a new epoch (running its first batch) the hook name has changed
        #    # reset any tensors for the new hook name
        #    results.reset(metrics=False, fx=self._current_fx_name)
        if 'train' in name:
            results = self.train_results
        elif 'val' in name:
            results = self.val_results
            
        assert results is not None
        results.log(name,
                    value,
                    prog_bar=prog_bar,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    sync_dist=sync_dist,
                    sync_dist_group=sync_dist_group,
                    batch_size=batch_size,
                    metric_attribute=metric_attribute,
                    rank_zero_only=rank_zero_only)

    def log_dict(
        self,
        dictionary: Union[Mapping[str, Union[Metric, Tensor]], MetricCollection],
        prog_bar: bool = False,
        on_step: Optional[bool] = False,
        on_epoch: Optional[bool] = True,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        batch_size: Optional[int] = None,
        rank_zero_only: bool = False,
    ) -> None:

        kwargs: Dict[str, bool] = {}

        if isinstance(dictionary, MetricCollection):
            kwargs["keep_base"] = False
            if dictionary._enable_compute_groups:
                kwargs["copy_state"] = False

        # TODO
        # currently only supported for train_results
        for k, v in dictionary.items(**kwargs):
            self.train_results.log(
                name=k,
                value=v,
                prog_bar=prog_bar,
                on_step=on_step,
                on_epoch=on_epoch,
                sync_dist=sync_dist,
                sync_dist_group=sync_dist_group,
                batch_size=batch_size,
                rank_zero_only=rank_zero_only,
            )
        return None

    #@rank_zero_only
    def log_metrics(self, metrics: Dict[str, Tensor], step: int) -> None:
        if not metrics:
            return

        #convert single-item tensors to scalar values.
        #metrics = {k: v.item() for k, v in metrics.items()}
        
        if step is None:
            step = metrics.pop("step", None)

        metrics.setdefault("epoch", self.current_epoch)

        assert step is not None, "You must provide a step when logging with the TensorBoardLogger."

        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.logger.add_scalars(k, v, step)
            else:
                try:
                    self.logger.add_scalar(k, v, step)
                # TODO(fabric): specify the possible exception
                except Exception as ex:
                    raise ValueError(
                        f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    ) from ex
                
        self.save()        

    #@rank_zero_experiment
    #@rank_zero_only
    def log_hyperparams(  # type: ignore[override]
        self, params: Union[Dict[str, Any], Namespace], metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        
        # in case converting from namespace
        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        # store params to output
        if isinstance(params, Container):
            self.hparams = OmegaConf.merge(self.hparams, params)
        else:
            self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = _flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self.default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            self.log_metrics(metrics, 0)

            exp, ssi, sei = hparams(params, metrics)

            writer = self.logger._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

    def reset_results(self) -> None:
        if self.train_results is not None:
            self.train_results.reset()
        elif self.val_results is not None:
            self.val_results.reset()

    def update_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def on_train_batch_start(self, batch) -> None:
        #if self._first_loop_iter is None:
        #    self._first_loop_iter = True
        #elif self._first_loop_iter is True:
        #    self._first_loop_iter = False
        assert self.train_results is not None
        # attach reference to the new batch and remove the cached batch_size
        self.train_results.batch = batch
        self.train_results.batch_size = None

    def on_train_batch_end(self, step) -> None:
        #assert isinstance(self._first_loop_iter, bool)
        assert self.train_results is not None
        # drop the reference to current batch and batch_size
        self.train_results.batch = None
        self.train_results.batch_size = None

        # Log Batch Metrics
        # when metrics should be logged and no gradient accumulation is selected
        #assert isinstance(self._first_loop_iter, bool)
        if self.should_update_logs(step):    # False if step % trainer.log_every_n_steps != 0 or log_every_n_steps == 0
            self.log_metrics(self.train_results.metrics(on_step=True), step=step)
    
    def on_val_batch_start(self, batch) -> None:
        #if self._first_loop_iter is None:
        #    self._first_loop_iter = True
        #elif self._first_loop_iter is True:
        #    self._first_loop_iter = False
        assert self.val_results is not None
        # attach reference to the new batch and remove the cached batch_size
        self.val_results.batch = batch
        self.val_results.batch_size = None

    def on_val_batch_end(self, step) -> None:
        #assert isinstance(self._first_loop_iter, bool)
        assert self.val_results is not None
        # drop the reference to current batch and batch_size
        self.val_results.batch = None
        self.val_results.batch_size = None

        # Log Batch Metrics
        #assert isinstance(self._first_loop_iter, bool)
        # logs user requested information to logger
        self.log_metrics(self.val_results.metrics(on_step=True), step=step)

    def teardown(self) -> None:
        self.train_results.cpu()
        self.val_results.cpu()

    def log_results(self, step: int) -> None:
        #assert self._first_loop_iter is None
        self.log_metrics(self.val_results.metrics(on_step=False), step=step) #on evaluation end
        self.log_metrics(self.train_results.metrics(on_step=False), step=step)   #on training epoch end

    def should_update_logs(self, step) -> bool:
        log_every_n_steps = 50
        if log_every_n_steps == 0:
            return False
        # + 1 inherited from pytorch lightning
        should_log = (step + 1) % log_every_n_steps == 0
        return should_log

    #@rank_zero_only
    def save(self) -> None:
        self.logger.flush()

        # prepare the file path
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist and the log directory exists
        if os.path.isdir(self.log_dir) and not os.path.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)

    #@rank_zero_only
    def finalize(self, status: str) -> None:
        if self.logger is not None:
            self.logger.flush()
            self.logger.close()
        if status == "success":
            # saving hparams happens independent of experiment manager
            self.save()

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> Union[int, str]:
        assert self._version is not None
        return self._version
    
    @property
    def log_dir(self) -> str:
        return self._log_dir
    
    @property
    def train_state(self) -> str:
        return self._train_state
    
    @train_state.setter
    def train_state(self, val: bool) -> None:
        if val:
            self._train_state = True
        elif self._train_state:
            self._train_state = None
        
    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        params = sanitize_params(params)
        # logging of arrays with dimension > 1 is not supported, sanitize as string
        return {k: str(v) if hasattr(v, "ndim") and v.ndim > 1 else v for k, v in params.items()}
