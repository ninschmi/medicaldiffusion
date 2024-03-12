"Adapted from pytorch-lightning: https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/trainer/connectors/logger_connector/result.py"

from dataclasses import dataclass, is_dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, Iterable, List, Mapping, Optional, Tuple, Union, cast
from typing_extensions import TypedDict

import torch
from torch import Tensor
from torchmetrics import Metric

_VALUE = Union[Metric, Tensor]  # Do not include scalars as they were converted to tensors
_OUT_DICT = Dict[str, Tensor]
BType = Union[Tensor, str, Mapping[Any, "BType"], Iterable["BType"]]

class _METRICS(TypedDict):
    callback: _OUT_DICT
    log: _OUT_DICT

@dataclass
class _Sync:
    fn: Optional[Callable] = None
    _should: bool = False
    rank_zero_only: bool = False
    _op: Optional[str] = None
    _group: Optional[Any] = None

    def __post_init__(self) -> None:
        self._generate_sync_fn()

    @property
    def should(self) -> bool:
        return self._should

    @should.setter
    def should(self, should: bool) -> None:
        self._should = should
        # `self._fn` needs to be re-generated.
        self._generate_sync_fn()

    @property
    def op(self) -> Optional[str]:
        return self._op

    @op.setter
    def op(self, op: Optional[str]) -> None:
        self._op = op
        # `self._fn` needs to be re-generated.
        self._generate_sync_fn()

    @property
    def group(self) -> Optional[Any]:
        return self._group

    @group.setter
    def group(self, group: Optional[Any]) -> None:
        self._group = group
        # `self._fn` needs to be re-generated.
        self._generate_sync_fn()

    def _generate_sync_fn(self) -> None:
        """Used to compute the syncing function and cache it."""
        fn = self.no_op if self.fn is None or not self.should or self.rank_zero_only else self.fn
        # save the function as `_fn` as the meta are being re-created and the object references need to match.
        # ignore typing, bad support for `partial`: mypy/issues/1484
        self._fn: Callable = partial(fn, reduce_op=self.op, group=self.group)  # type: ignore [arg-type]

    @property
    def __call__(self) -> Any:
        return self._fn

    @staticmethod
    def no_op(value: Any, *_: Any, **__: Any) -> Any:
        return value


@dataclass
class _Metadata:
    name: str
    prog_bar: bool = False
    on_step: bool = False
    on_epoch: bool = True
    # https://github.com/pytorch/pytorch/issues/96197
    reduce_fx: Callable = torch.mean  # type: ignore[assignment]
    metric_attribute: Optional[str] = None
    _sync: Optional[_Sync] = None

    def __post_init__(self) -> None:
        if not self.on_step and not self.on_epoch:
            raise Exception("`self.log(on_step=False, on_epoch=False)` is not useful.")

    @property
    def sync(self) -> _Sync:
        assert self._sync is not None
        return self._sync

    @sync.setter
    def sync(self, sync: _Sync) -> None:
        if sync.op is None:
            sync.op = self.reduce_fx.__name__
        self._sync = sync

    def forked_name(self, on_step: bool) -> str:
        if self.on_step and self.on_epoch:
            return f'{self.name}_{"step" if on_step else "epoch"}'
        return self.name


class _ResultMetric(Metric):
    """Wraps the value provided to `:meth:`~pytorch_lightning.core.LightningModule.log`"""

    def __init__(self, metadata: _Metadata, is_tensor: bool) -> None:
        super().__init__()
        self.is_tensor = is_tensor
        self.meta = metadata
        self.has_reset = False
        if is_tensor:
            default = 0.0
            # the logged value will be stored in float32 or higher to maintain accuracy
            self.add_state("value", torch.tensor(default, dtype=_get_default_dtype()), dist_reduce_fx=torch.sum)
            self.cumulated_batch_size: Tensor
            self.add_state("cumulated_batch_size", torch.tensor(0), dist_reduce_fx=torch.sum)
        # this is defined here only because upstream is missing the type annotation
        self._forward_cache: Optional[Any] = None

    def update(self, value: _VALUE, batch_size: int) -> None:
        if self.is_tensor:
            value = cast(Tensor, value)
            dtype = _get_default_dtype()
            if not torch.is_floating_point(value):
                print(
                    # do not include the value to avoid cache misses
                    f"You called `self.log({self.meta.name!r}, ...)` but the value needs to"
                    f" be floating point. Converting it to {dtype}."
                )
                value = value.to(dtype)
            if value.dtype not in (torch.float32, torch.float64):
                value = value.to(dtype)

            if self.meta.on_step:
                self._forward_cache = self.meta.sync(value.clone())  # `clone` because `sync` is in-place
                # performance: no need to accumulate on values only logged on_step
                if not self.meta.on_epoch:
                    self.value = self._forward_cache
                    return

            # perform accumulation with reduction
            # do not use `+=` as it doesn't do type promotion
            self.value = self.value + value * batch_size
            self.cumulated_batch_size = self.cumulated_batch_size + batch_size

        else:
            value = cast(Metric, value)
            self.value = value
            self._forward_cache = value._forward_cache

    def compute(self) -> Tensor:
        if self.is_tensor:
            value = self.meta.sync(self.value.clone())  # `clone` because `sync` is in-place
            cumulated_batch_size = self.meta.sync(self.cumulated_batch_size)
            return value / cumulated_batch_size
        return self.value.compute()

    def reset(self) -> None:
        if self.is_tensor:
            super().reset()
        else:
            self.value.reset()
        self.has_reset = True

    def forward(self, value: _VALUE, batch_size: int) -> None:
        # performance: skip the `torch.no_grad` context manager by calling `update` directly
        self.update(value, batch_size)

    def _wrap_compute(self, compute: Any) -> Any:
        # Override to avoid syncing - we handle it ourselves.
        @wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            update_called = self.update_called
            if not update_called:
                print(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated.",
                )

            # return cached value
            if self._computed is not None:
                return self._computed
            self._computed = compute(*args, **kwargs)
            return self._computed

        return wrapped_func

    def __setattr__(self, key: str, value: Any) -> None:
        # performance: skip the `torch.nn.Module.__setattr__` checks
        object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        state = f"{repr(self.meta.name)}, value={self.value}"
        if self.is_tensor:
            state += f", cumulated_batch_size={self.cumulated_batch_size}"
        return f"{self.__class__.__name__}({state})"

    def to(self, *args: Any, **kwargs: Any) -> "_ResultMetric":
        d = self.__dict__
        d = dict(d) # https://github.com/pytorch/pytorch/issues/96198
        self.__dict__.update({k: move_data_to_device(v, *args, **kwargs) for k, v in d.items()})
        return self


class _ResultCollection(dict):
    """Collection (dictionary) of :class:`~pytorch_lightning.trainer.connectors.logger_connector.result._ResultMetric`

        # arguments: key, value, metadata

    """

    def __init__(self, training: bool) -> None:
        super().__init__()
        self.training = training
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None

    @property
    def result_metrics(self) -> List[_ResultMetric]:
        return list(self.values())

    def _extract_batch_size(self, value: _ResultMetric, batch_size: Optional[int], meta: _Metadata) -> int:
        # check if we have extracted the batch size already
        if batch_size is None:
            batch_size = self.batch_size

        if batch_size is not None:
            return batch_size

        batch_size = 1
        if self.batch is not None and value.is_tensor and meta.on_epoch:
            # Unpack a batch to find a ``torch.Tensor``.
            # returns: ``len(tensor)`` when found, or ``1`` when it hits an empty or non iterable.
            error_msg = (
                "We could not infer the batch_size from the batch. Either simplify its structure"
                " or provide the batch_size as `self.log(..., batch_size=batch_size)`."
            )
            batch_size = None

            try:
                for bs in _extract_batch_size(self.batch):
                    if batch_size is None:
                        batch_size = bs
                    elif batch_size != bs:
                        print(
                            "Trying to infer the `batch_size` from an ambiguous collection. The batch size we"
                            f" found is {batch_size}. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`."
                        )
                        break
            except RecursionError:
                raise RecursionError(error_msg)

            if batch_size is None:
                raise Exception(error_msg)

            self.batch_size = batch_size

        return batch_size

    def log(
        self,
        name: str,
        value: _VALUE,
        prog_bar: bool = False,
        on_step: bool = False,
        on_epoch: bool = True,
        # https://github.com/pytorch/pytorch/issues/96197
        reduce_fx: Callable = torch.mean,  # type: ignore[assignment]
        sync_dist: bool = False,
        sync_dist_fn: Callable = _Sync.no_op,
        sync_dist_group: Optional[Any] = None,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        # no metrics should be logged with graphs:
        value = detach_and_move(value, to_cpu=False) if isinstance(value, Tensor) else value

        # storage key
        key = f"{name}"

        meta = _Metadata(
            name=name,
            prog_bar=prog_bar,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            metric_attribute=metric_attribute,
        )
        meta.sync = _Sync(_should=sync_dist, fn=sync_dist_fn, _group=sync_dist_group, rank_zero_only=rank_zero_only)

        # register logged value if it doesn't exist
        if key not in self:
            self.register_key(key, meta, value)

        # check the stored metadata and the current one match
        elif meta != self[key].meta:
            raise Exception(
                f"You called `self.log({name}, ...)` twice with different arguments. This is not allowed"
            )

        batch_size = self._extract_batch_size(self[key], batch_size, meta)
        self.update_metrics(key, value, batch_size)

    def register_key(self, key: str, meta: _Metadata, value: _VALUE) -> None:
        """Create one _ResultMetric object per value.
        Value can be provided as a nested collection
        """
        metric = _ResultMetric(meta, isinstance(value, Tensor)).to(value.device)
        self[key] = metric

    def update_metrics(self, key: str, value: _VALUE, batch_size: int) -> None:
        result_metric = self[key]
        # performance: avoid calling `__call__` to avoid the checks in `torch.nn.Module._call_impl`
        result_metric.forward(value, batch_size)
        result_metric.has_reset = False

    @staticmethod
    def _get_cache(result_metric: _ResultMetric, on_step: bool) -> Optional[Tensor]:
        cache = None
        if on_step and result_metric.meta.on_step:
            cache = result_metric._forward_cache
        elif not on_step and result_metric.meta.on_epoch:
            if result_metric._computed is None:
                should = result_metric.meta.sync.should
                if not should and result_metric.is_tensor and torch.distributed.is_initialized():
                    print(
                        f"It is recommended to use `self.log({result_metric.meta.name!r}, ..., sync_dist=True)`"
                        " when logging on epoch level in distributed setting to accumulate the metric across"
                        " devices.")
                result_metric.compute()
                result_metric.meta.sync.should = should

            cache = result_metric._computed

        if cache is not None:
            if not isinstance(cache, Tensor):
                raise ValueError(
                    f"The `.compute()` return of the metric logged as {result_metric.meta.name!r} must be a tensor."
                    f" Found {cache}"
                )
            return cache.detach()

        return cache

    def valid_items(self) -> Generator:
        """This function is used to iterate over current valid metrics."""
        return ((k, v) for k, v in self.items() if not v.has_reset)

    def _forked_name(self, result_metric: _ResultMetric, on_step: bool) -> Tuple[str, str]:
        name = result_metric.meta.name
        forked_name = result_metric.meta.forked_name(on_step)
        return name, forked_name

    def metrics(self, on_step: bool) -> _METRICS:
        metrics = _METRICS(callback={}, log={})

        for _, result_metric in self.valid_items():
            # extract forward_cache or computed from the _ResultMetric
            value = self._get_cache(result_metric, on_step)
            if not isinstance(value, Tensor):
                continue

            name, forked_name = self._forked_name(result_metric, on_step)

            # populate logging metrics
            metrics["log"][forked_name] = value

            # populate callback metrics. callback metrics don't take `_step` forked metrics
            if self.training or result_metric.meta.on_epoch and not on_step:
                metrics["callback"][name] = value
                metrics["callback"][forked_name] = value

            #TODO
            ## populate progress_bar metrics. convert tensors to numbers
            #if result_metric.meta.prog_bar:
            #    metrics["pbar"][forked_name] = convert_tensors_to_scalars(value)

        return metrics

    def reset(self, metrics: Optional[bool] = None, fx: Optional[str] = None) -> None:
        """Reset the result collection.

        Args:
            metrics: If True, only ``torchmetrics.Metric`` results are reset,
                if False, only ``torch.Tensors`` are reset,
                if ``None``, both are.
            fx: Function to reset

        """
        for item in self.values():
            requested_type = metrics is None or metrics ^ item.is_tensor
            same_fx = fx is None or fx == item.meta.fx
            if requested_type and same_fx:
                item.reset()

    def to(self, *args: Any, **kwargs: Any) -> "_ResultCollection":
        """Move all data to the given device."""
        self.update({k: move_data_to_device(v, *args, **kwargs) for k, v in dict(self).items()})
        return self

    def cpu(self) -> "_ResultCollection":
        """Move all data to CPU."""
        return self.to(device="cpu")

    def __str__(self) -> str:
        # remove empty values
        self_str = str({k: v for k, v in self.items() if v})
        return f"{self.__class__.__name__}({self_str})"

    def __repr__(self) -> str:
        return f"{{{self.training}, {super().__repr__()}}}"


def _get_default_dtype() -> torch.dtype:
    """The default dtype for new tensors, but no lower than float32."""
    dtype = torch.get_default_dtype()
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32


def move_data_to_device(batch: Any, device: Union[torch.device, str, int]) -> Any:
    if isinstance(device, str):
        device = torch.device(device)
    def batch_to(data: Any) -> Any:
        kwargs = {}
        # Don't issue non-blocking transfers to CPU
        # Same with MPS due to a race condition bug: https://github.com/pytorch/pytorch/issues/83015
        if isinstance(data, Tensor) and isinstance(device, torch.device) and device.type not in ("cpu", "mps"):
            kwargs["non_blocking"] = True
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        # user wrongly implemented the `_TransferableDataType` and forgot to return `self`.
        return data
    return batch_to(batch) if isinstance(batch, Tensor) else batch

def detach_and_move(t: Tensor, to_cpu: bool) -> Tensor:
    t = t.detach()
    if to_cpu:
        t = t.cpu()
    return t

def _extract_batch_size(batch: BType) -> Generator[Optional[int], None, None]:
    if isinstance(batch, Tensor):
        if batch.ndim == 0:
            yield 1
        else:
            yield batch.size(0)
    elif isinstance(batch, (Iterable, Mapping)) and not isinstance(batch, str):
        if isinstance(batch, Mapping):
            batch = batch.values()

        for sample in batch:
            yield from _extract_batch_size(sample)
    elif is_dataclass(batch) and not isinstance(batch, type):
        for field in fields(batch):  # type: ignore[arg-type]
            yield from _extract_batch_size(getattr(batch, field.name))
    else:
        yield None