""" Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import warnings
import torch
import imageio
import inspect

import math
import numpy as np
import skvideo.io

import sys
import pdb as pdb_original
import SimpleITK as sitk
import logging

from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple, Type

import imageio.core.util
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def adopt_weight(global_step, threshold=0, value=0.):
    weight = 1
    if global_step < threshold:
        weight = value
    return weight

def comp_getattr(args, attr_name, default=None):
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    else:
        return default


def visualize_tensors(t, name=None, nest=0):
    if name is not None:
        print(name, "current nest: ", nest)
    print("type: ", type(t))
    if 'dict' in str(type(t)):
        print(t.keys())
        for k in t.keys():
            if t[k] is None:
                print(k, "None")
            else:
                if 'Tensor' in str(type(t[k])):
                    print(k, t[k].shape)
                elif 'dict' in str(type(t[k])):
                    print(k, 'dict')
                    visualize_tensors(t[k], name, nest + 1)
                elif 'list' in str(type(t[k])):
                    print(k, len(t[k]))
                    visualize_tensors(t[k], name, nest + 1)
    elif 'list' in str(type(t)):
        print("list length: ", len(t))
        for t2 in t:
            visualize_tensors(t2, name, nest + 1)
    elif 'Tensor' in str(type(t)):
        print(t.shape)
    else:
        print(t)
    return ""


def _default_map_location(storage: "UntypedStorage", location: str) -> Optional["UntypedStorage"]:
    if (
        location.startswith("mps")
        or location.startswith("cuda")
        and not torch.cuda.device_count()
        or location.startswith("xla")
    ):
        return storage.cpu()
    return None  # default behavior by `torch.load()`


def _load_state(
    cls: Type[torch.nn.Module],
    checkpoint: Dict[str, Any],
    strict: Optional[bool] = None,
    **cls_kwargs_new: Any,
) -> torch.nn.Module:
    cls_spec = inspect.getfullargspec(cls.__init__)
    cls_init_args_name = inspect.signature(cls.__init__).parameters.keys()

    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    drop_names = [n for n in (self_var, args_var, kwargs_var) if n]
    cls_init_args_name = list(filter(lambda n: n not in drop_names, cls_init_args_name))

    cls_kwargs_loaded = {}
    # pass in the values we saved automatically
    if "hyper_parameters" in checkpoint:
        # 2. Try to restore model hparams from checkpoint using the new key
        cls_kwargs_loaded.update(checkpoint.get("hyper_parameters", {}))

        # 3. Ensure that `cls_kwargs_old` has the right type, back compatibility between dict and Namespace
        cls_kwargs_loaded = _convert_loaded_hparams(cls_kwargs_loaded, checkpoint.get("hparams_type"))

        # 4. Update cls_kwargs_new with cls_kwargs_old, such that new has higher priority
        args_name = checkpoint.get("hparams_name")
        if args_name and args_name in cls_init_args_name:
            cls_kwargs_loaded = {args_name: cls_kwargs_loaded}

    _cls_kwargs = {}
    _cls_kwargs.update(cls_kwargs_loaded)
    _cls_kwargs.update(cls_kwargs_new)

    if not cls_spec.varkw:
        # filter kwargs according to class init unless it allows any argument via kwargs
        _cls_kwargs = {k: v for k, v in _cls_kwargs.items() if k in cls_init_args_name}

    obj = cls(**_cls_kwargs)

    # load the state_dict on the model automatically
    assert strict is not None
    keys = obj.load_state_dict(checkpoint["state_dict"], strict=strict)

    if not strict:
        if keys.missing_keys:
            #rank_zero_warn
            print(
                f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
            )
        if keys.unexpected_keys:
            #rank_zero_warn
            print(
                f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
            )

    return obj


def parse_class_init_keys(cls: Type) -> Tuple[str, Optional[str], Optional[str]]:
    init_parameters = inspect.signature(cls.__init__).parameters
    # docs claims the params are always ordered
    # https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
    init_params = list(init_parameters.values())
    # self is always first
    n_self = init_params[0].name

    def _get_first_if_any(
        params: List[inspect.Parameter],
        param_type: Literal[inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD],
    ) -> Optional[str]:
        for p in params:
            if p.kind == param_type:
                return p.name
        return None

    n_args = _get_first_if_any(init_params, inspect.Parameter.VAR_POSITIONAL)
    n_kwargs = _get_first_if_any(init_params, inspect.Parameter.VAR_KEYWORD)

    return n_self, n_args, n_kwargs


def _convert_loaded_hparams(
    model_args: Dict[str, Any], hparams_type: Optional[Union[Callable, str]] = None
) -> Dict[str, Any]:
    # if not hparams type define
    if not hparams_type:
        return model_args
    # if past checkpoint loaded, convert str to callable
    if isinstance(hparams_type, str):
        hparams_type = Dict
    # convert hparams
    return hparams_type(model_args)

