# utilies from pytorch-lightning https://github.com/Lightning-AI/pytorch-lightning/tree/master

import inspect
import os
import signal
import subprocess
import sys
import time
import types
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple, Type, Union

class _ChildProcessObserver:
    def __init__(self, main_pid: int, child_processes: List[subprocess.Popen], sleep_period: int = 5) -> None:
        self._main_pid = main_pid
        self._child_processes = child_processes
        self._sleep_period = sleep_period
        # Note: SIGTERM is not aggressive enough to terminate processes hanging in collectives
        self._termination_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
        self._finished = False

    def __call__(self) -> None:
        while not self._finished:
            time.sleep(self._sleep_period)
            self._finished = self._run()

    def _run(self) -> bool:
        """Runs once over all child processes to check whether they are still running."""
        for proc in self._child_processes:
            proc.poll()

        return_codes = [proc.returncode for proc in self._child_processes]
        if all(return_code == 0 for return_code in return_codes):
            return True

        for i, proc in enumerate(self._child_processes):
            if proc.returncode:
                message = print(
                    f"Child process with PID {proc.pid} terminated with code {proc.returncode}."
                    f" Forcefully terminating all other processes to avoid zombies 🧟",
                    rank=(i + 1),
                )
                self._terminate_all()
                return True

        return False

    def _terminate_all(self) -> None:
        """Terminates the main process and all its children."""
        for p in self._child_processes:
            p.send_signal(self._termination_signal)
        os.kill(self._main_pid, self._termination_signal)


class AttributeDict(Dict):
    """Extended dictionary accessible with dot notation.

    >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    >>> ad.key1
    1
    >>> ad.update({'my-key': 3.14})
    >>> ad.update(new_key=42)
    >>> ad.key1 = 2
    >>> ad
    "key1":    2
    "key2":    abc
    "my-key":  3.14
    "new_key": 42

    """

    def __getattr__(self, key: str) -> Optional[Any]:
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        return "\n".join(rows)


def collect_init_args(
    frame: types.FrameType,
    path_args: List[Dict[str, Any]],
    inside: bool = False,
    classes: Tuple[Type, ...] = (),
) -> List[Dict[str, Any]]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    # frame.f_back must be of a type types.FrameType for get_init_args/collect_init_args due to mypy
    if not isinstance(frame.f_back, types.FrameType):
        return path_args

    local_self, local_args = _get_init_args(frame)
    if "__class__" in local_vars and (not classes or isinstance(local_self, classes)):
        # recursive update
        path_args.append(local_args)
        return collect_init_args(frame.f_back, path_args, inside=True, classes=classes)
    if not inside:
        return collect_init_args(frame.f_back, path_args, inside=False, classes=classes)
    return path_args

def _get_init_args(frame: types.FrameType) -> Tuple[Optional[Any], Dict[str, Any]]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if "__class__" not in local_vars:
        return None, {}
    cls = local_vars["__class__"]
    init_parameters = inspect.signature(cls.__init__).parameters
    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, "__class__", "frame", "frame_args")
    # only collect variables that appear in the signature
    local_args = {k: local_vars[k] for k in init_parameters}
    # kwargs_var might be None => raised an error by mypy
    if kwargs_var:
        local_args.update(local_args.get(kwargs_var, {}))
    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}
    self_arg = local_vars.get(self_var, None)
    return self_arg, local_args

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