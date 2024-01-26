import os
import signal
import subprocess
import sys
import time
from typing import List

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