# coding: utf-8
"""
Analysis-wide utility functions
"""

import tracemalloc
from time import perf_counter

from law.util import human_duration, human_bytes


class Timer:
    """
    Provides a context for timed execution of tasks.

    To use, instantiate a new *Timer* object in a `with` block
    and wrap the task to be executed:
    ```
    with Timer() as t:
        do_something()

    print(f"Execution took {t.human_duration}"
    ```
    """
    def __enter__(self):
        # don't stop tracing on exit it it was enabled on on enter
        self._stop_tracing_on_exit = True
        if tracemalloc.is_tracing():
            self._stop_tracing_on_exit = False
        else:
            tracemalloc.start()
        self.mem_start, self.mem_peak = tracemalloc.get_traced_memory()
        self.time_start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time_stop = perf_counter()
        self.mem_stop, self.mem_peak = tracemalloc.get_traced_memory()
        if self._stop_tracing_on_exit:
            tracemalloc.stop()

    @property
    def duration(self):
        return self.time_stop - self.time_start

    @property
    def mem_diff(self):
        return self.mem_stop - self.mem_start

    @property
    def human_duration(self):
        return human_duration(seconds=self.duration)

    @property
    def human_mem_start(self):
        return human_bytes(self.mem_start, fmt=True)

    @property
    def human_mem_stop(self):
        return human_bytes(self.mem_stop, fmt=True)

    @property
    def human_mem_diff(self):
        return human_bytes(self.mem_diff, fmt=True)

    @property
    def human_mem_peak(self):
        return human_bytes(self.mem_peak, fmt=True)

    @property
    def human_mem(self):
        return f"{self.human_mem_start} -> {self.human_mem_stop} ({self.human_mem_diff} diff, {self.human_mem_peak} peak)"
