# coding: utf-8
"""
Tools for profiling task execution
"""
import gc
import tracemalloc
import time
import textwrap

from contextlib import AbstractContextManager
from enum import Enum, auto, unique
from law.util import human_duration, human_bytes


def property_with_default(default_value=None):
    """
    Property decorator, defaulting to a used-defined value if attribute
    lookup fails.
    """
    def inner(f):
        @property
        def wrapper(self, *args, **kwargs):
            try:
                return f(self, *args, **kwargs)
            except AttributeError:
                return default_value
        return wrapper
    return inner


class ProfilerBaseMeta(type(AbstractContextManager)):
    """
    Metaclass for profilers: registers mixin methods
    to be run on entering or exiting execution context
    """
    def __new__(mcl, name, bases, nmspc):
        cls = super(ProfilerBaseMeta, mcl).__new__(mcl, name, bases, nmspc)

        # register hook methods
        cls.__hook_methods__ = {}
        for method_name in cls.__hooks__:
            cls.__hook_methods__[method_name] = [
                (base.__enable_if__, report_method)
                for base in bases
                if (report_method := getattr(base, method_name, None)) is not None
            ]

        return cls


class ProfilerBase(AbstractContextManager, metaclass=ProfilerBaseMeta):
    """
    Base context manager for executing tasks and collecting
    performance metrics. Add metrics by inheriting from respecting
    mixin classes.
    """

    __hooks__ = ("enter", "exit", "report")

    @unique
    class State(Enum):
        INIT = auto()
        ENTER = auto()
        EXIT = auto()
        ERROR = auto()

    def __init__(self, *args, **kwargs):
        if args:
            raise ValueError("unexpected positional args")
        if kwargs:
            raise ValueError(f"unexpected keyword args: {list(kwargs)}")
        self._state = self.__class__.State.INIT

    def __enter__(self):
        for enable_if, method in self.__hook_methods__["enter"]:
            if getattr(self, enable_if, None):
                method(self)
        self._state = self.__class__.State.ENTER
        return super().__enter__()

    def __exit__(self, type, value, traceback):
        super().__exit__(type, value, traceback)
        for enable_if, method in self.__hook_methods__["exit"][::-1]:
            if getattr(self, enable_if, None):
                method(self)
        self._state = self.__class__.State.EXIT

    def collect_reports(self):
        return [
            method(self)
            for enable_if, method in self.__hook_methods__["report"]
            if getattr(self, enable_if, None)
        ]


class MemoryMixin:
    """
    Profiler mixin to measure memory use of wrapped tasks.
    """
    __enable_if__ = "prof_mem"

    def __init__(self, *args, prof_mem=True, **kwargs):
        self.prof_mem = prof_mem
        super().__init__(*args, **kwargs)

    def enter(self):
        # start memory tracing
        self._stop_tracing_on_exit = True
        if tracemalloc.is_tracing():
            # don't stop tracing on exit if it was enabled on on enter
            self._stop_tracing_on_exit = False
        else:
            tracemalloc.start()

        # record initial memory state
        self.mem_start, self.mem_peak = tracemalloc.get_traced_memory()

    def exit(self):
        # record final memory state
        self.mem_stop, self.mem_peak = tracemalloc.get_traced_memory()
        if self._stop_tracing_on_exit:
            tracemalloc.stop()

    def report(self):
        return (
            f"memory start: {self.human_mem_start}\n"
            f"memory end:   {self.human_mem_stop}\n"
            f"memory diff:  {self.human_mem_diff}\n"
            f"memory peak:  {self.human_mem_peak}"
        ) if self.prof_mem else None

    @property_with_default(None)
    def mem_diff(self):
        return self.mem_stop - self.mem_start

    @property
    def human_mem_start(self):
        if self.mem_start is not None:
            return human_bytes(self.mem_start, fmt=True)

    @property
    def human_mem_stop(self):
        if self.mem_stop is not None:
            return human_bytes(self.mem_stop, fmt=True)

    @property
    def human_mem_diff(self):
        if self.mem_diff is not None:
            return human_bytes(self.mem_diff, fmt=True)

    @property
    def human_mem_peak(self):
        if self.mem_peak is not None:
            return human_bytes(self.mem_peak, fmt=True)


class DurationMixin:
    """
    Profiler mixin to measure duration of wrapped tasks.
    """
    __enable_if__ = "prof_time"

    def __init__(self, *args, prof_time=True, **kwargs):
        self.prof_time = prof_time
        super().__init__(*args, **kwargs)

    def enter(self):
        self.time_start = time.perf_counter()

    def exit(self):
        self.time_stop = time.perf_counter()

    def report(self):
        return f"time:         {self.human_duration}" if self.prof_time else None

    @property_with_default(None)
    def duration(self):
        return self.time_stop - self.time_start

    @property
    def human_duration(self):
        if self.duration is None:
            return "<not available>"
        return human_duration(seconds=self.duration)


class GCMixin:
    """
    Profiler mixin to run garbage collection after task completion.
    """
    __enable_if__ = "gc_on_exit"

    def __init__(self, *args, gc_on_exit=True, **kwargs):
        self.gc_on_exit = gc_on_exit
        super().__init__(*args, **kwargs)

    def enter(self):
        pass

    def exit(self):
        gc.collect()


class TaskReportMixin:
    """
    Profiler mixin to emit reports on execution status
    (start, end, final report)
    """
    __enable_if__ = "msg_func"

    def __init__(self, *args, task_name=None, msg_func=None, indent_str=None, n_cols_text=80, **kwargs):
        self.msg_func = msg_func
        self.task_name = task_name or "<unnamed>"
        self.indent_str = indent_str or ""
        self.n_cols_text = n_cols_text
        super().__init__(*args, **kwargs)

    def enter(self):
        self._maybe_emit(
            f"-- started task:  {self.task_name} --".ljust(self.n_cols_text, "-"),
        )

    def exit(self):
        # emit reports collected from mixins
        for r in self.collect_reports():
            if not r:
                continue
            self._maybe_emit(textwrap.indent(r, "> "))

        self._maybe_emit(
            f"== finished task: {self.task_name} ==".ljust(self.n_cols_text, "="),
        )

    def _maybe_emit(self, msg):
        if self.msg_func is None:
            return

        # format message
        msg = textwrap.indent(
            msg,
            prefix=self.indent_str,
        )

        # emit message
        self.msg_func(msg)


# -- standard profiler class with all mixins

class Profiler(
    TaskReportMixin,
    MemoryMixin,
    DurationMixin,
    GCMixin,
    ProfilerBase,
):
    pass
