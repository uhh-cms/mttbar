# coding: utf-8
"""
Analysis-wide utility functions
"""
from __future__ import annotations

# NOTE: needs to be added to cf sandbox
import os
import time
from typing import Hashable, Callable
from functools import wraps
import tracemalloc
import math

import law

from columnflow.types import Any
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
psutil = maybe_import("psutil")

_logger = law.logger.get_logger(__name__)


def iter_chunks(*arrays, max_chunk_size):
    """
    Iterate over one or more arrays in chunks of at most `max_chunk_size`.
    If `max_chunked_size` is negative or zero, no chunking is done.
    """

    # validate input sizes
    size = len(arrays[0])
    if any(len(arr) != size for arr in arrays[1:]):
        lengths_str = ", ".join(str(len(arr)) for arr in arrays)
        raise ValueError(f"array length mismatch: {lengths_str}")

    # compute number of chunks
    n_chunks = 1
    if max_chunk_size > 0:
        n_chunks = int(math.ceil(size / max_chunk_size))

    # if no chunking is needed, yield the arrays directly
    if n_chunks == 1:
        yield arrays
        return

    # loop over chunks
    for i_chunk in range(n_chunks):
        end = min(size, (i_chunk + 1) * max_chunk_size)
        slc = slice(i_chunk * max_chunk_size, end)

        yield tuple(a[slc] for a in arrays)


def print_log_msg(
        msg: str,
        print_msg: bool = False,
) -> None:
    """
    Print a log message if `print_msg` is True.
    """
    if print_msg:
        print(msg)
    else:
        # in case of no printing, we could log it to a file or similar
        pass  # Placeholder for future logging implementation


def get_memory_usage():
    """Get current memory usage in MB"""
    # from hbw utils
    if not psutil:
        return "? (psutil not available)"
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory(
    message: str = "",
    unit: str = "MB",
    restart: bool = False,
    logger=None,
    prefer_psutil: bool = True,
):
    # from hbw utils
    if logger is None:
        logger = _logger

    if psutil:
        current_memory = get_memory_usage()
        if unit == "GB":
            current_memory /= 1024
        logger.info(f"Memory after {message}: {current_memory:.3f}{unit}")
        return

    if restart or not tracemalloc.is_tracing():
        logger.info("Start tracing memory")
        tracemalloc.start()

    unit_transform = {
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }[unit]

    current, peak = [x / unit_transform for x in tracemalloc.get_traced_memory()]
    logger.info(f"Memory after {message}: {current:.3f}{unit} (peak: {peak:.3f}{unit})")


def round_sig(
    value: int | float | np.number,
    sig: int = 4,
    convert: Callable | None = None,
) -> int | float | np.number:
    """
    Helper function to round number *value* on *sig* significant digits and
    optionally transform output to type *convert*
    """
    if not np.isfinite(value):
        # cannot round infinite
        _logger.warning("cannot round infinite number")
        return value

    from math import floor, log10

    def try_rounding(_value):
        try:
            n_digits = sig - int(floor(log10(abs(_value)))) - 1
            if convert in (int, np.int8, np.int16, np.int32, np.int64):
                # do not round on decimals when converting to integer
                n_digits = min(n_digits, 0)
            return round(_value, n_digits)
        except Exception:
            _logger.warning(f"Cannot round number {value} to {sig} significant digits. Number will not be rounded")
            return value

    # round first to not lose information from type conversion
    rounded_value = try_rounding(value)

    # convert number if "convert" is given
    if convert not in (None, False):
        try:
            rounded_value = convert(rounded_value)
        except Exception:
            _logger.warning(f"Cannot convert {rounded_value} to {convert.__name__}")
            return rounded_value

        # some types need rounding again after converting (e.g. np.float32 to float)
        rounded_value = try_rounding(rounded_value)

    return rounded_value


def timeit(func):
    """
    Simple wrapper to measure execution time of a function.
    """
    # from hbw utils
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        _logger.info(f"Function '{func.__name__}' done; took {round_sig(total_time)} seconds")
        return result
    return timeit_wrapper


def timeit_multiple(func):
    """ Wrapper to measure the number of execution calls and the added execution time of a function """
    # from hbw utils
    log_method = "info"
    log_func = getattr(_logger, log_method)

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        func.total_calls = getattr(func, "total_calls", 0) + 1
        _repr = func.__name__

        if len(args) >= 1 and hasattr(args[0], "__name__"):
            # some classmethod
            _repr = f"{args[0].__name__}.{_repr}"

            if len(args) >= 2 and isinstance(args[1], dict):
                params = args[1]
            elif len(args) >= 3 and isinstance(args[2], dict):
                params = args[2]
            else:
                params = {}

            for param in ("branch", "dataset"):
                if param in params:
                    _repr = f"{_repr} ({param} {params[param]})"

        elif len(args) >= 1 and hasattr(args[0], "cls_name"):
            # probably a CSP function
            inst = args[0]
            params = {}
            _repr = f"{inst.cls_name}.{_repr}"
            if hasattr(inst, "config_inst"):
                _repr = f"{_repr} ({inst.config_inst.name})"
            if hasattr(inst, "dataset_inst"):
                _repr = f"{_repr} ({inst.dataset_inst.name})"
            if hasattr(inst, "shift_inst"):
                _repr = f"{_repr} ({inst.shift_inst.name})"

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        func.total_time = getattr(func, "total_time", 0) + total_time
        log_func(f"{_repr} has been run {func.total_calls} times ({round_sig(func.total_time)} seconds)")
        return result

    return timeit_wrapper


def make_dict_hashable(d: dict, deep: bool = True):
    """Small helper that converts dict into a hashable representation."""
    # from hbw utils
    d_out = d.copy()
    for key, value in d.items():
        if isinstance(value, Hashable):
            # Skip values that are already hashable
            continue
        elif isinstance(value, dict):
            # Convert nested dictionaries to a hashable form
            if deep:
                value = make_dict_hashable(value)
            d_out[key] = tuple(value)
        else:
            # Convert other types to tuples
            d_out[key] = law.util.make_tuple(value)

    return d_out.items()


def dict_diff(dict1: dict, dict2: dict):
    """Return the differences between two dictionaries."""
    # from hbw utils
    set1 = set(make_dict_hashable(dict1))
    set2 = set(make_dict_hashable(dict2))

    return set1 ^ set2


def call_func_safe(func, *args, **kwargs) -> Any:
    """
    Small helper to make sure that our training does not fail due to plotting
    """

    # get the function name without the possibility of raising an error
    try:
        func_name = func.__name__
    except Exception:
        # default to empty name
        func_name = ""

    t0 = time.perf_counter()

    try:
        outp = func(*args, **kwargs)
        _logger.info(f"Function '{func_name}' done; took {(time.perf_counter() - t0):.2f} seconds")
    except Exception as e:
        _logger.warning(f"Function '{func_name}' failed due to {type(e)}: {e}")
        outp = None

    return outp
