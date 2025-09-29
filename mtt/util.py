# coding: utf-8
"""
Analysis-wide utility functions
"""
import math


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
