# coding: utf-8

"""
Useful selection methods.
"""
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


def masked_sorted_indices(
    mask: ak.Array,
    sort_var: ak.Array,
    ascending: bool = False,
) -> ak.Array:
    """
    Return the indices that would sort *sort_var*, dropping the ones for which the
    corresponding *mask* is False.
    """
    # get indices that would sort the `sort_var` array
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]
