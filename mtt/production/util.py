# coding: utf8

"""
collection of useful function for producers
"""
import itertools

from functools import partial

from columnflow.util import maybe_import

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")


#
# general awkward array functions
#

def ak_extract_fields(arr, fields, **kwargs):
    """
    Build an array containing only certain `fields` of an input array `arr`,
    preserving behaviors.
    """
    # reattach behavior
    if "behavior" not in kwargs:
        kwargs["behavior"] = arr.behavior

    return ak.zip(
        {
            field: getattr(arr, field)
            for field in fields
        },
        **kwargs,
    )


def ak_argcartesian(*arrs, as_type=np.uint8):
    """
    Like `ak.argcartesian`, but allows specifying a custom
    type for the index instead of the default `np.uint64`.
    """
    return tuple(
        ak.values_astype(item, as_type)
        for item in ak.unzip(ak.argcartesian(arrs))
    )


def ak_arg_grouped_combinations(
    array: ak.Array,
    group_sizes: list[int],
    axis: int = 1,
    cache: dict[int, tuple[ak.Array]] = None,
    as_type=np.uint8,
) -> tuple[tuple[ak.Array]]:
    """
    Like `ak.argcombinations`, but considers all possible ways to arrange
    the entries of an input `array` into non-overlapping groups with specified
    sizes `group_sizes`.

    Optionally, an `axis` along which to build the combinations can be specified
    (default is `1`).

    An optional `cache` dictionary with pre-computed index combinations ca
    nbe given.

    Returns a nested tuple of unzipped index arrrays (`combinations`) organized
    by group index and object position within the group. For example,
    `combinations[0][2]` yields the index of the third object in the first
    group.

    As an example, consider an array with four objects:

    >>> arr = ak.Array([[10, 31, 25, 19]])
    <Array [[10, 31, 25, 19]] type='1 * var * int64'>

    We can obtain all possible combinations of objects arranged into two groups
    with sizes 1 and 2:

    >>> combs = ak_arg_grouped_combinations(arr, [1, 2], axis=1)
    ((<Array [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]] type='1 * var * int64'>,),
     (<Array [[1, 1, 2, 0, 0, 2, 0, 0, 1, 0, 0, 1]] type='1 * var * int64'>,
      <Array [[2, 3, 3, 2, 3, 3, 1, 3, 3, 1, 2, 2]] type='1 * var * int64'>))

    The resulting arrays can be used to index the original array. For example,
    we can sum over all elements of the second group like this:

    >>> arr[combs[1][0]] + arr[combs[1][1]]
    <Array [[56, 50, 44, 35, 29, ..., 29, 50, 41, 35, 56]] type='1 * var * int64'>
    """

    if cache is None:
        cache = {}

    n_groups = len(group_sizes)
    if n_groups < 1:
        raise ValueError("at least one group size is required")

    # build combinations for each individual group size
    for n in group_sizes:
        if n not in cache:
            cache[n] = tuple(
                ak.values_astype(item, as_type)
                for item in ak.unzip(
                    ak.argcombinations(array, n, axis=axis),
                )
            )

    combs = [cache[n] for n in group_sizes]

    # build cartesian product of all different-size combinations
    comb_prod = tuple(
        ak.values_astype(item, as_type)
        for item in ak.unzip(
            ak.argcartesian([c[0] for c in combs]),
        )
    )

    # -- exclude combinations with a common index across any two groups

    keep = ak.ones_like(comb_prod[0], dtype=bool)
    # loop over all configurations of object indices within groups
    for obj_idxs in itertools.product(*(range(n) for n in group_sizes)):
        # loop over all combinations of two groups
        for g_idx_1, g_idx_2 in itertools.product(range(n_groups), range(n_groups)):
            # skip lower triangle
            if g_idx_1 >= g_idx_2:
                continue
            # drop cases where the same index is contained in two groups
            keep = keep & (
                combs[g_idx_1][obj_idxs[g_idx_1]][comb_prod[g_idx_1]] !=
                combs[g_idx_2][obj_idxs[g_idx_2]][comb_prod[g_idx_2]]
            )

    # apply filter to combination product
    comb_prod_filtered = tuple(p[keep] for p in comb_prod)

    # build index arrays from combinations product and return
    grouped_combs = tuple(
        tuple(
            ak.values_astype(
                c[comb_prod_filtered[g_idx]],
                as_type,
            )
            for c in combs[g_idx]
        )
        for g_idx in range(n_groups)
    )

    return grouped_combs


#
# functions for operating on lorentz vectors
#

_lv_base = partial(ak_extract_fields, behavior=coffea.nanoevents.methods.nanoaod.behavior)

lv_xyzt = partial(_lv_base, fields=["x", "y", "z", "t"], with_name="LorentzVector")
lv_xyzt.__doc__ = """Construct a `LorentzVectorArray` from an input array."""

lv_mass = partial(_lv_base, fields=["pt", "eta", "phi", "mass"], with_name="PtEtaPhiMLorentzVector")
lv_mass.__doc__ = """Construct a `PtEtaPhiMLorentzVectorArray` from an input array."""

lv_energy = partial(_lv_base, fields=["pt", "eta", "phi", "energy"], with_name="PtEtaPhiELorentzVector")
lv_energy.__doc__ = """Construct a `PtEtaPhiELorentzVectorArray` from an input array."""


def lv_sum(lv_arrays):
    """
    Return the sum of identically-structured arrays containing Lorentz vectors.
    """
    # don't use `reduce` or list comprehensions
    # to keep memory use as low as pposible
    tmp_lv_sum = None
    for lv in lv_arrays:
        if tmp_lv_sum is None:
            tmp_lv_sum = lv
        else:
            tmp_lv_sum = tmp_lv_sum + lv

    return tmp_lv_sum


#
# functions for matching between collections of Lorentz vectors
#

def delta_r_match(dst_lvs, src_lv, max_dr=None):
    """
    Match entries in the source array `src_lv` to the closest entry
    in the destination array `dst_lvs` using delta-R as a metric.

    The array `src_lv` should contain a single entry per event and
    `dst_lvs` should be a list of possible matches.

    The parameter `max_dr` optionally indicates the maximum possible
    delta-R value for a match (if the best possible match has a higher
    value, it is not considered a valid match).

    Returns an array containing the best match in `dst_lvs` per event,
    and a view of `dst_lvs` with the matches removed.
    """
    # calculate delta_r for all possible src-dst pairs
    delta_r = ak.singletons(src_lv).metric_table(dst_lvs)

    # invalidate entries above delta_r threshold
    if max_dr is not None:
        delta_r = ak.mask(delta_r, delta_r < max_dr)

    # get index and value of best match
    best_match_idx = ak.argmin(delta_r, axis=2)
    best_match_dst_lv = ak.firsts(dst_lvs[best_match_idx])

    # filter dst_lvs to remove the best matches (if any)
    keep = (ak.local_index(dst_lvs, axis=1) != ak.firsts(best_match_idx))
    keep = ak.fill_none(keep, True)
    dst_lvs = ak.mask(dst_lvs, keep)
    dst_lvs = ak.where(ak.is_none(dst_lvs, axis=0), [[]], dst_lvs)

    return best_match_dst_lv, dst_lvs


def delta_r_match_multiple(dst_lvs, src_lvs, max_dr=None):
    """
    Like `delta_r_match`, except source array `src_lvs` can contain more than
    one entry per event. The matching is done sequentially for each entry in
    `src_lvs`, with previous matches being filetered from the destination array
    each time to prevent double counting.
    """

    # save the index structure of the supplied source array
    src_lvs_idx = ak.local_index(src_lvs, axis=1)

    # pad sub-lists to the same length (but at least 1)
    max_num = max(1, ak.max(ak.num(src_lvs)))
    src_lvs = ak.pad_none(src_lvs, max_num)

    # run matching for each position,
    # filtering the destination array each time
    best_match_dst_lvs = []
    for i in range(max_num):
        best_match_dst_lv, dst_lvs = delta_r_match(dst_lvs, src_lvs[:, i], max_dr=max_dr)
        best_match_dst_lv = ak.unflatten(best_match_dst_lv, 1)
        best_match_dst_lvs.append(best_match_dst_lv)

    # concatenate matching results
    result = ak.concatenate(best_match_dst_lvs, axis=-1)

    # remove padding to make result index-compatible with input
    result = result[src_lvs_idx]

    return result, dst_lvs
