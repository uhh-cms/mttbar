# coding: utf-8

"""
Useful selection methods.
"""
from typing import Optional

from columnflow.columnar_util import Route, TaskArrayFunction
from columnflow.util import maybe_import
from columnflow.selection import Selector, selector

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


# -- helper functions for constructing selectors

def make_selector_not(name: str, input_selector: Selector):
    """
    Construct a selector that corresponds to the logical *NOT* of a given input selector.
    """
    @selector(cls_name=name, uses={input_selector})
    def sel(self: Selector, events: ak.Array, **kwargs) -> ak.Array:

        # ensure dependencies are present
        for dep in self.uses:
            if not isinstance(dep, TaskArrayFunction):
                continue
            events = self[dep](events, **kwargs)

        # return logical NOT of input selector mask
        return ~self[input_selector](events, **kwargs)

    return sel


def make_selector_and(name: str, selectors: set[Selector]):
    """
    Construct a selector that corresponds to the logical *AND* of a set of dependent selectors.
    """

    @selector(cls_name=name, uses=set(selectors))
    def sel(self: Selector, events: ak.Array, **kwargs) -> ak.Array:

        # ensure dependencies are present
        for dep in self.uses:
            if not isinstance(dep, TaskArrayFunction):
                continue
            events = self[dep](events, **kwargs)

        # return logical AND of all input selector masks
        return ak.all(
            ak.concatenate(
                [ak.singletons(self[s](events, **kwargs)) for s in self.uses],
                axis=1,
            ),
            axis=1,
        )

    return sel


def make_selector_range(
    name: str,
    route: str,
    min_val: float,
    max_val: float,
    route_func: Optional[callable] = None,
    **decorator_kwargs,
):
    """
    Construct a selector that evaluates to *True* whenever the value of the specified *route*
    lies between *min_val* and *max_val*. If supplied, an *route_func* is applied to the route
    value before performing the comparison.
    """

    route_func_name = getattr(route_func, __name__, "<lambda>")
    route_repr = f"{route_func_name}({route})" if route_func else route

    @selector(cls_name=name, **decorator_kwargs)
    def sel(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
        f"""Select only events where value of {route_repr} is in range ({min_val}, {max_val})."""

        # ensure dependencies are present
        for dep in self.uses:
            if not isinstance(dep, TaskArrayFunction):
                continue
            events = self[dep](events, **kwargs)

        # calculate route value
        val = Route(route).apply(events)
        if route_func:
            val = route_func(val)

        # return selection mask
        return (min_val <= val) & (val < max_val)

    return sel
