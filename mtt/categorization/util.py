# coding: utf-8

"""
Useful selection methods.
"""
from typing import Optional

from columnflow.columnar_util import Route, TaskArrayFunction
from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer

np = maybe_import("numpy")
ak = maybe_import("awkward")


# -- helper functions for constructing categorizers

def make_categorizer_not(name: str, input_categorizer: Categorizer):
    """
    Construct a categorizer that corresponds to the logical *NOT* of a given input categorizer.
    """
    @categorizer(cls_name=name, uses={input_categorizer})
    def cat(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:

        # ensure dependencies are present
        for dep in self.uses:
            if not isinstance(dep, TaskArrayFunction):
                continue
            events = self[dep](events, **kwargs)

        # return logical NOT of input categorizer mask
        events, input_mask = self[input_categorizer](events, **kwargs)
        return events, ~input_mask

    return cat


def make_categorizer_and(name: str, categorizers: set[Categorizer]):
    """
    Construct a categorizer that corresponds to the logical *AND* of a set of dependent categorizers.
    """

    @categorizer(cls_name=name, uses=set(categorizers))
    def cat(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:

        # ensure dependencies are present
        for dep in self.uses:
            if not isinstance(dep, TaskArrayFunction):
                continue
            events = self[dep](events, **kwargs)

        input_masks = [
            self[input_categorizer](events, **kwargs)[1]
            for input_categorizer in self.uses
        ]

        # return logical AND of all input categorizer masks
        mask = ak.all(
            ak.concatenate(
                [ak.singletons(mask) for mask in input_masks],
                axis=1,
            ),
            axis=1,
        )
        return events, mask

    return cat


def make_categorizer_range(
    name: str,
    route: str,
    min_val: float,
    max_val: float,
    route_func: Optional[callable] = None,
    **decorator_kwargs,
):
    """
    Construct a categorizer that evaluates to *True* whenever the value of the specified *route*
    lies between *min_val* and *max_val*. If supplied, an *route_func* is applied to the route
    value before performing the comparison.
    """

    route_func_name = getattr(route_func, __name__, "<lambda>")
    route_repr = f"{route_func_name}({route})" if route_func else route

    @categorizer(cls_name=name, **decorator_kwargs)
    def cat(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
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
        mask = (min_val <= val) & (val < max_val)
        return events, mask

    return cat
