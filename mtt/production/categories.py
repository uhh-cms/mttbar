# coding: utf-8

"""
Categorizers related to defining category masks following the production step.
"""
from functools import partial

from columnflow.columnar_util import Route
from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer

from mtt.categorization.util import make_categorizer_not, make_categorizer_range

np = maybe_import("numpy")
ak = maybe_import("awkward")


# -- basic categorizers for event categorization

# pass/fail chi2 criterion
@categorizer(uses={"TTbar.chi2"})
def sel_chi2pass(
    self: Categorizer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """Select only events with a chi2 lower than a configured threshold."""
    chi2_max = self.config_inst.x.categorization.chi2_max
    mask = Route("TTbar.chi2").apply(events) < chi2_max
    return events, mask


sel_chi2fail = make_categorizer_not("sel_chi2fail", sel_chi2pass)


# |cos(theta*)| regions
make_categorizer_range_acts = partial(
    make_categorizer_range,
    route="TTbar.cos_theta_star",
    route_func=abs,
    uses={"TTbar.cos_theta_star"},
)
# TODO: make configurable
acts_bins = [0.0, 0.5, 0.7, 0.9, 1.0]
acts_names = ["0_5", "5_7", "7_9", "9_1"]
sels_acts = tuple(
    make_categorizer_range_acts(
        name=f"sel_acts_{acts_name}",
        min_val=min_val,
        max_val=max_val,
    )
    for min_val, max_val, acts_name in zip(acts_bins[:-1], acts_bins[1:], acts_names)
)
