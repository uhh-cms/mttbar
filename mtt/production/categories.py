# coding: utf-8

"""
Selectors related to defining category masks following the froduction step.
"""
from functools import partial

from columnflow.columnar_util import Route
from columnflow.util import maybe_import
from columnflow.selection import Selector, selector

from mtt.selection.util import make_selector_not, make_selector_range
from mtt.production.ttbar_reco import ttbar

np = maybe_import("numpy")
ak = maybe_import("awkward")


# -- basic selectors for event categorization

# pass/fail chi2 criterion
@selector(uses={"TTbar.chi2"})
def sel_chi2pass(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """Select only events with a chi2 lower than a configured threshold."""
    chi2_max = self.config_inst.x.categorization.chi2_max
    return Route("TTbar.chi2").apply(events) < chi2_max


sel_chi2fail = make_selector_not("sel_chi2fail", sel_chi2pass)


# |cos(theta*)| regions
make_selector_range_acts = partial(
    make_selector_range,
    route="TTbar.cos_theta_star",
    route_func=abs,
    uses={"TTbar.cos_theta_star"},
)
# TODO: make configurable
acts_bins = [0.0, 0.5, 0.7, 0.9, 1.0]
acts_names = ["0_5", "5_7", "7_9", "9_1"]
sels_acts = tuple(
    make_selector_range_acts(
        name=f"sel_acts_{acts_name}",
        min_val=min_val,
        max_val=max_val,
    )
    for min_val, max_val, acts_name in zip(acts_bins[:-1], acts_bins[1:], acts_names)
)
