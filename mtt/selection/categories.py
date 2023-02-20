# coding: utf-8

"""
Selection methods defining masks for categories.
"""
import itertools

from columnflow.util import maybe_import
from columnflow.columnar_util import Route
from columnflow.selection import Selector, selector

from mtt.selection.util import make_selector_and
from mtt.selection.cutflow_features import cutflow_features


np = maybe_import("numpy")
ak = maybe_import("awkward")


# -- basic selectors

@selector(uses={"event"})
def sel_incl(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """Passes every event."""
    return ak.ones_like(events.event, dtype=bool)


@selector(uses={"event", "channel_id"})
def sel_1m(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events in the muon channel."""
    ch = self.config_inst.get_channel("mu")
    return events["channel_id"] == ch.id


@selector(uses={"event", "channel_id"})
def sel_1e(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events in the electron channel."""
    ch = self.config_inst.get_channel("e")
    return events["channel_id"] == ch.id


@selector(uses={"cutflow.n_toptag"})
def sel_0t(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events with zero top-tagged fat jets."""
    return Route("cutflow.n_toptag").apply(events) == 0


@selector(uses={"cutflow.n_toptag"})
def sel_1t(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events with exactly one top-tagged fat jet."""
    return Route("cutflow.n_toptag").apply(events) == 1

