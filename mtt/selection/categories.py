# coding: utf-8

"""
Selection methods defining masks for categories.
"""
from columnflow.util import maybe_import
from columnflow.columnar_util import Route
from columnflow.categorization import Categorizer, categorizer

np = maybe_import("numpy")
ak = maybe_import("awkward")


# -- basic categorizers

@categorizer(uses={"event"})
def sel_incl(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    """Passes every event."""
    mask = ak.ones_like(events.event, dtype=bool)
    return events, mask


@categorizer(uses={"event", "channel_id"})
def sel_1m(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events in the muon channel."""
    ch = self.config_inst.get_channel("mu")
    mask = events["channel_id"] == ch.id
    return events, mask


@categorizer(uses={"event", "channel_id"})
def sel_1e(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events in the electron channel."""
    ch = self.config_inst.get_channel("e")
    mask = events["channel_id"] == ch.id
    return events, mask


@categorizer(uses={"cutflow.n_toptag_delta_r_lepton"})
def sel_0t(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events with zero top-tagged fat jets."""
    mask = (Route("cutflow.n_toptag_delta_r_lepton").apply(events) == 0)
    return events, mask


@categorizer(uses={"cutflow.n_toptag_delta_r_lepton"})
def sel_1t(self: Categorizer, events: ak.Array, **kwargs) -> ak.Array:
    """Select only events with exactly one top-tagged fat jet."""
    mask = (Route("cutflow.n_toptag_delta_r_lepton").apply(events) == 1)
    return events, mask
