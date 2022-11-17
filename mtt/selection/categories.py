# coding: utf-8

"""
Selection methods defining masks for categories.
"""

from columnflow.util import maybe_import
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column

#from mtt.selection.default import muon_selection, electron_selection
from mtt.selection.lepton import lepton_selection

np = maybe_import("numpy")
ak = maybe_import("awkward")


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
