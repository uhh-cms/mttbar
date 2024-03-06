# coding: utf-8

"""
Selection reducing spiking QCD behavior.
"""

from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        attach_coffea_behavior,
        "Jet.pt",
        "LHE.HT",
    },
    exposed=True,
)
def qcd_spikes(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    sel_jets = ak.fill_none(events.LHE.HT > ak.firsts(events.Jet.pt), True)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "QCDSpikes": sel_jets,
        },
    )
