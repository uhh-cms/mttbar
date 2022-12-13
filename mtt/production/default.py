# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.btag import btag_weights
from columnflow.production.categories import category_ids
from columnflow.production.electron import electron_weights
from columnflow.production.mc_weight import mc_weight
from columnflow.production.muon import muon_weights
from columnflow.production.normalization import normalization_weights
from columnflow.production.pileup import pu_weight
from columnflow.util import maybe_import

from mtt.production.features import features

ak = maybe_import("awkward")


@producer(
    uses={
        features, category_ids, normalization_weights, pu_weight, mc_weight,
        electron_weights, muon_weights, btag_weights,
    },
    produces={
        features, category_ids, normalization_weights, pu_weight, mc_weight,
        electron_weights, muon_weights, btag_weights,
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    events = self[features](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # compute electron weights
    electron_mask = (events.Electron.pt >= 35)
    events = self[electron_weights](events, electron_mask=electron_mask, **kwargs)

    # compute muon weights
    muon_mask = (events.Muon.pt >= 30) & (abs(events.Muon.eta) < 2.4)
    events = self[muon_weights](events, muon_mask=muon_mask, **kwargs)

    # compute btag weights
    jet_mask = (events.Jet.pt >= 100)
    events = self[btag_weights](events, jet_mask=jet_mask, **kwargs)

    # compute normalization weights
    events = self[normalization_weights](events, **kwargs)

    # compute MC weights
    events = self[mc_weight](events, **kwargs)

    # compute pu weights
    events = self[pu_weight](events, **kwargs)

    return events
