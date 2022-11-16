# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.electron import electron_weights
from columnflow.production.mc_weight import mc_weight
from columnflow.production.normalization import normalization_weights
from columnflow.production.pileup import pu_weight
from columnflow.util import maybe_import

from mtt.production.features import features

ak = maybe_import("awkward")


@producer(
    uses={
        features, category_ids, normalization_weights, pu_weight, mc_weight,
        electron_weights, muon_weights,
    },
    produces={
        features, category_ids, normalization_weights, pu_weight, mc_weight,
        electron_weights, muon_weights,
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    events = self[features](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # compute normalization weights
    events = self[normalization_weights](events, **kwargs)

    # compute MC weights
    events = self[mc_weight](events, **kwargs)

    # compute pu weights
    events = self[pu_weight](events, **kwargs)

    # electron sf weights
    events = self[electron_weights](events, **kwargs)

    # muon sf weights
    events = self[muon_weights](events, **kwargs)

    return events
