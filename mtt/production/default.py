# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.util import maybe_import

from mtt.production.features import features
from mtt.production.weights import weights

ak = maybe_import("awkward")


@producer(
    uses={
        features, category_ids, weights,
    },
    produces={
        features, category_ids, weights,
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # features
    events = self[features](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    # weights
    events = self[weights](events, **kwargs)

    return events
