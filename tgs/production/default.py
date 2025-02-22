# coding: utf-8

"""
Column production methods related to higher-level features
for the trigger study analysis.
"""


from columnflow.production import Producer, producer
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.categories import category_ids
from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

# use weight producer from main mtt analysis
from mtt.production.weights import weights

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        # nano columns
        "Jet.pt",
    },
    produces={
        # new columns
        "ht", "n_jet",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=1), value_type=np.int32)

    return events


@producer(
    uses={
        mc_weight, category_ids,
        # nano columns
        "Jet.pt",
        "Electron.pt",
        "Muon.pt",
    },
    produces={
        mc_weight, category_ids,
        # new columns
        "cutflow.jet1_pt",
        "cutflow.electron_pt",
        "cutflow.muon_pt",
    },
)
def cutflow_features(
    self: Producer,
    events: ak.Array,
    object_masks: dict[str, dict[str, ak.Array]],
    **kwargs,
) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # apply object masks and create new collections
    reduced_events = create_collections_from_masks(events, object_masks)

    # create category ids per event and add categories back to the
    events = self[category_ids](reduced_events, target_events=events, **kwargs)

    # add cutflow columns
    events = set_ak_column(
        events,
        "cutflow.jet1_pt",
        Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT),
    )
    events = set_ak_column(
        events,
        "cutflow.electron_pt",
        Route("Electron.pt[:,0]").apply(events, EMPTY_FLOAT),
    )
    events = set_ak_column(
        events,
        "cutflow.muon_pt",
        Route("Muon.pt[:,0]").apply(events, EMPTY_FLOAT),
    )

    return events


@producer(
    uses={
        features, category_ids, weights,
    },
    produces={
        features, category_ids, weights,
    },
)
def default_tgs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    # features
    events = self[features](events, **kwargs)

    # weights
    events = self[weights](events, **kwargs)

    # category ids
    events = self[category_ids](events, **kwargs)

    return events
