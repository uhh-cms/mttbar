# coding: utf-8

"""
Selection reducing spiking QCD behavior.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.met_filters import met_filters
from columnflow.selection.cms.json_filter import json_filter
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.processes import process_ids

from mtt.selection.util import masked_sorted_indices
from mtt.selection.general import increment_stats, jet_energy_shifts
from mtt.selection.lepton import lepton_selection
from mtt.selection.cutflow_features import cutflow_features
from mtt.selection.early import check_early

from mtt.production.lepton import choose_lepton
from mtt.production.gen_top import gen_parton_top
from mtt.production.gen_v import gen_v_boson

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

