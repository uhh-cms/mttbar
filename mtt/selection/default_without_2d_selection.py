# coding: utf-8

"""
Default selection without jet lepton 2D selection.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.selection.cms.met_filters import met_filters
from columnflow.selection.cms.json_filter import json_filter
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.processes import process_ids

from mtt.selection.general import jet_energy_shifts
from mtt.selection.lepton import lepton_selection
from mtt.selection.cutflow_features import cutflow_features
from mtt.selection.jets import jet_selection, top_tagged_jets
from mtt.selection.jets import met_selection
from mtt.selection.qcd_spikes import qcd_spikes
from mtt.selection.data_trigger_veto import data_trigger_veto

from mtt.production.gen_top import gen_parton_top
from mtt.production.gen_v import gen_v_boson

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        jet_selection, lepton_selection, met_selection, top_tagged_jets,
        qcd_spikes,
        data_trigger_veto,
        cutflow_features,
        category_ids,
        process_ids, increment_stats, attach_coffea_behavior,
        mc_weight,
        met_filters,
        gen_parton_top,
        gen_v_boson,
        json_filter,
    },
    produces={
        jet_selection, lepton_selection, met_selection, top_tagged_jets,
        qcd_spikes,
        data_trigger_veto,
        cutflow_features,
        category_ids,
        process_ids, increment_stats, attach_coffea_behavior,
        mc_weight,
        met_filters,
        gen_parton_top,
        gen_v_boson,
        json_filter,
    },
    shifts={
        jet_energy_shifts,
    },
    exposed=True,
)
def default_without_2d_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # MET filters
    results.steps.METFilters = self[met_filters](events, **kwargs)

    # JSON filter (data-only)
    if self.dataset_inst.is_data:
        results.steps.JSON = self[json_filter](events, **kwargs)

    # lepton selection
    events, lepton_results = self[lepton_selection](events, **kwargs)
    results += lepton_results

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # met selection
    events, met_results = self[met_selection](events, **kwargs)
    results += met_results

    # all-hadronic veto
    events, top_tagged_jets_results = self[top_tagged_jets](events, **kwargs)
    results += top_tagged_jets_results

    if self.dataset_inst.has_tag("is_qcd"):
        events, qcd_sel_results = self[qcd_spikes](events, **kwargs)
        results += qcd_sel_results

    if not self.dataset_inst.is_mc:
        events, trigger_veto_results = self[data_trigger_veto](events, **kwargs)
        results += trigger_veto_results

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel

    for step, sel in results.steps.items():
        n_sel = ak.sum(sel, axis=-1)
        print(f"{step}: {n_sel}")

    n_sel = ak.sum(event_sel, axis=-1)
    print(f"__all__: {n_sel}")

    # produce features relevant for selection and event weights
    if self.dataset_inst.has_tag("is_sm_ttbar"):
        events = self[gen_parton_top](events, **kwargs)

    if self.dataset_inst.has_tag("is_v_jets"):
        events = self[gen_v_boson](events, **kwargs)

    # add cutflow features
    events = self[cutflow_features](events, results=results, **kwargs)

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add mc weights (needed for cutflow plots)
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        "num_events_selected": results.event,
    }
    group_map = {
        # per category
        "category": {
            "values": events.category_ids,
            "mask_fn": (lambda v: ak.any(events.category_ids == v, axis=1)),
        },
        # per step
        "step": {
            "values": list(results.steps),
            "mask_fn": (lambda v: results.steps[v]),
        },
    }
    if self.dataset_inst.is_mc:
        weight_map = {
            **weight_map,
            # mc weight for all events
            "sum_mc_weight": (events.mc_weight, Ellipsis),
            "sum_mc_weight_selected": (events.mc_weight, results.event),
        }
        group_map = {
            **group_map,
            # per process
            "process": {
                "values": events.process_id,
                "mask_fn": (lambda v: events.process_id == v),
            },
        }
    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        **kwargs,
    )

    return events, results


@default_without_2d_selection.init
def default_init(self: Selector) -> None:

    if hasattr(self, "dataset_inst") and self.dataset_inst.has_tag("is_qcd"):
        self.uses |= {qcd_spikes}
        self.produces |= {qcd_spikes}

    if hasattr(self, "dataset_inst") and not self.dataset_inst.is_mc:
        self.uses |= {data_trigger_veto}

        self.produces |= {data_trigger_veto}
