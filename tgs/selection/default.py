# coding: utf-8

"""
Selectors for m(ttbar) trigger study.
"""
from __future__ import annotations

from collections import defaultdict
from functools import reduce
from operator import and_

from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.stats import increment_stats
from columnflow.columnar_util import sorted_indices_from_mask
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.processes import process_ids

from tgs.production.default import cutflow_features


np = maybe_import("numpy")
ak = maybe_import("awkward")


#
# unexposed selectors
# (not selectable from the command line but used by other, exposed selectors)
#

@selector
def lepton_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # -- muons

    # get selection config
    mu_sel = self.config_inst.x.lepton_selection.mu

    # low-pT
    muon_mask_low_pt = (
        (events.Muon.pt > mu_sel.min_pt.low_pt) &
        (events.Muon.pt <= mu_sel.min_pt.high_pt) &
        (abs(events.Muon.eta) < mu_sel.max_abseta) &
        (events.Muon[mu_sel.iso.column] >= mu_sel.iso.min_value) &
        (events.Muon[mu_sel.id.low_pt.column]) &
        (events.HLT[mu_sel.trigger.low_pt])
    )

    # high-pT
    muon_mask_high_pt = (
        (events.Muon.pt > mu_sel.min_pt.high_pt) &
        (abs(events.Muon.eta) < mu_sel.max_abseta) &
        (events.Muon[mu_sel.id.high_pt.column] == mu_sel.id.high_pt.value) &
        (events.HLT[mu_sel.trigger.high_pt])
    )

    # merge low- and high-pT masks
    muon_mask = (
        muon_mask_high_pt | muon_mask_low_pt
    )
    # muon_indices = sorted_indices_from_mask(muon_mask, events.Muon.pt)
    muon_sel = (ak.sum(muon_mask, axis=1) == 1)

    # -- electrons

    # get selection config
    el_sel = self.config_inst.x.lepton_selection.e

    # low-pT
    electron_eta_with_sc = events.Electron.eta + events.Electron.deltaEtaSC
    electron_mask_low_pt = (
        (events.Electron.pt > el_sel.min_pt.low_pt) &
        (events.Electron.pt <= el_sel.min_pt.high_pt) &
        (abs(electron_eta_with_sc) < el_sel.max_abseta) &
        (events.Electron[el_sel.mva_id.low_pt])
    )

    # high-pT
    electron_mask_high_pt = (
        (events.Electron.pt > el_sel.min_pt.high_pt) &
        (abs(electron_eta_with_sc) < el_sel.max_abseta) &
        (events.Electron[el_sel.mva_id.high_pt])
    )

    # merge low- and high-pT masks
    electron_mask = (
        electron_mask_high_pt | electron_mask_low_pt
    )
    # electron_indices = sorted_indices_from_mask(electron_mask, events.Electron.pt)
    electron_sel = (ak.sum(electron_mask, axis=1) == 1)

    # require the electron trigger to calculate efficiencies
    # NOTE: not needed here for baseline selection -> use categories
    # electron_trigger_sel = events.HLT.Ele35_WPTight_Gsf

    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Muon": {"MySelectedMuon": indices_applied_to_Muon}}

    return events, SelectionResult(
        steps={
            "muon": ak.fill_none(muon_sel, False),
            "electron": ak.fill_none(electron_sel, False),
        },
        objects={
            "Muon": {
                "Muon": ak.fill_none(muon_mask, False),
                "MuonHighPt": ak.fill_none(muon_mask_high_pt, False),
                "MuonLowPt": ak.fill_none(muon_mask_low_pt, False),
            },
            "Electron": {
                "Electron": ak.fill_none(electron_mask, False),
                "ElectronHighPt": ak.fill_none(electron_mask_high_pt, False),
                "ElectronLowPt": ak.fill_none(electron_mask_low_pt, False),
            },
        },
    )


@lepton_selection.init
def lepton_selection_init(self: Selector) -> None:
    config_inst = getattr(self, "config_inst", None)

    if not config_inst:
        return

    # lepton columns
    for sel_key in ("e", "mu"):
        sel = self.config_inst.x.lepton_selection[sel_key]
        column = sel.get("column")
        if column:
            self.uses |= {
                f"{column}.pt",
                f"{column}.eta",
                f"{column}.phi",
                f"{column}.mass",
            }
            if sel_key == "e":
                self.uses |= {
                    f"{column}.deltaEtaSC",
                    f"{column}.{sel.mva_id.low_pt}",
                    f"{column}.{sel.mva_id.high_pt}",
                }
            elif sel_key == "mu":
                self.uses |= {
                    f"{column}.{sel.iso.column}",
                    f"{column}.{sel.id.low_pt.column}",
                    f"{column}.{sel.id.high_pt.column}",
                }

    # trigger columns
    self.uses |= {
        f"HLT.{self.config_inst.x.triggers.low_pt}",
        f"HLT.{self.config_inst.x.triggers.high_pt}",
    }


#
# jets
#

@selector
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    jet_sel = self.config_inst.x.jet_selection.ak4

    # only consider jets within a certain kinematic range
    jet_mask = (events.Jet.pt >= jet_sel.all.min_pt) & (abs(events.Jet.eta) < jet_sel.all.max_abseta)
    jet_indices = sorted_indices_from_mask(jet_mask, events.Jet.pt, ascending=False)

    # B-tagged and light jets
    bjet_mask = (jet_mask) & (events.Jet.btagDeepFlavB >= jet_sel.btagger.wp)
    bjet_indices = sorted_indices_from_mask(bjet_mask, events.Jet.pt, ascending=False)
    sel_bjet = ak.sum(bjet_mask, axis=-1) >= 1

    # get kinematics for first two jets (fill none if missing)
    n_leading_jet_sel = len(jet_sel.leading)
    jet_pt = ak.pad_none(events.Jet.pt, n_leading_jet_sel)
    jet_eta = ak.pad_none(events.Jet.eta, n_leading_jet_sel)

    # select events where two leading jets pass criteria
    sel_jet1 = (jet_pt[:, 0] >= jet_sel.leading[0]["min_pt"]) & (abs(jet_eta[:, 0]) < jet_sel.leading[0]["max_abseta"])
    sel_jet2 = (jet_pt[:, 1] >= jet_sel.leading[1]["min_pt"]) & (abs(jet_eta[:, 1]) < jet_sel.leading[1]["max_abseta"])

    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Jet": {"MyCustomJetCollection": indices_applied_to_Jet}}
    return events, SelectionResult(
        steps={
            "jet1": ak.fill_none(sel_jet1, False),
            "jet2": ak.fill_none(sel_jet2, False),
            "bjet": ak.fill_none(sel_bjet, False),
        },
        objects={
            "Jet": {
                "Jet": jet_indices,
                "BJet": bjet_indices,
            },
        },
        aux={
            "n_jets": ak.sum(jet_mask, axis=1),
        },
    )


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    config_inst = getattr(self, "config_inst", None)

    if not config_inst:
        return

    sel = self.config_inst.x.jet_selection.ak4

    column = sel.get("column")
    if column:
        self.uses |= {
            f"{column}.pt",
            f"{column}.eta",
            f"{column}.phi",
            f"{column}.mass",
            f"{column}.btagDeepFlavB",
        }


#
# MET
#

@selector
def met_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    # MET selection
    sel_met = events.MET.pt > self.config_inst.x.met_selection.min_pt

    return events, SelectionResult(
        steps={
            "MET": ak.fill_none(sel_met, False),
        },
    )


@met_selection.init
def met_selection_init(self: Selector) -> None:
    config_inst = getattr(self, "config_inst", None)

    if not config_inst:
        return

    sel = self.config_inst.x.met_selection

    column = sel.get("column")
    if column:
        self.uses |= {
            f"{column}.pt",
        }


#
# 2D cut
#

@selector(
    # uses={
    #     attach_coffea_behavior,
    # },
)
def lepton_jet_2d_selection(
    self: Selector,
    events: ak.Array,
    electron_mask: ak.Array | None = None,
    muon_mask: ak.Array | None = None,
    jet_mask: ak.Array | None = None,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    cfg_2d_iso = self.config_inst.x.lepton_jet_iso

    # apply masks
    electron = events.Electron
    if electron_mask is not None:
        electron = electron[electron_mask]

    muon = events.Muon
    if muon_mask is not None:
        muon = muon[muon_mask]

    jet = events.Jet
    if jet_mask is not None:
        jet = jet[jet_mask]

    electron = ak.with_name(electron, "PtEtaPhiMLorentzVector")
    muon = ak.with_name(muon, "PtEtaPhiMLorentzVector")
    jet = ak.with_name(jet, "PtEtaPhiMLorentzVector")

    # distance between leading electron and nearest jet
    leading_el = ak.firsts(electron)
    leading_el_nearest_jet = ak.firsts(leading_el.nearest(jet))
    leading_el_nearest_jet_delta_r = (
        leading_el_nearest_jet.delta_r(leading_el)
    )
    leading_el_nearest_jet_pt_rel = (
        leading_el.pvec.cross(leading_el_nearest_jet.pvec).p /
        leading_el_nearest_jet.p
    )

    # distance between leading muon and nearest jet
    leading_mu = ak.firsts(muon)
    leading_mu_nearest_jet = ak.firsts(leading_mu.nearest(jet))
    leading_mu_nearest_jet_delta_r = (
        leading_mu_nearest_jet.delta_r(leading_mu)
    )
    leading_mu_nearest_jet_pt_rel = (
        leading_mu.pvec.cross(leading_mu_nearest_jet.pvec).p /
        leading_mu_nearest_jet.p
    )

    # cut on distances
    sel_2d_el = (
        (leading_el_nearest_jet_pt_rel > cfg_2d_iso.min_pt_rel) |
        (leading_el_nearest_jet_delta_r > cfg_2d_iso.min_delta_r)
    )
    sel_2d_mu = (
        (leading_mu_nearest_jet_pt_rel > cfg_2d_iso.min_pt_rel) |
        (leading_mu_nearest_jet_delta_r > cfg_2d_iso.min_delta_r)
    )

    return events, SelectionResult(
        steps={
            "electron_jet_2d_cut": ak.fill_none(sel_2d_el, True),
            "muon_jet_2d_cut": ak.fill_none(sel_2d_mu, True),
        },
    )


@lepton_jet_2d_selection.init
def lepton_jet_2d_selection_init(self: Selector) -> None:
    config_inst = getattr(self, "config_inst", None)

    if not config_inst:
        return

    sel_ele = self.config_inst.x.lepton_selection.e
    sel_muo = self.config_inst.x.lepton_selection.mu
    sel_jet = self.config_inst.x.jet_selection.ak4

    for sel in (sel_ele, sel_muo, sel_jet):
        column = sel.get("column")
        if column:
            self.uses |= {
                f"{column}.pt",
                f"{column}.eta",
                f"{column}.phi",
                f"{column}.mass",
            }


# exposed selectors
# (those that can be invoked from the command line)
#

@selector(
    uses={
        # selectors / producers called within _this_ selector
        mc_weight,
        cutflow_features,
        process_ids,
        lepton_selection,
        jet_selection,
        met_selection,
        lepton_jet_2d_selection,
        category_ids,
        increment_stats,
        json_filter,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight,
        cutflow_features,
        process_ids,
        category_ids,
        json_filter,
    },
    exposed=True,
)
def default_tgs(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # JSON filter (data-only)
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results

    # muon selection
    events, lepton_results = self[lepton_selection](events, **kwargs)
    results += lepton_results

    # jet-electron and jet-muon 2D selections
    electron_mask_for_2d_cut = (
        lepton_results.objects.Electron.ElectronHighPt &
        (events.Electron.pt >= self.config_inst.x.lepton_jet_iso.min_pt)
    )
    muon_mask_for_2d_cut = (
        lepton_results.objects.Muon.MuonHighPt &
        (events.Muon.pt >= self.config_inst.x.lepton_jet_iso.min_pt)
    )
    events, lepton_jet_2d_results = self[lepton_jet_2d_selection](
        events,
        electron_mask=electron_mask_for_2d_cut,
        muon_mask=muon_mask_for_2d_cut,
        jet_mask=(events.Jet.pt > 15),
        **kwargs,
    )
    results += lepton_jet_2d_results

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # met selection
    events, met_results = self[met_selection](events, **kwargs)
    results += met_results

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel

    for step, sel in results.steps.items():
        n_sel = ak.sum(sel, axis=-1)
        print(f"{step}: {n_sel}")

    n_sel = ak.sum(results.event, axis=-1)
    print(f"__all__: {n_sel}")

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add the mc weight
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # add cutflow features, passing per-object masks
    events = self[cutflow_features](events, results.objects, **kwargs)

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
