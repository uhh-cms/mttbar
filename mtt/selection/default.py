# coding: utf-8

"""
Selection methods for m(ttbar).
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

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "Jet.pt", "Jet.eta", "Jet.btagDeepFlavB",
        "Jet.electronIdx1", "Jet.electronIdx2",
        "Jet.muonIdx1", "Jet.muonIdx2",
    },
    shifts={jet_energy_shifts},
    exposed=True,
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """Baseline jet selection for m(ttbar)."""
    # - require at least one AK4 jet with pt>50 and abseta<2.5
    # - require a second AK4 jet with pt>30 and abseta<2.5
    # - require that at least one AK4 jet (pt>50 and abseta<2.5) is b-tagged

    # loose jets (pt>0.1) - to filter out cleaned jets etc.
    loose_jet_mask = (events.Jet.pt > 0.1)
    loose_jet_indices = masked_sorted_indices(loose_jet_mask, events.Jet.pt)

    # jets (pt>30)
    jet_mask = (
        (abs(events.Jet.eta) < 2.5) &
        (events.Jet.pt > 30)
    )
    jet_indices = masked_sorted_indices(jet_mask, events.Jet.pt)

    # at least two jets, leading jet pt > 50,
    # subleading jet pt > 30
    jet = ak.pad_none(events.Jet[jet_indices], 2)
    sel_jet = (
        (jet[:, 0].pt > 50) &
        (jet[:, 1].pt > 30)
    )
    sel_jet = ak.fill_none(sel_jet, False)

    # MISSING: match AK4 PUPPI jets to AK4 CHS jets for b-tagging

    # b-tagged jets, DeepCSV medium working point
    # TODO: update to DeepJet
    wp_med = self.config_inst.x.btag_working_points.deepcsv.medium
    bjet_mask = (jet_mask) & (events.Jet.btagDeepFlavB >= wp_med)
    lightjet_mask = (jet_mask) & (events.Jet.btagDeepFlavB < wp_med)
    sel_bjet = ak.sum(bjet_mask, axis=-1) >= 1

    # indices of the b-tagged and non-b-tagged (light) jets
    bjet_indices = masked_sorted_indices(bjet_mask, events.Jet.pt)
    lightjet_indices = masked_sorted_indices(lightjet_mask, events.Jet.pt)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "Jet": sel_jet,
            "BJet": sel_bjet,
        },
        objects={
            "Jet": {
                "Jet": loose_jet_indices,
                "BJet": bjet_indices,
                "LightJet": lightjet_indices,
            },
        },
    )


@selector(
    uses={
        choose_lepton,
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass",
        "FatJet.deepTagMD_TvsQCD", "FatJet.msoftdrop",
    },
    exposed=True,
)
def top_tagged_jets(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Apply top tagging criteria to AK8 jets.

    Veto events with more than one top-tagged AK8 jet with pT>400 GeV and |eta|<2.5.
    """

    # top-tagger working point
    wp_top_md = self.config_inst.x.toptag_working_points.deepak8.top_md

    # top-tagging criteria
    fatjet_mask_toptag = (
        # kinematic cuts
        (events.FatJet.pt > 400) &
        (abs(events.FatJet.eta) < 2.5) &
        # 1st topjet requirement: top tagger working point
        (events.FatJet.deepTagMD_TvsQCD > wp_top_md) &
        # 2nd topjet requirement: softdrop mass window
        (events.FatJet.msoftdrop > 105) &
        (events.FatJet.msoftdrop < 210)
    )
    fatjet_indices_toptag = masked_sorted_indices(
        fatjet_mask_toptag,
        events.FatJet.pt,
    )

    # veto events with more than one top-tagged AK8 jet
    sel_all_had_veto = (ak.sum(fatjet_mask_toptag, axis=-1) < 2)

    # get main lepton
    events = self[choose_lepton](events, **kwargs)
    lepton = events["Lepton"]

    # separation from main lepton
    delta_r_fatjet_lepton = ak.firsts(events.FatJet.metric_table(lepton), axis=-1)
    fatjet_mask_toptag_delta_r_lepton = (
        fatjet_mask_toptag &
        # pass if no main lepton exists
        ak.fill_none(delta_r_fatjet_lepton > 0.8, True)
    )
    fatjet_indices_toptag_delta_r_lepton = masked_sorted_indices(
        fatjet_mask_toptag_delta_r_lepton,
        events.FatJet.pt,
    )

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "AllHadronicVeto": sel_all_had_veto,
        },
        objects={
            "FatJet": {
                "FatJetTopTag": fatjet_indices_toptag,
                "FatJetTopTagDeltaRLepton": fatjet_indices_toptag_delta_r_lepton,
            },
        },
    )


@selector(
    uses={
        "event",
        "MET.pt",
    },
    exposed=True,
)
def met_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """Missing transverse momentum (MET) selection."""

    # missing transverse momentum > 50 GeV
    sel_met = (events.MET.pt > 50)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "MET": sel_met,
        },
        objects={
        },
    )


@selector(
    uses={
        attach_coffea_behavior,
        lepton_selection,
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
    },
    exposed=True,
)
def lepton_jet_2d_selection(
    self: Selector,
    events: ak.Array,
    lepton_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Lepton/Jet 2D cut (replaces isolation criterion for high-pt lepton).

    The 2D requirement is defined as

      delta_r(l, jet) > 0.4  ||  pt_rel(l, jet) > 25 GeV,

    where *pt_rel* denotes the magnitude of the perpendicular component
    of the lepton three-vector with respect to the jet axis:

      pt_rel = p_l * sin(angle(p_l, p_jet))

    and can be calculated eqivalently via the cross product of the jet
    and lepton three-momenta as:

      pt_rel = |cross(p_l, p_jet)| / |p_jet|
    """

    # note: returns only 'events' if lepton_selection has been called before
    #       and is cached (we assume this here), otherwise returns a tuple
    #       (events, SelectionResult)
    events = self[lepton_selection](events, **kwargs)

    # select jets
    jets_mask = (events.Jet.pt > 15)
    jets = events.Jet[jets_mask]

    ch_e = self.config_inst.get_channel("e")
    ch_m = self.config_inst.get_channel("mu")

    selections = {}
    for ch, route in [
        (ch_e, "Electron"),
        (ch_m, "Muon"),
    ]:
        lepton_indices = lepton_results.objects[route][route]

        # if chunk contains no leptons, return early
        # (seems awkward is unable to handle arrays where
        # every entry is masked here)
        if len(ak.flatten(lepton_indices)) == 0:
            selections[ch.id] = ak.ones_like(events.event, dtype=bool)
            continue

        leptons = ak.firsts(events[route][lepton_indices])
        lepton_jet_deltar = ak.firsts(jets.metric_table(leptons), axis=-1)

        lepton_closest_jet = ak.firsts(
            jets[masked_sorted_indices(jets_mask, lepton_jet_deltar, ascending=True)],
        )

        # veto events where there is a jet too close to the lepton
        sel = ak.all(lepton_jet_deltar > 0.4, axis=-1)

        # but keep events where the perpendicular lepton momentum relative
        # to the jet is sufficiently large
        pt_rel = leptons.cross(lepton_closest_jet).p / lepton_closest_jet.p
        sel = ak.where(
            pt_rel > 25,
            True,
            sel,
        )

        selections[ch.id] = sel

    channel_id = events.channel_id
    is_highpt = (lepton_results.x.pt_regime == 2)

    # combine muon and electron selection depending on the channel
    sel_lepton = ak.ones_like(events.event, dtype=bool)
    for ch in [ch_e, ch_m]:
        sel_lepton = ak.where(
            channel_id == ch.id,
            selections[ch.id],
            sel_lepton,
        )

    # only apply selection in high pt regime
    sel_lepton = ak.where(
        is_highpt,
        sel_lepton,
        True,
    )

    # include undefined events in selection
    sel_lepton = ak.fill_none(sel_lepton, True)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "JetLepton2DCut": sel_lepton,
        },
    )


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


@selector(
    uses={
        attach_coffea_behavior,
        lepton_selection,
        check_early,
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
    },
    exposed=True,
)
def data_trigger_veto(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    # get trigger requirements
    trigger_config = self.config_inst.x.triggers

    # check if event is in early run period
    is_early = self[check_early](events, trigger_config=trigger_config, **kwargs)

    # ensure lepton selection was run, get lepton pT regime
    events = self[lepton_selection](events, **kwargs)
    pt_regime = events["pt_regime"]

    # pt regime booleans for convenience
    is_lowpt = (pt_regime == 1)
    is_highpt = (pt_regime == 2)

    triggers = {}
    trigger_masks = {}
    pass_trigger = {}
    for object_name in ["muon", "electron", "photon"]:
        triggers[object_name] = {
            "lowpt": trigger_config.get("lowpt", {}).get("all", {}).get("triggers", {}).get(object_name, {}),
            "highpt_early": trigger_config.get("highpt", {}).get("early", {}).get("triggers", {}).get(object_name, {}),
            "highpt_late": trigger_config.get("highpt", {}).get("late", {}).get("triggers", {}).get(object_name, {}),
        }
        trigger_masks[object_name] = object_trigger_masks = {}
        # get trigger decisions if trigger is available
        for key, trigger_names in triggers[object_name].items():
            object_trigger_masks[key] = ak.zeros_like(events.event, dtype=bool)
            for trigger_name in trigger_names:
                if trigger_name in events.HLT.fields:
                    object_trigger_masks[key] = (
                        object_trigger_masks[key] |
                        events.HLT[trigger_name]
                    )
                else:
                    object_trigger_masks[key] = (False)

        object_trigger_masks["highpt"] = ak.where(
            is_early,
            object_trigger_masks["highpt_early"],
            object_trigger_masks["highpt_late"],
        )

        # trigger selection
        pass_object_trigger = ak.zeros_like(events.event, dtype=bool)
        pass_object_trigger = ak.where(
            is_lowpt,
            object_trigger_masks["lowpt"],
            pass_object_trigger,
        )
        pass_object_trigger = ak.where(
            is_highpt,
            object_trigger_masks["highpt"],
            pass_object_trigger,
        )
        pass_trigger[object_name] = pass_object_trigger

    if getattr(self.dataset_inst.x, "is_e_data", None):
        sel_veto = ak.fill_none(pass_trigger["electron"], False)
    if getattr(self.dataset_inst.x, "is_pho_data", None):
        sel_veto = ak.fill_none(pass_trigger["photon"] & ~pass_trigger["electron"], False)
    if getattr(self.dataset_inst.x, "is_mu_data", None):
        sel_veto = ak.fill_none(pass_trigger["muon"] & ~pass_trigger["electron"] & ~pass_trigger["photon"], False)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "TriggerVeto": sel_veto,
        },
    )


@selector(
    uses={
        jet_selection, lepton_selection, met_selection, top_tagged_jets, lepton_jet_2d_selection,
        cutflow_features,
        category_ids,
        process_ids, increment_stats, attach_coffea_behavior,
        mc_weight,
        met_filters,
        json_filter,
    },
    produces={
        jet_selection, lepton_selection, met_selection, top_tagged_jets, lepton_jet_2d_selection,
        cutflow_features,
        category_ids,
        process_ids, increment_stats, attach_coffea_behavior,
        mc_weight,
        met_filters,
        json_filter,
    },
    shifts={
        jet_energy_shifts,
    },
    exposed=True,
)
def default(
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

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # lepton selection
    events, lepton_results = self[lepton_selection](events, **kwargs)
    results += lepton_results

    # jet-lepton 2D cut
    events, lepton_jet_2d_results = self[lepton_jet_2d_selection](
        events,
        lepton_results,
        **kwargs,
    )
    results += lepton_jet_2d_results

    # met selection
    events, met_results = self[met_selection](events, **kwargs)
    results += met_results

    # all-hadronic veto
    events, top_tagged_jets_results = self[top_tagged_jets](events, **kwargs)
    results += top_tagged_jets_results

    if getattr(self.dataset_inst.x, "is_qcd", None):
        events, qcd_sel_results = self[qcd_spikes](events, **kwargs)
        results += qcd_sel_results

    if getattr(self.dataset_inst.x, "is_data", None):
        events, trigger_veto_results = self[data_trigger_veto](events, **kwargs)
        results += trigger_veto_results

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.main["event"] = event_sel

    for step, sel in results.steps.items():
        n_sel = ak.sum(sel, axis=-1)
        print(f"{step}: {n_sel}")

    n_sel = ak.sum(event_sel, axis=-1)
    print(f"__all__: {n_sel}")

    # add cutflow features
    events = self[cutflow_features](events, results=results, **kwargs)

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add mc weights (needed for cutflow plots)
    if getattr(self.dataset_inst.x, "is_mc", None):
        events = self[mc_weight](events, **kwargs)

    # increment stats
    self[increment_stats](events, results, stats, **kwargs)

    return events, results


@default.init
def default_init(self: Selector) -> None:

    if hasattr(self, "dataset_inst") and getattr(self.dataset_inst.x, "is_qcd", None):
        self.uses |= {qcd_spikes}
        self.produces |= {qcd_spikes}

    if hasattr(self, "dataset_inst") and getattr(self.dataset_inst.x, "is_data", None):
        self.uses |= {data_trigger_veto}
        self.produces |= {data_trigger_veto}


@selector(
    uses={
        jet_selection, lepton_selection, met_selection, top_tagged_jets,
        cutflow_features,
        category_ids,
        process_ids, increment_stats, attach_coffea_behavior,
        mc_weight,
        met_filters,
        json_filter,
    },
    produces={
        jet_selection, lepton_selection, met_selection, top_tagged_jets,
        cutflow_features,
        category_ids,
        process_ids, increment_stats, attach_coffea_behavior,
        mc_weight,
        met_filters,
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

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # lepton selection
    events, lepton_results = self[lepton_selection](events, **kwargs)
    results += lepton_results

    # met selection
    events, met_results = self[met_selection](events, **kwargs)
    results += met_results

    # all-hadronic veto
    events, top_tagged_jets_results = self[top_tagged_jets](events, **kwargs)
    results += top_tagged_jets_results

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.main["event"] = event_sel

    for step, sel in results.steps.items():
        n_sel = ak.sum(sel, axis=-1)
        print(f"{step}: {n_sel}")

    n_sel = ak.sum(event_sel, axis=-1)
    print(f"__all__: {n_sel}")

    # add cutflow features
    events = self[cutflow_features](events, results=results, **kwargs)

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add mc weights (needed for cutflow plots)
    events = self[mc_weight](events, **kwargs)

    # increment stats
    self[increment_stats](events, results, stats, **kwargs)

    return events, results
