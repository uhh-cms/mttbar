# coding: utf-8

"""
Selection methods for m(ttbar).
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.calibration.jets import ak_random  # TODO: move function

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids

from mtt.selection.general import increment_stats, jet_energy_shifts
from mtt.selection.cutflow_features import cutflow_features

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(uses={"event"})
def select_all(self: Selector, events: ak.Array, **kwargs) -> ak.Array:
    """Passes every event."""
    return ak.ones_like(events.event)


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


@selector(
    uses={"Jet.pt", "Jet.eta", "Jet.btagDeepFlavB"},
    shifts={jet_energy_shifts},
    exposed=True,
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """Baseline jet selection for m(ttbar)."""
    # - require at least one AK4 jet with pt>50 and abseta<2.5
    # - require a second AK4 jet with pt>30 and abseta<2.5
    # - require that at least one AK4 jet (pt>50 and abseta<2.5) is b-tagged

    # jets
    jet_mask_50 = (events.Jet.pt > 50) & (abs(events.Jet.eta) < 2.5)
    sel_jet_50 = ak.sum(jet_mask_50, axis=-1) >= 1
    jet_indices_50 = masked_sorted_indices(jet_mask_50, events.Jet.pt)

    # jets (pt>30), not including jets matched previously
    jet_mask_30 = (events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.5) & (~jet_mask_50)
    sel_jet_30 = ak.sum(jet_mask_30, axis=-1) >= 1
    jet_indices_30 = masked_sorted_indices(jet_mask_30, events.Jet.pt)

    # MISSING: match AK4 PUPPI jets to AK4 CHS jets for b-tagging

    # b-tagged jets, DeepCSV medium working point
    # TODO: update to DeepJet
    wp_med = self.config_inst.x.btag_working_points.deepcsv.medium
    bjet_mask = (jet_mask_50) & (events.Jet.btagDeepFlavB >= wp_med)
    sel_bjet = ak.sum(bjet_mask, axis=-1) >= 1

    # sort jets after b-score and define b-jets as the two b-score leading jets
    bjet_indices = masked_sorted_indices(jet_mask_50, events.Jet.btagDeepFlavB)[:, :2]

    # lightjets are the remaining jets (TODO: b-score sorted but should be pt-sorted?)
    lightjet_indices = masked_sorted_indices(jet_mask_50, events.Jet.btagDeepFlavB)[:, 2:]

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "Jet50": sel_jet_50,
            "Jet30": sel_jet_30,
            "Bjet": sel_bjet,
        },
        objects={
            "Jet": {
                "Jet50": jet_indices_50,
                "Jet30": jet_indices_30,
                "Bjet": bjet_indices,
                "Lightjet": lightjet_indices
            }
        },
    )


@selector(
    uses={
        "Muon.pt", "Muon.eta", "Muon.tightId", "Muon.highPtId", "Muon.pfIsoId",
        "HLT.*",
    },
    produces={
        "mtt_muon_is_high_pt",
    },
    exposed=True,
)
def muon_selection(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """Baseline muon selection for m(ttbar)."""
    # - muon triggers
    # - require exactly 1 muon with abseta < 2.4 and either (30 < pt < 55 GeV) and CutBasedIdTight,
    #   or with pt>55 GeV and CutBasedIdGlobalHighPt
    # - low-pt muons have to satisfy the isolation ID "PfIsoTight"
    # - high-pt muons have to satisfy the 2D cut defined as:
    #   delta_r(l, jet) > 0.4 || pt_rel(l, jet) > 25 GeV

    # muon kinematics
    mu_mask_eta = (abs(events.Muon.eta) < 2.4)
    mu_mask_lowpt = (
        mu_mask_eta &
        (events.Muon.pt > 30) &
        (events.Muon.pt <= 55) &
        # CutBasedIdTight
        (events.Muon.tightId) &
        # PFIsoTight
        (events.Muon.pfIsoId == 4)
    )
    mu_mask_highpt = (
        mu_mask_eta &
        (events.Muon.pt > 55) &
        # CutBasedIdGlobalHighPt
        (events.Muon.highPtId == 2)
    )

    # muon multiplicity
    n_mu_lowpt = ak.sum(mu_mask_lowpt, axis=-1)
    n_mu_highpt = ak.sum(mu_mask_highpt, axis=-1)

    # high-pt muon isolation
    mu_mask_highpt_iso = ak.where(
        mu_mask_highpt,
        # TODO: calc iso wrt closest jet
        # (
        #     ak.min(
        #         deltaR(event.Muon, jets_pt_15),
        #         axis=-1
        #     ) > 0.4 |
        #     abs(event.Muon.pt - closest_jet.pt) > 25
        # ),
        True,
        True
    )

    # muon indices
    mu_mask = (mu_mask_lowpt | (mu_mask_highpt & mu_mask_highpt_iso))
    mu_indices = masked_sorted_indices(mu_mask, events.Muon.pt)

    # produce column: true if muon is high-pt
    events = set_ak_column(events, "mtt_muon_is_high_pt", n_mu_highpt == 1)

    # check trigger requirements
    trigger_config = self.config_inst.x.triggers
    triggers = {
        "lowpt": trigger_config.lowpt.all.triggers.muon,
        "highpt_early": trigger_config.highpt.early.triggers.muon,
        "highpt_late": trigger_config.highpt.late.triggers.muon,
    }

    # precompute trigger masks
    mu_trigger_masks = {}
    for pt_regime, trigger_names in triggers.items():
        mu_trigger_masks[pt_regime] = ak.zeros_like(events.event, dtype=bool)
        for trigger_name in trigger_names:
            # mu_trigger_masks[pt_regime] |= ... throws error. Numpy bug?
            mu_trigger_masks[pt_regime] = mu_trigger_masks[pt_regime] | events.HLT[trigger_name]

    # determine if in early run period (or MC equivalent)
    if self.dataset_inst.is_mc:
        # in MC, by predefined event fraction using uniformly distributed random numbers

        # use event numbers in chunk to seed random number generator
        # TODO: use seeds!
        rand_gen = np.random.Generator(np.random.SFC64(events.event.to_list()))

        # uniformly distributed random numbers in [0, 100]
        random_percent = ak_random(
            ak.zeros_like(events.event),
            ak.ones_like(events.event) * 100,
            rand_func=rand_gen.uniform,
        )

        condition_early = (
            random_percent < trigger_config.highpt.early.mc_trigger_percent
        )
    else:
        # in data, by run number
        condition_early = (
            events.run <= trigger_config.highpt.early.run_range_max
        )

    # determine which high-pt trigger combination is used
    mu_trigger_masks["highpt"] = ak.where(
        condition_early,
        mu_trigger_masks["highpt_early"],
        mu_trigger_masks["highpt_late"],
    )

    # trigger selection
    sel_mu_trigger = ak.where(
        events.mtt_muon_is_high_pt,
        mu_trigger_masks["highpt"],
        mu_trigger_masks["lowpt"]
    )

    # offline selection
    sel_mu = (n_mu_lowpt + n_mu_highpt) == 1

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "MuonTrigger": sel_mu_trigger,
            "Muon": sel_mu,
        },
        objects={
            "Muon": {
                "Muon": mu_indices,
            }
        },
    )

@selector(
    uses={
        "nFatJet", "FatJet.pt", "FatJet.eta",
        "FatJet.deepTagMD_TvsQCD", "FatJet.msoftdrop",
    },
    exposed=True,
)
def all_had_veto(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """Veto events with more than one AK8 jet with pT>400 GeV and |eta|<2.5
    passing the top-tagging requirements."""

    fatjet_mask_toptag = (
        #events.FatJet.deepTagMD_TvsQCD > 0.0  # TODO
        ak.zeros_like(events.FatJet.deepTagMD_TvsQCD, dtype=bool)
    )
    fatjet_indices_toptag = masked_sorted_indices(
        fatjet_mask_toptag,
        events.FatJet.pt
    )

    fatjet_mask_msoftdrop = (
        (events.FatJet.msoftdrop > 105) &
        (events.FatJet.msoftdrop < 210)
    )
    fatjet_indices_msoftdrop = masked_sorted_indices(
        fatjet_mask_msoftdrop,
        events.FatJet.pt
    )

    fatjet_mask_vetoregion = (
        (events.FatJet.pt > 400) &
        (abs(events.FatJet.eta) < 2.5)
    )

    fatjet_mask = (fatjet_mask_vetoregion & fatjet_mask_msoftdrop & fatjet_mask_toptag)
    sel_all_had_veto = (ak.sum(fatjet_mask, axis=-1) < 2)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "AllHadronicVeto": sel_all_had_veto,
        },
        objects={
            "FatJet": {
                "TopTag": fatjet_indices_toptag,
                "MSoftDrop": fatjet_indices_msoftdrop,
            }
        },
    )


@selector(
    uses={
        "MET.pt",
    },
    exposed=True,
)
def met_selection(
        self: Selector,
        events: ak.Array,
        stats: defaultdict,
        **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """Missing transverse momentum (MET) selection."""

    # TODO
    sel_met = ak.ones_like(events.event, dtype=bool)

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
        jet_selection, muon_selection, met_selection, all_had_veto, cutflow_features,
        category_ids, process_ids, increment_stats, attach_coffea_behavior,
        "mc_weight",  # not opened per default but always required in Cutflow tasks
    },
    produces={
        jet_selection, muon_selection, met_selection, all_had_veto, cutflow_features,
        category_ids, process_ids, increment_stats, attach_coffea_behavior,
        "mc_weight",  # not opened per default but always required in Cutflow tasks
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
) -> Tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # jet selection
    events, jet_results = self[jet_selection](events, stats, **kwargs)
    results += jet_results

    # muon selection
    events, muon_results = self[muon_selection](events, stats, **kwargs)
    results += muon_results

    # met selection
    events, met_results = self[met_selection](events, stats, **kwargs)
    results += met_results

    # all-hadronic veto
    events, all_had_veto_results = self[all_had_veto](events, stats, **kwargs)
    results += all_had_veto_results

    # combined event selection after all steps
    event_sel = (
        # jet selection
        jet_results.steps.Jet50 &
        jet_results.steps.Jet30 &
        jet_results.steps.Bjet &
        # muon selection
        muon_results.steps.Muon &
        muon_results.steps.MuonTrigger &
        # met selection
        met_results.steps.MET &
        # all-hadronic veto
        all_had_veto_results.steps.AllHadronicVeto
    )
    results.main["event"] = event_sel

    for k, v in [
        ("Jet50", jet_results.steps.Jet50),
        ("Jet30", jet_results.steps.Jet30),
        ("Bjet", jet_results.steps.Bjet),
        ("Muon", muon_results.steps.Muon),
        ("MuonTrigger", muon_results.steps.MuonTrigger),
        ("MET", met_results.steps.MET),
        ("AllHadronicVeto", all_had_veto_results.steps.AllHadronicVeto),
    ]:
        n_sel = ak.sum(v, axis=-1)
        print(f"{k}: {n_sel}")

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add cutflow features
    events = self[cutflow_features](events, results=results, **kwargs)

    # increment stats
    self[increment_stats](events, event_sel, stats, **kwargs)

    return events, results
