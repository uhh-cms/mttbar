# coding: utf-8

"""
Selection methods for m(ttbar).
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.calibration.jets import ak_random  # TODO: move function

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.categories import category_ids
from columnflow.production.mc_weight import mc_weight
from columnflow.production.processes import process_ids

from mtt.selection.util import masked_sorted_indices
from mtt.selection.general import increment_stats, jet_energy_shifts
from mtt.selection.lepton import lepton_selection
from mtt.selection.cutflow_features import cutflow_features

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
        (jet[..., 0].pt > 50) &
        (jet[..., 1].pt > 30)
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
            "Bjet": sel_bjet,
        },
        objects={
            "Jet": {
                "Jet": jet_indices,
                "Bjet": bjet_indices,
                "Lightjet": lightjet_indices
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
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """Veto events with more than one AK8 jet with pT>400 GeV and |eta|<2.5
    passing the top-tagging requirements."""

    wp_top_md = self.config_inst.x.toptag_working_points.deepak8.top_md

    fatjet_mask_toptag = (
        events.FatJet.deepTagMD_TvsQCD > wp_top_md  # TODO
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
    """Lepton/Jet 2D cut (replaces isolation criterion for high-pt lepton)."""

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
        lepton_indices = lepton_results.objects[route].Lepton
        leptons = ak.firsts(events[route][lepton_indices])
        lepton_jet_deltar = ak.firsts(jets.metric_table(leptons), axis=-1)

        lepton_closest_jet = ak.firsts(
            jets[masked_sorted_indices(jets_mask, lepton_jet_deltar)]
        )

        # veto events where there is a jet too close to the lepton
        sel = ak.all(lepton_jet_deltar > 0.4, axis=-1)

        # but keep events where the lepton/jet pt difference is
        # sufficiently large
        sel = ak.where(
            (leptons.pt - lepton_closest_jet.pt) > 25,
            True,
            sel
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
        True
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
        jet_selection, lepton_selection, met_selection, all_had_veto, lepton_jet_2d_selection,
        cutflow_features, category_ids, process_ids, increment_stats, attach_coffea_behavior,
        mc_weight,
    },
    produces={
        jet_selection, lepton_selection, met_selection, all_had_veto, lepton_jet_2d_selection,
        cutflow_features, category_ids, process_ids, increment_stats, attach_coffea_behavior,
        mc_weight,
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
        **kwargs
    )
    results += lepton_jet_2d_results

    # met selection
    events, met_results = self[met_selection](events, **kwargs)
    results += met_results

    # all-hadronic veto
    events, all_had_veto_results = self[all_had_veto](events, **kwargs)
    results += all_had_veto_results

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.main["event"] = event_sel

    for step, sel in results.steps.items():
        n_sel = ak.sum(sel, axis=-1)
        print(f"{step}: {n_sel}")

    n_sel = ak.sum(event_sel, axis=-1)
    print(f"__all__: {n_sel}")

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add cutflow features
    events = self[cutflow_features](events, results=results, **kwargs)

    # increment stats
    self[increment_stats](events, event_sel, stats, **kwargs)

    return events, results
