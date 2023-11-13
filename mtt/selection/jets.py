# coding: utf-8

"""
Selection involving jets and MET.
"""

from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector

from mtt.selection.util import masked_sorted_indices
from mtt.selection.general import jet_energy_shifts
from mtt.selection.lepton import lepton_selection

from mtt.production.lepton import choose_lepton

np = maybe_import("numpy")
ak = maybe_import("awkward")

@selector(
    uses={
        "Jet.pt", "Jet.eta", "Jet.btagDeepFlavB",
        "Jet.electronIdx1", "Jet.electronIdx2",
        "Jet.muonIdx1", "Jet.muonIdx2",
        "channel_id",
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
    # electron channel:
    # - require at least one AK4 jet with pt>50 and abseta<2.5
    # - require a second AK4 jet with pt>40 and abseta<2.5
    # muon channel:
    # - require at least one AK4 jet with pt>50 and abseta<2.5
    # - require a second AK4 jet with pt>50 and abseta<2.5
    # both channels:
    # - require that at least one AK4 jet (pt>50 and abseta<2.5) is b-tagged
    ch_ids = events.channel_id

    ch_e = self.config_inst.get_channel("e")
    ch_mu = self.config_inst.get_channel("mu") 

    el_id = (ch_ids == ch_e.id)
    mu_id = (ch_ids == ch_mu.id)

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
    # subleading jet pt > 40
    jet = ak.pad_none(events.Jet[jet_indices], 2)
    el_jet_sel = (
        (jet[:, 0].pt > 50) &
        (jet[:, 1].pt > 40)
    )
    el_jet_sel = ak.fill_none(el_jet_sel, False)

    # at least two jets, leading jet pt > 50,
    # subleading jet pt > 50
    mu_jet_sel = (
        (jet[:, 0].pt > 50) &
        (jet[:, 1].pt > 50)
    )
    mu_jet_sel = ak.fill_none(mu_jet_sel, False)

    el_sel_mask = el_id & el_jet_sel
    mu_sel_mask = mu_id & mu_jet_sel

    sel_jet = el_sel_mask | mu_sel_mask

    # MISSING: match AK4 PUPPI jets to AK4 CHS jets for b-tagging

    # b-tagged jets, DeepJet medium working point
    wp_med = self.config_inst.x.btag_working_points.deepjet.medium
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
        "channel_id",
    },
    exposed=True,
)
def met_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """Missing transverse momentum (MET) selection."""
    MET_pt = events.MET.pt
    ch_ids = events.channel_id

    ch_e = self.config_inst.get_channel("e")
    ch_mu = self.config_inst.get_channel("mu") 

    el_id = (ch_ids == ch_e.id)
    mu_id = (ch_ids == ch_mu.id)

    el_sel = MET_pt > 60
    mu_sel = MET_pt > 70

    el_sel_mask = el_id & el_sel
    mu_sel_mask = mu_id & mu_sel

    sel_met = el_sel_mask | mu_sel_mask
    sel_met = ak.fill_none(sel_met, False)

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