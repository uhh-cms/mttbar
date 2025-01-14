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
    sel_params = self.config_inst.x.jet_selection.ak4
    jet = events[sel_params.column]

    ch_ids = events.channel_id

    ch_e = self.config_inst.get_channel("e")
    ch_mu = self.config_inst.get_channel("mu")

    el_id = (ch_ids == ch_e.id)
    mu_id = (ch_ids == ch_mu.id)

    # loose jets (pt>0.1) - to filter out cleaned jets etc.
    loose_jet_mask = (jet.pt > 0.1)
    loose_jet_indices = masked_sorted_indices(loose_jet_mask, jet.pt)

    # jets (pt>30)
    jet_mask = (
        (abs(jet.eta) < sel_params.max_abseta) &
        (jet.pt > sel_params.min_pt.baseline)
    )
    jet_indices = masked_sorted_indices(jet_mask, jet.pt)

    # at least two jets, leading jet pt > 50,
    # subleading jet pt > 40
    leading_jets = ak.pad_none(jet[jet_indices], 2)
    el_jet_sel = (
        (leading_jets[:, 0].pt > sel_params.min_pt.e[0]) &
        (leading_jets[:, 1].pt > sel_params.min_pt.e[1])
    )
    el_jet_sel = ak.fill_none(el_jet_sel, False)

    # at least two jets, leading jet pt > 50,
    # subleading jet pt > 50
    mu_jet_sel = (
        (leading_jets[:, 0].pt > sel_params.min_pt.mu[0]) &
        (leading_jets[:, 1].pt > sel_params.min_pt.mu[1])
    )
    mu_jet_sel = ak.fill_none(mu_jet_sel, False)

    el_sel_mask = el_id & el_jet_sel
    mu_sel_mask = mu_id & mu_jet_sel

    sel_jet = el_sel_mask | mu_sel_mask

    # MISSING: match AK4 PUPPI jets to AK4 CHS jets for b-tagging

    # b-tagged jets, DeepJet medium working point
    bjet_mask = (jet_mask) & (jet[sel_params.btagger.column] >= sel_params.btagger.wp)
    lightjet_mask = (jet_mask) & (jet[sel_params.btagger.column] < sel_params.btagger.wp)
    sel_bjet = ak.sum(bjet_mask, axis=-1) >= 1

    # indices of the b-tagged and non-b-tagged (light) jets
    bjet_indices = masked_sorted_indices(bjet_mask, jet.pt)
    lightjet_indices = masked_sorted_indices(lightjet_mask, jet.pt)

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


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return
    params = config_inst.x.jet_selection.ak4
    column = params.get("column")
    if column:
        self.uses |= {
            f"{column}.pt",
            f"{column}.eta",
            f"{column}.{params.btagger.column}",
        }


@selector(
    uses={
        choose_lepton,
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
    sel_params = self.config_inst.x.jet_selection.ak8
    fatjet = events[sel_params.column]

    # top-tagger working point
    wp_top_md = sel_params.toptagger.wp

    # top-tagging criteria
    fatjet_mask_toptag = (
        # kinematic cuts
        (fatjet.pt > sel_params.min_pt) &
        (abs(fatjet.eta) < sel_params.max_abseta) &
        # 1st topjet requirement: top tagger working point
        (fatjet[sel_params.toptagger.column] > wp_top_md) &
        # 2nd topjet requirement: softdrop mass window
        (fatjet.msoftdrop > sel_params.msoftdrop[0]) &
        (fatjet.msoftdrop < sel_params.msoftdrop[1])
    )
    fatjet_indices_toptag = masked_sorted_indices(
        fatjet_mask_toptag,
        fatjet.pt,
    )

    # veto events with more than one top-tagged AK8 jet
    sel_all_had_veto = (ak.sum(fatjet_mask_toptag, axis=-1) < 2)

    # get main lepton
    events = self[choose_lepton](events, **kwargs)
    lepton = events["Lepton"]

    # separation from main lepton
    delta_r_fatjet_lepton = ak.firsts(fatjet.metric_table(lepton), axis=-1)
    fatjet_mask_toptag_delta_r_lepton = (
        fatjet_mask_toptag &
        # pass if no main lepton exists
        ak.fill_none(delta_r_fatjet_lepton > sel_params.delta_r_lep, True)
    )
    fatjet_indices_toptag_delta_r_lepton = masked_sorted_indices(
        fatjet_mask_toptag_delta_r_lepton,
        fatjet.pt,
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


@top_tagged_jets.init
def top_tagged_jets_init(self: Selector) -> None:
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return
    params = config_inst.x.jet_selection.ak8
    column = params.get("column")
    if column:
        self.uses |= {
            f"{column}.pt",
            f"{column}.eta",
            f"{column}.phi",
            f"{column}.mass",
            f"{column}.msoftdrop",
            f"{column}.{params.toptagger.column}",
        }


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
    sel_params = self.config_inst.x.met_selection
    met = events[sel_params.column]

    MET_pt = met['pt']
    ch_ids = events.channel_id

    ch_e = self.config_inst.get_channel("e")
    ch_mu = self.config_inst.get_channel("mu")

    el_id = (ch_ids == ch_e.id)
    mu_id = (ch_ids == ch_mu.id)

    el_sel = MET_pt > sel_params.min_pt.e
    mu_sel = MET_pt > sel_params.min_pt.mu

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
    sel_params = self.config_inst.x.lepton_jet_iso

    # ensure lepton selection was run
    events, _ = self[lepton_selection](events, **kwargs)

    # select jets
    jets_mask = (events.Jet.pt > sel_params.min_pt)
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
        sel = ak.all(lepton_jet_deltar > sel_params.min_delta_r, axis=-1)

        # but keep events where the perpendicular lepton momentum relative
        # to the jet is sufficiently large
        # convert vectors to 3D vectors (due to bug in 'vector' library?)
        # pt_rel = leptons.cross(lepton_closest_jet).p / lepton_closest_jet.p
        lepton_3d = leptons.to_Vector3D()
        jet_3d = lepton_closest_jet.to_Vector3D()
        pt_rel = lepton_3d.cross(jet_3d).p / jet_3d.p
        sel = ak.where(
            pt_rel > sel_params.min_pt_rel,
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
