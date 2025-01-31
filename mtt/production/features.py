# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production.util import attach_coffea_behavior
from mtt.production.ttbar_reco import choose_lepton
from mtt.selection.util import masked_sorted_indices

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer
def jet_energy_shifts(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Pseudo-producer that registers jet energy shifts.
    """
    return events


@jet_energy_shifts.init
def jet_energy_shifts_init(self: Producer) -> None:
    """
    Register shifts.
    """
    self.shifts |= {
        f"jec_{junc_name}_{junc_dir}"
        for junc_name in self.config_inst.x.jec.Jet.uncertainty_sources
        for junc_dir in ("up", "down")
    } | {"jer_up", "jer_down"}


@producer(
    uses={"Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass"},
    produces={"dijet_mass", "dijet_delta_r"},
)
def jj_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Jet"] = ak.with_name(events.Jet, "PtEtaPhiMLorentzVector")

    # ensure at least 2 jets (pad with None if nonexistent)
    jets = ak.pad_none(events.Jet, 2)

    # calculate and save invariant mass
    dijet_mass = (jets[:, 0] + jets[:, 1]).mass
    events = set_ak_column(events, "dijet_mass", dijet_mass)

    # calculate and save delta-R
    dijet_delta_r = jets[:, 0].delta_r(jets[:, 1])
    events = set_ak_column(events, "dijet_delta_r", dijet_delta_r)

    return events


@producer(
    uses={
        attach_coffea_behavior,
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        choose_lepton,
    },
    produces={
        attach_coffea_behavior,
        "jet_lep_pt_rel", "jet_lep_delta_r",
    },
)
def jet_lepton_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produces jet lepton pTrel and deltaR.
    """
    # load coffea behaviors for simplified arithmetic with vectors
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Jet"] = ak.with_name(events.Jet, "PtEtaPhiMLorentzVector")
    events["Electron"] = ak.with_name(events.Electron, "PtEtaPhiMLorentzVector")
    events["Muon"] = ak.with_name(events.Muon, "PtEtaPhiMLorentzVector")

    # select jets with pT > 15
    jets_mask = (events.Jet.pt > 15)
    jets = events.Jet[jets_mask]

    # select lepton of event
    events = self[choose_lepton](events, **kwargs)
    lepton = events["Lepton"]

    # attach lorentz vector behavior to lepton
    lepton = ak.with_name(lepton, "PtEtaPhiMLorentzVector")

    # calculate deltaR between lepton and every jet
    lepton_jet_deltar = ak.firsts(jets.metric_table(lepton), axis=-1)

    # define closest lepton to jet
    lepton_closest_jet = ak.firsts(
        jets[masked_sorted_indices(jets_mask, lepton_jet_deltar, ascending=True)],
    )

    # calculate pTrel and deltaR
    # convert vectors to 3D vectors (due to bug in 'vector' library?)
    # jet_lep_pt_rel = lepton.cross(lepton_closest_jet).pt / lepton_closest_jet.p  # FIXME pt or p?
    lepton_3d = lepton.to_Vector3D()
    lepton_closest_jet_3d = lepton_closest_jet.to_Vector3D()
    jet_lep_pt_rel = lepton_3d.cross(lepton_closest_jet_3d).p / lepton_closest_jet.p
    jet_lep_delta_r = lepton_closest_jet.delta_r(lepton)

    # save as new columns
    events = set_ak_column(events, "jet_lep_pt_rel", ak.fill_none(jet_lep_pt_rel, EMPTY_FLOAT))
    events = set_ak_column(events, "jet_lep_delta_r", ak.fill_none(jet_lep_delta_r, EMPTY_FLOAT))

    return events


@producer(
    uses={
        attach_coffea_behavior,
        jj_features,
        jet_lepton_features,
        "Electron.pt", "Muon.pt", "FatJet.pt", "Jet.pt",
    },
    produces={
        attach_coffea_behavior,
        jj_features,
        jet_lepton_features,
        "ht",
        "n_jet",
        "n_fatjet",
        "n_muon",
        "n_electron",
    },
    shifts={
        jet_energy_shifts,
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """All high-level featues, e.g. scalar jet pt sum (ht), number of jets, electrons, muons, etc."""

    # note: the missing Lorentz vector columns cause issues for
    # arrays with registered names/behaviors, so we remove them
    # before evaluating `ak.num`
    jet = ak.without_parameters(events.Jet)
    fatjet = ak.without_parameters(events.FatJet)
    muon = ak.without_parameters(events.Muon)
    electron = ak.without_parameters(events.Electron)

    # count objects using `pt` fields as a proxy
    events = set_ak_column(events, "n_jet", ak.num(jet.pt, axis=-1))
    events = set_ak_column(events, "n_fatjet", ak.num(fatjet.pt, axis=-1))
    events = set_ak_column(events, "n_muon", ak.num(muon.pt, axis=-1))
    events = set_ak_column(events, "n_electron", ak.num(electron.pt, axis=-1))

    # count objects using `pt` fields as a proxy
    events = set_ak_column(events, "ht", ak.sum(jet.pt, axis=-1))

    # dijet features
    events = self[jj_features](events, **kwargs)

    # jet lepton features
    events = self[jet_lepton_features](events, **kwargs)

    return events
