# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior

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
        for junc_name in self.config_inst.x.jec.uncertainty_sources
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
        jj_features,
        "Electron.pt", "Muon.pt", "FatJet.pt", "Jet.pt",
    },
    produces={
        attach_coffea_behavior,
        jj_features,
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

    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=-1))
    events = set_ak_column(events, "n_fatjet", ak.num(events.FatJet.pt, axis=-1))
    events = set_ak_column(events, "n_muon", ak.num(events.Muon.pt, axis=-1))
    events = set_ak_column(events, "n_electron", ak.num(events.Electron.pt, axis=-1))

    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=-1))

    # dijet features
    events = self[jj_features](events, **kwargs)

    return events
