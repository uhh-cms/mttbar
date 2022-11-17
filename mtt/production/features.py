# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.mc_weight import mc_weight
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
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
    produces={"m_jj", "deltaR_jj"},
)
def jj_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Jet"] = ak.with_name(events.Jet, "PtEtaPhiMLorentzVector")

    if ak.any(ak.num(events.Jet, axis=-1) <= 2):
        print("In features.py: there should be at least 2 jets in each event")
        from IPython import embed; embed()
        raise Exception("In features.py: there should be at least 2 jets in each event")

    m_jj = (events.Jet[:, 0] + events.Jet[:, 1]).mass
    events = set_ak_column(events, "m_jj", m_jj)

    deltaR_jj = events.Jet[:, 0].delta_r(events.Jet[:, 1])
    events = set_ak_column(events, "deltaR_jj", deltaR_jj)

    return events


@producer(
    uses={
        attach_coffea_behavior,
        jj_features,
        "Muon.pt", "Jet.pt", "BJet.pt",
    },
    produces={
        attach_coffea_behavior,
        jj_features,
        "ht",
        "n_jet_ak4",
        "n_jet_ak8",
        "n_muon",
    },
    shifts={
        jet_energy_shifts,
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """All high-level featues, e.g. scalar jet pt sum (ht), number of jets, electrons, muons, etc."""

    events = set_ak_column(events, "n_jet", ak.num(events.Jet.pt, axis=-1))
    events = set_ak_column(events, "n_electron", ak.num(events.Electron.pt, axis=-1))
    events = set_ak_column(events, "n_muon", ak.num(events.Muon.pt, axis=-1))

    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=-1))

    # dijet features
    events = self[jj_features](events, **kwargs)

    return events
