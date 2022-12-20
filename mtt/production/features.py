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
import numpy as np

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
    produces={"avgpt_jj", "m_jj", "deltaR_jj", "deltaeta_jj", "deltaphi_jj"},
)
def jj_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Jet"] = ak.with_name(events.Jet, "PtEtaPhiMLorentzVector")

    # ensure at least 2 jets (pad with None if nonexistent)
    jets = ak.pad_none(events.Jet, 2)

    # q=__import__("functools").partial(__import__("os")._exit,0)
    # __import__("IPython").embed()

    # calculate and save average pT
    avgpt_jj = (jets[:, 0].pt + jets[:, 1].pt)/2
    events = set_ak_column(events, "avgpt_jj", avgpt_jj)

    # calculate and save invariant mass
    m_jj = (jets[:, 0] + jets[:, 1]).mass
    events = set_ak_column(events, "m_jj", m_jj)

    # calculate and save delta-R
    deltaR_jj = jets[:, 0].delta_r(jets[:, 1])
    events = set_ak_column(events, "deltaR_jj", deltaR_jj)

    # calculate and save delta eta
    deltaeta_jj = abs((jets[:, 0].eta - jets[:, 1].eta))
    events = set_ak_column(events, "deltaeta_jj", deltaeta_jj)

    # calculate and save delta phi
    deltaphi_jj = np.pi - abs(abs(jets[:, 0].phi - jets[:, 1].phi) - np.pi)
    events = set_ak_column(events, "deltaphi_jj", deltaphi_jj)

    return events


@producer(
    uses={
        attach_coffea_behavior,
        jj_features,
        "Electron.pt", "Muon.pt", "FatJet.pt", "Jet.pt", "Bjet.pt",
    },
    produces={
        attach_coffea_behavior,
        jj_features,
        "ht",
        "n_jet",
        "n_bjet",
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
    events = set_ak_column(events, "n_bjet", ak.num(events.Bjet.pt, axis=-1))
    events = set_ak_column(events, "n_fatjet", ak.num(events.FatJet.pt, axis=-1))
    events = set_ak_column(events, "n_muon", ak.num(events.Muon.pt, axis=-1))
    events = set_ak_column(events, "n_electron", ak.num(events.Electron.pt, axis=-1))

    events = set_ak_column(events, "ht", ak.sum(events.Jet.pt, axis=-1))

    # dijet features
    events = self[jj_features](events, **kwargs)

    return events
