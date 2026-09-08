# coding: utf-8

"""
Selectors to set ak columns for cutflow features
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route, EMPTY_FLOAT
from columnflow.selection import Selector, SelectionResult, selector

ak = maybe_import("awkward")


@selector(
    uses={
        "Jet.pt", "Jet.eta", "FatJet.pt", "FatJet.eta",
    },
    produces={
        "cutflow.jet1_pt", "cutflow.jet2_pt", "cutflow.jet3_pt", "cutflow.jet4_pt",
        "cutflow.jet1_eta", "cutflow.jet2_eta", "cutflow.jet3_eta", "cutflow.jet4_eta",
        "cutflow.fatjet1_pt", "cutflow.fatjet2_pt", "cutflow.fatjet3_pt", "cutflow.fatjet4_pt",
        "cutflow.fatjet1_eta", "cutflow.fatjet2_eta", "cutflow.fatjet3_eta", "cutflow.fatjet4_eta",
        "cutflow.bjet1_pt", "cutflow.bjet2_pt", "cutflow.bjet3_pt", "cutflow.bjet4_pt",
        "cutflow.bjet1_eta", "cutflow.bjet2_eta", "cutflow.bjet3_eta", "cutflow.bjet4_eta",
        "cutflow.lightjet1_pt", "cutflow.lightjet2_pt", "cutflow.lightjet3_pt", "cutflow.lightjet4_pt",
        "cutflow.lightjet1_eta", "cutflow.lightjet2_eta", "cutflow.lightjet3_eta", "cutflow.lightjet4_eta",
        "cutflow.muon_pt", "cutflow.muon_eta", "cutflow.vetomuon_pt", "cutflow.vetomuon_eta",
        "cutflow.electron_pt", "cutflow.electron_eta", "cutflow.vetoelectron_pt", "cutflow.vetoelectron_eta",
        "cutflow.n_jet", "cutflow.n_bjet", "cutflow.n_lightjet",
        "cutflow.n_toptag", "cutflow.n_toptag_delta_r_lepton",
        "cutflow.n_muon", "cutflow.n_electron", "cutflow.n_veto_muon", "cutflow.n_veto_electron",
    },
)
def cutflow_features(self: Selector, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:

    # jet properties
    for jet_name in ["Jet", "FatJet", "BJet", "LightJet"]:
        if jet_name == "FatJet":
            jet_base_name = "FatJet"
        else:
            jet_base_name = "Jet"
        jet_indices = results.objects[jet_base_name][jet_name]
        jets = events[jet_base_name][jet_indices]
        for i in range(4):
            for var in ("pt", "eta"):
                events = set_ak_column(
                    events,
                    f"cutflow.{jet_name.lower()}{i+1}_{var}",
                    Route(f"{var}[:, {i}]").apply(jets, EMPTY_FLOAT),
                )

    # pt-leading electron/muon properties
    for lepton_name in ["Muon", "Electron", "VetoMuon", "VetoElectron"]:
        lepton_base_name = "Muon" if "Muon" in lepton_name else "Electron"
        lepton_indices = results.objects[lepton_base_name][lepton_name]
        leptons = events[lepton_base_name][lepton_indices]
        for var in ("pt", "eta"):
            events = set_ak_column(
                events,
                f"cutflow.{lepton_name.lower()}_{var}",
                Route(f"{var}[:, 0]").apply(leptons, EMPTY_FLOAT),
            )

    # count number of objects after appyling selection
    events = set_ak_column(events, "cutflow.n_bjet", ak.num(results.objects.Jet.BJet, axis=-1))
    events = set_ak_column(events, "cutflow.n_lightjet", ak.num(results.objects.Jet.LightJet, axis=-1))
    events = set_ak_column(events, "cutflow.n_jet", events.cutflow.n_bjet + events.cutflow.n_lightjet)

    events = set_ak_column(events, "cutflow.n_toptag", ak.num(results.objects.FatJet.FatJetTopTag, axis=-1))
    events = set_ak_column(
        events,
        "cutflow.n_toptag_delta_r_lepton",
        ak.num(results.objects.FatJet.FatJetTopTagDeltaRLepton, axis=-1),
    )

    events = set_ak_column(events, "cutflow.n_muon", ak.num(results.objects.Muon.Muon, axis=-1))
    events = set_ak_column(events, "cutflow.n_veto_muon", ak.num(results.objects.Muon.VetoMuon, axis=-1))
    events = set_ak_column(events, "cutflow.n_electron", ak.num(results.objects.Electron.Electron, axis=-1))
    events = set_ak_column(events, "cutflow.n_veto_electron", ak.num(results.objects.Electron.VetoElectron, axis=-1))

    if self.dataset_inst.is_mc and not self.dataset_inst.has_tag("is_diboson"):
        events = set_ak_column(events, "cutflow.lhe_ht", events.LHE.HT)

    return events


@cutflow_features.init
def cutflow_features_init(self: Selector) -> None:

    if (
        hasattr(self, "dataset_inst") and
        self.dataset_inst.is_mc and
        not self.dataset_inst.has_tag("is_diboson")
    ):
        self.uses |= {"LHE.HT"}
        self.produces |= {"cutflow.lhe_ht"}
