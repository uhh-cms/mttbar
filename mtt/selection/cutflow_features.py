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
        "cutflow.muon_pt", "cutflow.muon_eta",
        "cutflow.electron_pt", "cutflow.electron_eta",
        "cutflow.n_jet", "cutflow.n_bjet", "cutflow.n_lightjet",
        "cutflow.n_toptag", "cutflow.n_toptag_delta_r_lepton",
        "cutflow.n_muon", "cutflow.n_electron",
    },
)
def cutflow_features(self: Selector, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:

    # jet properties
    for jet_name in ["Jet", "FatJet"]:
        jet_indices = ak.argsort(events[jet_name].pt, ascending=False)
        jets = events[jet_name][jet_indices]
        for i in range(4):
            for var in ("pt", "eta"):
                events = set_ak_column(
                    events,
                    f"cutflow.{jet_name.lower()}{i+1}_{var}",
                    Route(f"{var}[:, {i}]").apply(jets, EMPTY_FLOAT),
                )

    # pt-leading electron/muon properties
    for lepton_name in ["Muon", "Electron"]:
        lepton_indices = results.objects[lepton_name][lepton_name]
        leptons = events[lepton_name][lepton_indices]
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
    events = set_ak_column(events, "cutflow.n_electron", ak.num(results.objects.Electron.Electron, axis=-1))

    if not self.dataset_inst.x.is_data and not self.dataset_inst.x.is_diboson:
        events = set_ak_column(events, "cutflow.lhe_ht", events.LHE.HT)

    return events


@cutflow_features.init
def cutflow_features_init(self: Selector) -> None:

    if hasattr(self, "dataset_inst") and not any(
        getattr(self.dataset_inst.x, flag, None)
        for flag in ("is_diboson", "is_data")
    ):
        self.uses |= {"LHE.HT"}
        self.produces |= {"cutflow.lhe_ht"}
