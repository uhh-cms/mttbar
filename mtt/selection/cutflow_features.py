# coding: utf-8

"""
Selectors to set ak columns for cutflow features
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route, EMPTY_FLOAT
from columnflow.selection import Selector, SelectionResult, selector

ak = maybe_import("awkward")


@selector(
    uses={"Jet.pt", "FatJet.pt"},
    produces={
        "cutflow.jet_ak4_1_pt", "cutflow.jet_ak4_2_pt", "cutflow.jet_ak4_3_pt", "cutflow.jet_ak4_4_pt",
        "cutflow.jet_ak8_1_pt", "cutflow.jet_ak8_2_pt", "cutflow.jet_ak8_3_pt", "cutflow.jet_ak8_4_pt",
        "cutflow.n_jet_30", "cutflow.n_jet_50", "cutflow.n_muon",
    },
)
def cutflow_features(self: Selector, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:

    # determine jet pt before applying jet pt cut (and ideally after applying eta cut?)i
    for jet_name, jet_cutflow_name in [('Jet', 'jet_ak4'), ('FatJet', 'jet_ak8')]:
        jet_indices = ak.argsort(events[jet_name].pt, ascending=False)
        jets = events[jet_name][jet_indices]
        for i in range(4):
            events = set_ak_column(
                events,
                f"cutflow.{jet_cutflow_name}_{i+1}_pt",
                Route(f"pt[:, {i}]").apply(jets, EMPTY_FLOAT),
            )

    # Number of objects should be counted after appyling
    events = set_ak_column(events, "cutflow.n_jet_50", ak.num(results.objects.Jet.Jet50, axis=1))
    events = set_ak_column(events, "cutflow.n_jet_30", ak.num(results.objects.Jet.Jet30, axis=1))
    events = set_ak_column(events, "cutflow.n_muon", ak.num(results.objects.Muon.Muon, axis=1))

    return events
