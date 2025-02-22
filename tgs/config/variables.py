# coding: utf-8

"""
Definition of variables for trigger study analysis (tgs).
"""
import order as od

from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT


ak = maybe_import("awkward")


def add_variables(config: od.Config) -> None:
    """
    Add trigger-study-specific variables to a *config*.
    """

    # (the "event", "run" and "lumi" variables are required for some cutflow plotting task,
    # and also correspond to the minimal set of columns that coffea's nano scheme requires)
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )

    #
    # jets
    #

    config.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    config.add_variable(
        name="jets_pt",
        expression="Jet.pt",
        binning=(80, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_{T} of all jets$",
    )
    config.add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(80, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="ht",
        expression=lambda events: ak.sum(events.Jet.pt, axis=1),
        binning=(40, 0.0, 800.0),
        unit="GeV",
        x_title="HT",
    )

    #
    # lepton properties
    #

    # -- muon

    config.add_variable(
        name="muon_pt",
        expression="Muon.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(80, 0.0, 400.0),
        unit="GeV",
        x_title=r"Muon $p_{T}$",
    )
    config.add_variable(
        name="muon_eta",
        expression="Muon.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=[-2.5, -1.57, -1.44, -1.3, -0.7, 0, 0.7, 1.3, 1.44, 1.57, 2.5],
        x_title=r"Muon $\eta$",
    )
    config.add_variable(
        name="muon_eta_fine",
        expression="Muon.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, -2.5, 2.5),
        x_title=r"Muon $\eta$",
    )

    # -- electron

    config.add_variable(
        name="electron_pt",
        expression="Electron.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(80, 0.0, 400.0),
        unit="GeV",
        x_title=r"Electron $p_{T}$",
    )
    config.add_variable(
        name="electron_eta",
        expression="Electron.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=[-2.5, -1.57, -1.44, -1.3, -0.7, 0, 0.7, 1.3, 1.44, 1.57, 2.5],
        x_title=r"Electron $\eta$",
    )
    config.add_variable(
        name="electron_eta_fine",
        expression="Electron.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(50, -2.5, 2.5),
        x_title=r"Electron $\eta$",
    )

    # weights
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    # cutflow variables
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="cf_electron_pt",
        expression="cutflow.electron_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Electron $p_{T}$",
    )
    config.add_variable(
        name="cf_muon_pt",
        expression="cutflow.muon_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Muon $p_{T}$",
    )
