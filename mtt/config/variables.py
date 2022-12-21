# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )

    # Event properties
    config.add_variable(
        name="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
    )
    config.add_variable(
        name="n_bjet",
        binning=(11, -0.5, 10.5),
        x_title="Number of b jets",
    )
    config.add_variable(
        name="n_lightjet",
        binning=(11, -0.5, 10.5),
        x_title="Number of light jets",
    )
    config.add_variable(
        name="n_electron",
        binning=(11, -0.5, 10.5),
        x_title="Number of electrons",
    )
    config.add_variable(
        name="n_muon",
        binning=(11, -0.5, 10.5),
        x_title="Number of muons",
    )
    config.add_variable(
        name="cat_id",
        expression="category_ids",
        binning=(3, 0.5, 3.5),
        x_title="category id",
    )
    config.add_variable(
        name="chan_id",
        expression="channel_id",
        binning=(3, -0.5, 2.5),
        x_title="channel id",
    )

    # Object properties

    # Jets (4 pt-leading jets)
    for i in range(4):
        for obj in ("Jet", "FatJet"):
            config.add_variable(
                name=f"{obj.lower()}{i+1}_pt",
                expression=f"{obj}.pt[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, 0., 400.),
                unit="GeV",
                x_title=rf"{obj} {i+1} $p_{{T}}$",
            )
            config.add_variable(
                name=f"{obj.lower()}{i+1}_eta",
                expression=f"{obj}.eta[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(50, -2.5, 2.5),
                x_title=rf"{obj} {i+1} $\eta$",
            )
            config.add_variable(
                name=f"{obj.lower()}{i+1}_phi",
                expression=f"{obj}.phi[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, -3.2, 3.2),
                x_title=rf"{obj} {i+1} $\phi$",
            )

    # Leptons
    for obj in ["Electron", "Muon"]:
        config.add_variable(
            name=f"{obj.lower()}_pt",
            expression=f"{obj}.pt[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0., 400.),
            unit="GeV",
            x_title=obj + r" $p_{T}$",
        )
        config.add_variable(
            name=f"{obj.lower()}_pt_more",
            expression=f"{obj}.pt[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(70, 0., 900.),
            unit="GeV",
            x_title=obj + r" $p_{T}$",
        )
        config.add_variable(
            name=f"{obj.lower()}_phi",
            expression=f"{obj}.phi[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=obj + r" $\phi$",
        )
        config.add_variable(
            name=f"{obj.lower()}_eta",
            expression=f"{obj}.eta[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(50, -2.5, 2.5),
            x_title=obj + r" $\eta$",
        )
        config.add_variable(
            name=f"{obj.lower()}_mass",
            expression=f"{obj}.mass[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=obj + " mass",
        )

    # MET
    config.add_variable(
        name="met_pt",
        expression="MET.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"MET $p_{T}$",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"MET $\phi$",
    )

    # jj features
    config.add_variable(
        name="avgpt_jj",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$\bar{p}_{T}(j_{1},j_{2})$",
    )
    config.add_variable(
        name="m_jj",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$m_{jj}$",
    )
    config.add_variable(
        name="deltaR_jj",
        binning=(40, 0, 5),
        x_title=r"$\Delta R(j_{1},j_{2})$",
    )
    config.add_variable(
        name="deltaeta_jj",
        binning=(40, 0, 5),
        x_title=r"$\Delta \eta(j_{1},j_{2})$",
    )
    config.add_variable(
        name="deltaphi_jj",
        binning=(40, 0, 3.2),
        x_title=r"$\Delta \phi(j_{1},j_{2})$",
    )

    # cutflow variables

    # Jet properties
    for obj in ("Jet", "FatJet"):
        for i in range(4):
            config.add_variable(
                name=f"cf_{obj.lower()}{i+1}_pt",
                expression=f"cutflow.{obj.lower()}{i+1}_pt",
                binning=(40, 0., 400.),
                unit="GeV",
                x_title=rf"{obj} {i+1} $p_{{T}}$",
            )
            config.add_variable(
                name=f"cf_{obj.lower()}{i+1}_eta",
                expression=f"cutflow.{obj.lower()}{i+1}_eta",
                binning=(50, -2.5, 2.5),
                x_title=rf"{obj} {i+1} $\eta$",
            )

    for obj in ["Electron", "Muon"]:
        config.add_variable(
            name=f"cf_{obj.lower()}_pt",
            expression=f"cutflow.{obj.lower()}_pt",
            binning=(40, 0., 400.),
            unit="GeV",
            x_title=rf"{obj} $p_{{T}}$",
        )
        config.add_variable(
            name=f"cf_{obj.lower()}_eta",
            expression=f"cutflow.{obj.lower()}_eta",
            binning=(50, -2.5, 2.5),
            x_title=rf"{obj} $\eta$",
        )

    # Jet multiplicity
    config.add_variable(
        name="cf_n_jet",
        expression="cutflow.n_jet",
        binning=(11, -0.5, 10.5),
        x_title=r"Number of jets ($p_{T}$ > 30 GeV)",
    )
    config.add_variable(
        name="cf_n_electron",
        expression="cutflow.n_electron",
        binning=(5, -0.5, 4.5),
        x_title=r"Number of electrons",
    )
    config.add_variable(
        name="cf_n_muon",
        expression="cutflow.n_muon",
        binning=(5, -0.5, 4.5),
        x_title=r"Number of muons",
    )
