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

    # ttbar features
    config.add_variable(
        name=f"chi2",
        expression=f"TTbar.chi2",
        binning=(100, 0, 600),
        x_title=rf"$\chi^2$",
    )
    config.add_variable(
        name=f"chi2_lt30",
        expression=f"TTbar.chi2",
        binning=(15, 0, 30),
        x_title=rf"$\chi^2$",
    )
    config.add_variable(
        name=f"chi2_lt100",
        expression=f"TTbar.chi2",
        binning=(20, 0, 100),
        x_title=rf"$\chi^2$",
    )
    config.add_variable(
        name=f"ttbar_mass",
        expression=f"TTbar.mass",
        binning=[
            0, 400, 600, 800, 1000, 1200, 1400,
            1600, 1800, 2000, 2200, 2400, 2600,
            2800, 3000, 3200, 3400, 3600, 3800,
            4000, 4400, 4800, 5200, 5600, 6000,
        ],
        x_title=r"$m({t}\overline{t})$",
    )
    config.add_variable(
        name=f"ttbar_mass_narrow",
        expression=f"TTbar.mass",
        binning=(100, 400, 4400),
        x_title=r"$m({t}\overline{t})$",
    )
    config.add_variable(
        name=f"cos_theta_star",
        expression=f"TTbar.cos_theta_star",
        binning=(100, -1, 1),
        x_title=r"${cos}(\theta^{*})$",
    )
    config.add_variable(
        name=f"abs_cos_theta_star",
        expression=f"TTbar.abs_cos_theta_star",
        binning=(50, 0, 1),
        x_title=r"$|{cos}(\theta^{*})|$",
    )
    for decay in ('had', 'lep'):
        config.add_variable(
            name=f"top_{decay}_mass",
            expression=f"TTbar.top_{decay}_mass",
            binning=(100, 0, 700),
            unit="GeV",
            x_title=rf"$M_{{t}}^{{{decay}}}$",
        )
    for decay in ('had', 'lep'):
        config.add_variable(
            name=f"n_jet_{decay}",
            expression=f"n_jet_{decay}",
            binning=(11, -0.5, 10.5),
        )
    config.add_variable(
        name="n_jet_sum",
        expression="n_jet_sum",
        binning=(11, -0.5, 10.5),
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
        x_title=r"Number of jets ($p_{T}$ > 30 GeV, $|\eta| < 2.5$)",
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
    config.add_variable(
        name="cf_n_toptag",
        expression="cutflow.n_toptag",
        binning=(5, -0.5, 4.5),
        x_title=r"Number of top-tagged AK8 jets",
    )
