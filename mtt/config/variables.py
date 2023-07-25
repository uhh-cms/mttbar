# coding: utf-8

"""
Definition of variables.
"""

import itertools
import numpy as np
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
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0, 1e9),
        x_title="Event ID",
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
        name="dijet_mass",
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"$m_{jj}$",
    )
    config.add_variable(
        name="dijet_delta_r",
        binning=(40, 0, 5),
        x_title=r"$\Delta R(j_{1},j_{2})$",
    )

    # jet lepton features
    config.add_variable(
        name="jet_lep_pt_rel",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_{T}^{rel}$",
    )
    config.add_variable(
        name="jet_lep_delta_r",
        binning=(40, 0, 5),
        x_title=r"$\Delta R(jet, lep)$",
    )
    config.add_variable(
        name="jet_lep_pt_rel_zoom",
        expression="jet_lep_pt_rel",
        binning=(10, 0, 50),
        unit="GeV",
        x_title=r"$p_{T}^{rel}$",
    )
    config.add_variable(
        name="jet_lep_delta_r_zoom",
        expression="jet_lep_delta_r",
        binning=(15, 0, 1.5),
        x_title=r"$\Delta R(jet, lep)$",
    )

    # ttbar features
    config.add_variable(
        name="chi2",
        expression="TTbar.chi2",
        binning=(100, 0, 600),
        x_title=r"$\chi^2$",
    )
    config.add_variable(
        name="chi2_lt30",
        expression="TTbar.chi2",
        binning=(15, 0, 30),
        x_title=r"$\chi^2$",
    )
    config.add_variable(
        name="chi2_lt100",
        expression="TTbar.chi2",
        binning=(20, 0, 100),
        x_title=r"$\chi^2$",
    )
    config.add_variable(
        name="ttbar_mass",
        expression="TTbar.mass",
        binning=[
            0, 400, 600, 800, 1000, 1200, 1400,
            1600, 1800, 2000, 2200, 2400, 2600,
            2800, 3000, 3200, 3400, 3600, 3800,
            4000, 4400, 4800, 5200, 5600, 6000,
        ],
        unit="GeV",
        x_title=r"$m({t}\overline{t})$",
    )
    config.add_variable(
        name="ttbar_mass_narrow",
        expression="TTbar.mass",
        binning=(100, 400, 4400),
        unit="GeV",
        x_title=r"$m({t}\overline{t})$",
    )
    config.add_variable(
        name="cos_theta_star",
        expression="TTbar.cos_theta_star",
        binning=(100, -1, 1),
        x_title=r"${cos}(\theta^{*})$",
    )
    config.add_variable(
        name="abs_cos_theta_star",
        expression="TTbar.abs_cos_theta_star",
        binning=(50, 0, 1),
        x_title=r"$|{cos}(\theta^{*})|$",
    )
    for decay in ("had", "lep"):
        for var, var_label, var_unit, var_binning in [
            ("mass", "M", "GeV", (100, 0, 700)),
            ("pt", "p_{T}", "GeV", (50, 0, 800)),
            ("eta", r"\eta", None, (50, -5, 5)),
            ("phi", r"\phi", None, (50, -np.pi, np.pi)),
        ]:
            config.add_variable(
                name=f"top_{decay}_{var}",
                expression=f"TTbar.top_{decay}_{var}",
                binning=var_binning,
                unit=var_unit,
                x_title=rf"${var_label}({{t}}_{{{decay}}})$",
            )
        config.add_variable(
            name=f"n_jet_{decay}",
            expression=f"TTbar.n_jet_{decay}",
            binning=(11, -0.5, 10.5),
            x_title=rf"$n_{{AK4 jets}}^{decay}$",
        )
    config.add_variable(
        name="n_jet_sum",
        expression="TTbar.n_jet_sum",
        binning=(11, -0.5, 10.5),
        x_title=r"$n_{AK4 jets}^{lep+had}$",
    )

    # gen variables
    config.add_variable(
        name="gen_ttbar_mass",
        expression="TTbar.gen_mass",
        binning=config.get_variable("ttbar_mass").binning,
        unit="GeV",
        x_title=r"$m({t}\overline{t})^{gen}$",
    )
    config.add_variable(
        name="gen_ttbar_mass_narrow",
        expression="TTbar.gen_mass",
        binning=config.get_variable("ttbar_mass_narrow").binning,
        unit="GeV",
        x_title=r"$m({t}\overline{t})^{gen}$",
    )
    config.add_variable(
        name="gen_cos_theta_star",
        expression="TTbar.gen_cos_theta_star",
        binning=config.get_variable("cos_theta_star").binning,
        x_title=r"${cos}(\theta^{*}_{gen})$",
    )
    config.add_variable(
        name="gen_abs_cos_theta_star",
        expression="TTbar.gen_abs_cos_theta_star",
        binning=config.get_variable("abs_cos_theta_star").binning,
        x_title=r"$|{cos}(\theta^{*})^{gen}|$",
    )
    for decay in ("had", "lep"):
        config.add_variable(
            name=f"gen_top_{decay}_delta_r",
            expression=f"TTbar.gen_top_{decay}_delta_r",
            binning=(50, 0, 0.4),
            x_title=rf"$\Delta R({{t}}_{{{decay}}}, {{t}}_{{{decay}}}^{{gen}})$",
        )
        config.add_variable(
            name=f"gen_top_{decay}_delta_r_wide",
            expression=f"TTbar.gen_top_{decay}_delta_r",
            binning=(100, 0, 3),
            x_title=rf"$\Delta R({{t}}_{{{decay}}}, {{t}}_{{{decay}}}^{{gen}})$",
        )

        for var, var_label, var_unit in [
            ("mass", "M", "GeV"),
            ("pt", "p_{T}", "GeV"),
            ("eta", r"\eta", None),
            ("phi", r"\phi", None),
        ]:
            config.add_variable(
                name=f"gen_top_{decay}_{var}",
                expression=f"TTbar.gen_top_{decay}_{var}",
                binning=config.get_variable(f"top_{decay}_{var}").binning,
                unit=var_unit,
                x_title=rf"${var_label}({{t}}_{{{decay}}}^{{gen}})$",
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
    config.add_variable(
        name="cf_lhe_ht",
        expression="cutflow.lhe_ht",
        binning=(100, 50., 3000.),
        unit="GeV",
        x_title=r"LHE $H_{T}$",
    )
    config.add_variable(
        name="cf_jet1_pt_more",
        expression="cutflow.jet1_pt",
        binning=(100, 0., 5000.),
        unit="GeV",
        x_title=r"jet 1 $p_{T}$",
    )


def add_variables_ml(config: od.Config) -> None:
    """
    Variables specific to machine learning (input variables, output scores, etc..
    """
    # namespace/field under which ML input features are stored
    ns = "MLInput"

    pt_binning_jets = (150 // 2, 0, 3000)
    pt_binning_leptons = (50 // 2, 0, 1000)
    pt_binning_met = (150 // 2, 0, 1500)
    energy_binning_jets = (100 // 2, 0, 5000)
    energy_binning_leptons = (144 // 2, 0, 3000)
    mass_binning_jets = (50 // 2, 0, 300)
    eta_binning = (50 // 2, -2.5, 2.5)
    phi_binning = (30 // 2, -(np.pi + 0.2), (np.pi + 0.2))
    btag_binning = (50 // 2, 0, 1)
    msoftdrop_binning = (50 // 2, 0, 500)
    tau_binning = (24 // 2, 0, 1.2)

    variables = {
        "pt": [(100, 0, 3000), "$p_{T}$", "GeV"],
        "energy": [(100, 0, 5000), "$E$", "GeV"],
        "mass": [(50, 0, 300), "$m$", "GeV"],
        "eta": [(50, -2.5, 2.5), r"$\eta$", None],
        "phi": [(30, -np.pi, np.pi), r"$\phi$", None],
        "btag": [(50, 0, 1), "b-tag score", None],
        "msoftdrop": [(50, 0, 500), "$m_{SD}$", "GeV"],
        "tau21": [(30, 0, 1.2), r"\tau_{21}", None],
        "tau32": [(30, 0, 1.2), r"\tau_{32}", None],
    }
    objects = {
        "jet": "AK4 jet",
        "fatjet": "AK8 jet",
        "lepton": "Lepton",
        "met": "Missing transverse energy",
    }

    config.add_variable(
        name="mli_n_jet",
        expression=f"{ns}.n_jet",
        binning=(20, -0.5, 19.5),
        x_title=r"ML input (# of AK4 jets)",
    )

    config.add_variable(
        name="mli_n_fatjet",
        expression=f"{ns}.n_fatjet",
        binning=(20, -0.5, 19.5),
        x_title=r"ML input (# of AK8 jets)",
    )

    # binning adjusted variables to better compare with AN

    config.add_variable(
        name="mli_AN_lepton_energy",
        expression=f"{ns}.lepton_energy",
        binning=energy_binning_leptons,
        unit="GeV",
        x_title=r"ML input (Lepton $E$)",
    )

    config.add_variable(
        name="mli_AN_lepton_pt",
        expression=f"{ns}.lepton_pt",
        binning=pt_binning_leptons,
        unit="GeV",
        x_title=r"ML input (Lepton $p_{T}$)",
    )

    config.add_variable(
        name="mli_AN_lepton_eta",
        expression=f"{ns}.lepton_eta",
        binning=eta_binning,
        x_title=r"ML input (Lepton $\eta$)",
    )

    config.add_variable(
        name="mli_AN_lepton_phi",
        expression=f"{ns}.lepton_phi",
        binning=phi_binning,
        x_title=r"ML input (Lepton $\phi$)",
    )

    config.add_variable(
        name="mli_AN_met_pt",
        expression=f"{ns}.met_pt",
        binning=pt_binning_met,
        unit="GeV",
        x_title="ML input (Missing transverse $p_{T}$)",
    )

    config.add_variable(
        name="mli_AN_met_phi",
        expression=f"{ns}.met_phi",
        binning=phi_binning,
        x_title=r"ML input (MET $\phi$)",
    )

    for i in range(1, 6):
        config.add_variable(
            name=f"mli_AN_jet_energy_{i}",
            expression=f"{ns}.jet_energy_{i}",
            binning=energy_binning_jets,
            unit="GeV",
            x_title=f"ML input (AK4 jet #{i} $E$)",
        )
        config.add_variable(
            name=f"mli_AN_jet_pt_{i}",
            expression=f"{ns}.jet_pt_{i}",
            binning=pt_binning_jets,
            unit="GeV",
            x_title=f"ML input (AK4 jet #{i} $p_T$)",
        )
        config.add_variable(
            name=f"mli_AN_jet_eta_{i}",
            expression=f"{ns}.jet_eta_{i}",
            binning=eta_binning,
            x_title=rf"ML input (AK4 jet #{i} $\eta$)",
        )
        config.add_variable(
            name=f"mli_AN_jet_phi_{i}",
            expression=f"{ns}.jet_phi_{i}",
            binning=phi_binning,
            x_title=rf"ML input (AK4 jet #{i} $\phi$)",
        )
        config.add_variable(
            name=f"mli_AN_jet_mass_{i}",
            expression=f"{ns}.jet_mass_{i}",
            binning=mass_binning_jets,
            unit="GeV",
            x_title=f"ML input (AK4 jet #{i} $m$)",
        )
        config.add_variable(
            name=f"mli_AN_jet_btag_{i}",
            expression=f"{ns}.jet_btag_{i}",
            binning=btag_binning,
            x_title=f"ML input (AK4 jet #{i} b tag score)",
        )

    for i in range(1, 4):
        config.add_variable(
            name=f"mli_AN_fatjet_energy_{i}",
            expression=f"{ns}.fatjet_energy_{i}",
            binning=energy_binning_jets,
            unit="GeV",
            x_title=f"ML input (AK8 jet #{i} $E$)",
        )
        config.add_variable(
            name=f"mli_AN_fatjet_pt_{i}",
            expression=f"{ns}.fatjet_pt_{i}",
            binning=pt_binning_jets,
            unit="GeV",
            x_title=f"ML input (AK8 jet #{i} $p_T$)",
        )
        config.add_variable(
            name=f"mli_AN_fatjet_eta_{i}",
            expression=f"{ns}.fatjet_eta_{i}",
            binning=eta_binning,
            x_title=rf"ML input (AK8 jet #{i} $\eta$)",
        )
        config.add_variable(
            name=f"mli_AN_fatjet_phi_{i}",
            expression=f"{ns}.fatjet_phi_{i}",
            binning=phi_binning,
            x_title=rf"ML input (AK8 jet #{i} $\phi$)",
        )
        config.add_variable(
            name=f"mli_AN_fatjet_msoftdrop_{i}",
            expression=f"{ns}.fatjet_msoftdrop_{i}",
            binning=msoftdrop_binning,
            unit="GeV",
            x_title=f"ML input (AK8 jet #{i} $m_{{SD}}$)",
        )
        config.add_variable(
            name=f"mli_AN_fatjet_tau21_{i}",
            expression=f"{ns}.fatjet_tau21_{i}",
            binning=tau_binning,
            x_title=rf"ML input (AK8 jet #{i} $\tau_{{21}}$)",
        )
        config.add_variable(
            name=f"mli_AN_fatjet_tau32_{i}",
            expression=f"{ns}.fatjet_tau32_{i}",
            binning=tau_binning,
            x_title=rf"ML input (AK8 jet #{i} $\tau_{{32}}$)",
        )

    # -- helper functions

    def add_vars(name, n_max, attrs):
        obj_label = objects.get(name, name)
        for i, attr in itertools.product(range(n_max), attrs):
            # get variable info
            binning, var_label, unit = variables[attr]

            # add variable to config
            config.add_variable(
                name=f"mli_{name}_{attr}_{i}",
                expression=f"{ns}.{name}_{attr}_{i}",
                binning=binning,
                unit=unit,
                x_title=f"ML input ({obj_label} #{i+1} {var_label})",
            )

    def add_vars_single(name, attrs):
        obj_label = objects.get(name, name)
        for attr in attrs:
            # get variable info
            binning, var_label, unit = variables[attr]

            # add variable to config
            config.add_variable(
                name=f"mli_{name}_{attr}",
                expression=f"{ns}.{name}_{attr}",
                binning=binning,
                unit=unit,
                x_title=f"ML input ({obj_label} {var_label})",
            )

    add_vars("jet", 5, ("energy", "pt", "eta", "phi", "mass", "btag"))
    add_vars("fatjet", 3, ("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32"))

    add_vars_single("lepton", ("energy", "pt", "eta", "phi"))
    add_vars_single("met", ("pt", "phi"))
