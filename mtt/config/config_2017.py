# coding: utf-8

"""
Configuration of the 2017 m(ttbar) analysis.
"""

import functools
import os

import yaml
from scinum import Number
import cmsdb
import cmsdb.campaigns.run2_2017_nano_v9

from columnflow.util import DotDict
from columnflow.config_util import (
    add_shift_aliases,
    get_root_processes_from_campaign,
    get_shifts_from_sources,
)
from mtt.config.categories import add_categories_selection
from mtt.config.variables import add_variables

from mtt.config.analysis_mtt import analysis_mtt

thisdir = os.path.dirname(os.path.abspath(__file__))

#
# 2017 standard config
#

# copy the campaign, which in turn copies datasets and processes
campaign_run2_2017 = cmsdb.campaigns.run2_2017_nano_v9.campaign_run2_2017_nano_v9.copy()

# get all root processes
procs = get_root_processes_from_campaign(campaign_run2_2017)

# create a config by passing the campaign, so id and name will be identical
config_2017 = analysis_mtt.add_config(campaign_run2_2017)

# add processes we are interested in
config_2017.add_process(procs.n.data)
config_2017.add_process(procs.n.tt)
config_2017.add_process(procs.n.st)
config_2017.add_process(procs.n.w_lnu)
config_2017.add_process(procs.n.dy)
config_2017.add_process(procs.n.qcd)
config_2017.add_process(procs.n.vv)
# ttbar signal processes
config_2017.add_process(procs.n.zprime_tt)
config_2017.add_process(procs.n.hscalar_tt)
config_2017.add_process(procs.n.hpseudo_tt)
config_2017.add_process(procs.n.rsgluon_tt)

# set `unstack` flag for signal processes (used when plotting)
for process, _, _ in config_2017.walk_processes():
    if any(
        process.name.startswith(prefix)
        for prefix in [
            "zprime_tt",
            "hpseudo_tt",
            "hscalar_tt",
            "rsgluon_tt",
        ]
    ):
        process.color1 = "#aaaaaa"
        process.color2 = "#000000"
        process.x.is_mtt_signal = True
        process.unstack = True
        process.hide_errors = True
    else:
        process.x.is_mtt_signal = False

# set color of some processes
colors = {
    "data": "#000000",  # black
    "tt": "#E04F21",  # red
    "qcd": "#5E8FFC",  # blue
    "w_lnu": "#82FF28",  # green
    "higgs": "#984ea3",  # purple
    "st": "#3E00FB",  # dark purple
    "dy": "#FBFF36",  # yellow
    "vv": "#B900FC",  # pink
    "other": "#999999",  # grey
    "zprime_m_500_w_???": "#000000",  # black
    "zprime_m_1000_w_???": "#CCCCCC",  # light gray
    "zprime_m_3000_w_???": "#666666",  # dark gray
}

for proc in config_2017.processes:
    config_2017.get_process(proc).color1 = colors.get(proc.name, "#aaaaaa")
    config_2017.get_process(proc).color2 = colors.get(proc.name, "#000000")

# add datasets we need to study
dataset_names = [
    # DATA
    "data_e_b",
    "data_e_c",
    "data_e_d",
    "data_e_e",
    "data_e_f",
    "data_mu_b",
    "data_mu_c",
    "data_mu_d",
    "data_mu_e",
    "data_mu_f",
    "data_pho_b",
    "data_pho_c",
    "data_pho_d",
    "data_pho_e",
    "data_pho_f",
    # TTbar
    "tt_sl_powheg",
    "tt_dl_powheg",
    "tt_fh_powheg",
    # WJets
    "w_lnu_ht70to100_madgraph",
    "w_lnu_ht100to200_madgraph",
    "w_lnu_ht200to400_madgraph",
    "w_lnu_ht400to600_madgraph",
    "w_lnu_ht600to800_madgraph",
    "w_lnu_ht800to1200_madgraph",
    "w_lnu_ht1200to2500_madgraph",
    "w_lnu_ht2500toinf_madgraph",
    # DY
    "dy_m50toinf_ht70to100_madgraph",
    "dy_m50toinf_ht100to200_madgraph",
    "dy_m50toinf_ht200to400_madgraph",
    "dy_m50toinf_ht400to600_madgraph",
    "dy_m50toinf_ht600to800_madgraph",
    "dy_m50toinf_ht800to1200_madgraph",
    "dy_m50toinf_ht1200to2500_madgraph",
    "dy_m50toinf_ht2500toinf_madgraph",
    # Diboson
    "ww_pythia",
    "wz_pythia",
    "zz_pythia",
    # SingleTop
    "st_schannel_lep_4f_amcatnlo",
    "st_schannel_had_4f_amcatnlo",
    "st_tchannel_t_4f_powheg",
    "st_tchannel_tbar_4f_powheg",
    "st_twchannel_t_powheg",
    "st_twchannel_tbar_powheg",
    # QCD
    "qcd_ht50to100_madgraph",
    "qcd_ht100to200_madgraph",
    "qcd_ht200to300_madgraph",
    "qcd_ht300to500_madgraph",
    "qcd_ht500to700_madgraph",
    "qcd_ht700to1000_madgraph",
    "qcd_ht1000to1500_madgraph",
    "qcd_ht1500to2000_madgraph",
    "qcd_ht2000toinf_madgraph",
    # -- signals
    # Z prime (width/mass = 10%)
    "zprime_tt_m400_w40_madgraph",
    "zprime_tt_m500_w50_madgraph",
    "zprime_tt_m600_w60_madgraph",
    "zprime_tt_m700_w70_madgraph",
    "zprime_tt_m800_w80_madgraph",
    "zprime_tt_m900_w90_madgraph",
    "zprime_tt_m1000_w100_madgraph",
    "zprime_tt_m1200_w120_madgraph",
    "zprime_tt_m1400_w140_madgraph",
    "zprime_tt_m1600_w160_madgraph",
    "zprime_tt_m1800_w180_madgraph",
    "zprime_tt_m2000_w200_madgraph",
    "zprime_tt_m2500_w250_madgraph",
    "zprime_tt_m3000_w300_madgraph",
    "zprime_tt_m3500_w350_madgraph",
    "zprime_tt_m4000_w400_madgraph",
    "zprime_tt_m4500_w450_madgraph",
    "zprime_tt_m5000_w500_madgraph",
    "zprime_tt_m6000_w600_madgraph",
    "zprime_tt_m7000_w700_madgraph",
    "zprime_tt_m8000_w800_madgraph",
    "zprime_tt_m9000_w900_madgraph",
    # Z prime (width/mass = 30%)
    "zprime_tt_m400_w120_madgraph",
    "zprime_tt_m500_w150_madgraph",
    "zprime_tt_m600_w180_madgraph",
    "zprime_tt_m700_w210_madgraph",
    "zprime_tt_m800_w240_madgraph",
    "zprime_tt_m900_w270_madgraph",
    "zprime_tt_m1000_w300_madgraph",
    "zprime_tt_m1200_w360_madgraph",
    "zprime_tt_m1400_w420_madgraph",
    "zprime_tt_m1600_w480_madgraph",
    "zprime_tt_m1800_w540_madgraph",
    "zprime_tt_m2000_w600_madgraph",
    "zprime_tt_m2500_w750_madgraph",
    "zprime_tt_m3000_w900_madgraph",
    "zprime_tt_m3500_w1050_madgraph",
    "zprime_tt_m4000_w1200_madgraph",
    "zprime_tt_m4500_w1350_madgraph",
    "zprime_tt_m5000_w1500_madgraph",
    "zprime_tt_m6000_w1800_madgraph",
    "zprime_tt_m7000_w2100_madgraph",
    "zprime_tt_m8000_w2400_madgraph",
    "zprime_tt_m9000_w2700_madgraph",
    # Z prime (width/mass = 1%)
    "zprime_tt_m400_w4_madgraph",
    "zprime_tt_m500_w5_madgraph",
    "zprime_tt_m600_w6_madgraph",
    "zprime_tt_m700_w7_madgraph",
    "zprime_tt_m800_w8_madgraph",
    "zprime_tt_m900_w9_madgraph",
    "zprime_tt_m1000_w10_madgraph",
    "zprime_tt_m1200_w12_madgraph",
    "zprime_tt_m1400_w14_madgraph",
    "zprime_tt_m1600_w16_madgraph",
    "zprime_tt_m1800_w18_madgraph",
    "zprime_tt_m2000_w20_madgraph",
    "zprime_tt_m2500_w25_madgraph",
    "zprime_tt_m3000_w30_madgraph",
    "zprime_tt_m3500_w35_madgraph",
    "zprime_tt_m4000_w40_madgraph",
    "zprime_tt_m4500_w45_madgraph",
    "zprime_tt_m5000_w50_madgraph",
    "zprime_tt_m6000_w60_madgraph",
    "zprime_tt_m7000_w70_madgraph",
    "zprime_tt_m8000_w80_madgraph",
    "zprime_tt_m9000_w90_madgraph",
    # pseudoscalar heavy higgs (width/mass = 25%)
    "hpseudo_tt_sl_m365_w91p25_res_madgraph",
    "hpseudo_tt_sl_m365_w91p25_int_madgraph",
    "hpseudo_tt_sl_m400_w100p0_res_madgraph",
    "hpseudo_tt_sl_m400_w100p0_int_madgraph",
    "hpseudo_tt_sl_m500_w125p0_res_madgraph",
    "hpseudo_tt_sl_m500_w125p0_int_madgraph",
    "hpseudo_tt_sl_m600_w150p0_res_madgraph",
    "hpseudo_tt_sl_m600_w150p0_int_madgraph",
    "hpseudo_tt_sl_m800_w200p0_res_madgraph",
    "hpseudo_tt_sl_m800_w200p0_int_madgraph",
    "hpseudo_tt_sl_m1000_w250p0_res_madgraph",
    "hpseudo_tt_sl_m1000_w250p0_int_madgraph",
    # pseudoscalar heavy higgs (width/mass = 10%)
    "hpseudo_tt_sl_m365_w36p5_res_madgraph",
    "hpseudo_tt_sl_m365_w36p5_int_madgraph",
    "hpseudo_tt_sl_m400_w40p0_res_madgraph",
    "hpseudo_tt_sl_m400_w40p0_int_madgraph",
    "hpseudo_tt_sl_m500_w50p0_res_madgraph",
    "hpseudo_tt_sl_m500_w50p0_int_madgraph",
    "hpseudo_tt_sl_m600_w60p0_res_madgraph",
    "hpseudo_tt_sl_m600_w60p0_int_madgraph",
    "hpseudo_tt_sl_m800_w80p0_res_madgraph",
    "hpseudo_tt_sl_m800_w80p0_int_madgraph",
    "hpseudo_tt_sl_m1000_w100p0_res_madgraph",
    "hpseudo_tt_sl_m1000_w100p0_int_madgraph",
    # pseudoscalar heavy higgs (width/mass = 2.5%)
    "hpseudo_tt_sl_m365_w9p125_res_madgraph",
    "hpseudo_tt_sl_m365_w9p125_int_madgraph",
    "hpseudo_tt_sl_m400_w10p0_res_madgraph",
    "hpseudo_tt_sl_m400_w10p0_int_madgraph",
    "hpseudo_tt_sl_m500_w12p5_res_madgraph",
    "hpseudo_tt_sl_m500_w12p5_int_madgraph",
    "hpseudo_tt_sl_m600_w15p0_res_madgraph",
    "hpseudo_tt_sl_m600_w15p0_int_madgraph",
    "hpseudo_tt_sl_m800_w20p0_res_madgraph",
    "hpseudo_tt_sl_m800_w20p0_int_madgraph",
    "hpseudo_tt_sl_m1000_w25p0_res_madgraph",
    "hpseudo_tt_sl_m1000_w25p0_int_madgraph",
    # scalar heavy higgs (width/mass = 25%)
    "hscalar_tt_sl_m365_w91p25_res_madgraph",
    "hscalar_tt_sl_m365_w91p25_int_madgraph",
    "hscalar_tt_sl_m400_w100p0_res_madgraph",
    "hscalar_tt_sl_m400_w100p0_int_madgraph",
    "hscalar_tt_sl_m500_w125p0_res_madgraph",
    "hscalar_tt_sl_m500_w125p0_int_madgraph",
    "hscalar_tt_sl_m600_w150p0_res_madgraph",
    "hscalar_tt_sl_m600_w150p0_int_madgraph",
    "hscalar_tt_sl_m800_w200p0_res_madgraph",
    "hscalar_tt_sl_m800_w200p0_int_madgraph",
    "hscalar_tt_sl_m1000_w250p0_res_madgraph",
    "hscalar_tt_sl_m1000_w250p0_int_madgraph",
    # scalar heavy higgs (width/mass = 10%)
    "hscalar_tt_sl_m365_w36p5_res_madgraph",
    "hscalar_tt_sl_m365_w36p5_int_madgraph",
    "hscalar_tt_sl_m400_w40p0_res_madgraph",
    "hscalar_tt_sl_m400_w40p0_int_madgraph",
    "hscalar_tt_sl_m500_w50p0_res_madgraph",
    "hscalar_tt_sl_m500_w50p0_int_madgraph",
    "hscalar_tt_sl_m600_w60p0_res_madgraph",
    "hscalar_tt_sl_m600_w60p0_int_madgraph",
    "hscalar_tt_sl_m800_w80p0_res_madgraph",
    "hscalar_tt_sl_m800_w80p0_int_madgraph",
    "hscalar_tt_sl_m1000_w100p0_res_madgraph",
    "hscalar_tt_sl_m1000_w100p0_int_madgraph",
    # scalar heavy higgs (width/mass = 2.5%)
    "hscalar_tt_sl_m365_w9p125_res_madgraph",
    "hscalar_tt_sl_m365_w9p125_int_madgraph",
    "hscalar_tt_sl_m400_w10p0_res_madgraph",
    "hscalar_tt_sl_m400_w10p0_int_madgraph",
    "hscalar_tt_sl_m500_w12p5_res_madgraph",
    "hscalar_tt_sl_m500_w12p5_int_madgraph",
    "hscalar_tt_sl_m600_w15p0_res_madgraph",
    "hscalar_tt_sl_m600_w15p0_int_madgraph",
    "hscalar_tt_sl_m800_w20p0_res_madgraph",
    "hscalar_tt_sl_m800_w20p0_int_madgraph",
    "hscalar_tt_sl_m1000_w25p0_res_madgraph",
    "hscalar_tt_sl_m1000_w25p0_int_madgraph",
    # Kaluza-Klein gluon
    "rsgluon_tt_m500_pythia",
    "rsgluon_tt_m1000_pythia",
    "rsgluon_tt_m1500_pythia",
    "rsgluon_tt_m2000_pythia",
    "rsgluon_tt_m2500_pythia",
    "rsgluon_tt_m3000_pythia",
    "rsgluon_tt_m3500_pythia",
    "rsgluon_tt_m4000_pythia",
    "rsgluon_tt_m4500_pythia",
    "rsgluon_tt_m5000_pythia",
    "rsgluon_tt_m5500_pythia",
    "rsgluon_tt_m6000_pythia",
]
for dataset_name in dataset_names:
    dataset = config_2017.add_dataset(campaign_run2_2017.get_dataset(dataset_name))

    # add tags to datasets:
    #     has_top: any dataset containing top quarks
    #     has_ttbar: any dataset containing a ttbar pair
    #     is_sm_ttbar: standard model ttbar datasets
    #     is_mtt_signal: m(ttbar) search signal datasets
    #     is_v_jets: W/Z+jets (including Drell-Yan)
    #     is_diboson: diboson datasets
    #     is_qcd: QCD multijet datasets
    #     is_*_data: various data-related tags

    # standard model ttbar
    if dataset.name.startswith("tt"):
        dataset.add_tag({"has_top", "has_ttbar", "is_sm_ttbar"})

    if dataset.name == "tt_sl_powheg":
        dataset.add_tag("has_memory_intensive_reco")

    # single top
    if dataset.name.startswith("st"):
        dataset.add_tag("has_top")

    # signal datasets
    if any(
        dataset.name.startswith(prefix)
        for prefix in ("zprime_tt", "hscalar_tt", "hpseudo_tt", "rsgluon_tt")
    ):
        dataset.add_tag({"has_top", "has_ttbar", "is_mtt_signal"})

    # W/Z+jets datasets
    if dataset.name.startswith("dy"):
        dataset.add_tag({"is_v_jets", "is_z_jets"})
    if dataset.name.startswith("w_lnu"):
        dataset.add_tag({"is_v_jets", "is_w_jets"})

    # diboson datasets
    if any(
        dataset.name.startswith(prefix)
        for prefix in [
            "ww",
            "wz",
            "zz",
        ]
    ):
        dataset.add_tag("is_diboson")

    # qcd datasets
    if dataset.name.startswith("qcd"):
        dataset.add_tag("is_qcd")

    # various data-related tags
    if dataset.name.startswith("data_mu"):
        dataset.add_tag("is_mu_data")
    if dataset.name.startswith("data_e"):
        dataset.add_tag({"is_e_data", "is_egamma_data"})
    if dataset.name.startswith("data_pho"):
        dataset.add_tag({"is_pho_data", "is_egamma_data"})
    if dataset.name.startswith("data_pho"):
        dataset.add_tag({"is_pho_data", "is_egamma_data"})

# trigger paths for muon/electron channels
config_2017.set_aux("triggers", DotDict.wrap({
    "lowpt": {
        "all": {
            "triggers": {
                "muon": {
                    "IsoMu27",
                },
                "electron": {
                    "Ele35_WPTight_Gsf",
                },
            },
        },
    },
    "highpt": {
        # split by run range
        # (some triggers inactive active for early runs)
        "early": {
            "triggers": {
                "muon": {
                    "Mu50",
                },
                "electron": {
                    "Ele35_WPTight_Gsf",
                },
                "photon": {
                    "Photon200",
                },
            },
            "run_range_max": 299329,
            "mc_trigger_percent": 11.58,
        },
        "late": {
            "triggers": {
                "muon": {
                    "Mu50",
                    "TkMu100",
                    "OldMu100",
                },
                "electron": {
                    "Ele115_CaloIdVT_GsfTrkIdT",
                },
                "photon": {
                    "Photon200",
                },
            },
        },
    },
}))
# ensure mc trigger fraction add up to 100%
config_2017.x.triggers.highpt.late.mc_trigger_percent = (
    100. - config_2017.x.triggers.highpt.early.mc_trigger_percent
)

# MET filters
# https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2?rev=158#2018_2017_data_and_MC_UL
config_2017.set_aux("met_filters", {
    "Flag.goodVertices",
    "Flag.globalSuperTightHalo2016Filter",
    "Flag.HBHENoiseFilter",
    "Flag.HBHENoiseIsoFilter",
    "Flag.EcalDeadCellTriggerPrimitiveFilter",
    "Flag.BadPFMuonFilter",
    "Flag.BadPFMuonDzFilter",
    "Flag.eeBadScFilter",
    "Flag.ecalBadCalibFilter",
})

# default calibrator, selector, producer, ml model and inference model
config_2017.set_aux("default_calibrator", "skip_jecunc")
config_2017.set_aux("default_selector", "default")
config_2017.set_aux("default_producer", "default")
config_2017.set_aux("default_ml_model", None)
config_2017.set_aux("default_inference_model", "simple")
config_2017.set_aux("default_categories", ["incl", "1e", "1m"])
config_2017.set_aux("default_process_settings", [
    ["zprime_tt_m400_w40", "unstack"],
])

# process groups for conveniently looping over certain processs
# (used in wrapper_factory and during plotting)
config_2017.set_aux("process_groups", {
    "default": ["zprime_tt_m400_w40", "tt", "st", "dy", "w_lnu", "qcd", "vv"],
    "signal": ["zprime_tt_m400_w40"],
    "bkg": ["tt", "st", "w_lnu", "dy", "qcd", "vv"],
})

# dataset groups for conveniently looping over certain datasets
# (used in wrapper_factory and during plotting)
config_2017.set_aux("dataset_groups", {
    "all": ["*"],
    "data": [
        "data_mu_*", "data_e_*", "data_pho_*",
    ],
    "default": [
        "zprime_tt_*", "hpseudo_tt_*", "hscalar_tt_*", "rsgluon_tt_*",
        "tt_*", "st_*", "dy_*", "w_lnu_*",
        "qcd_*", "ww_*", "wz_*", "zz_*",
    ],
    "tt": ["tt_*"], "st": ["st_*"], "w": ["w_lnu_*"], "dy": ["dy_*"],
    "qcd": ["qcd_*"],
    "vv": ["ww_*", "wz_*", "zz_*"],
    "zprime_tt": ["zprime_tt_*"],
    "hpseudo_tt": ["hpseudo_tt_*"],
    "hscalar_tt": ["hscalar_tt_*"],
    "rsgluon_tt": ["rsgluon_tt_*"],
    "backgrounds": [
        "tt_*", "st_*", "w_lnu_*", "dy_*",
        "qcd_*", "ww_*", "wz_*", "zz_*",
    ],
    "zprime_default": [
        "zprime_tt_m500_w50_madgraph",
        "zprime_tt_m1000_w100_madgraph",
        "zprime_tt_m3000_w300_madgraph",
    ],
    "dy": ["dy_*"],
    "w_lnu": ["w_lnu_*"],
})

# category groups for conveniently looping over certain categories
# (used during plotting)
config_2017.set_aux("category_groups", {
    "default": ["incl", "1e", "1m"],
    "all": [
        "1e",
        "1m",
        "1e__0t", "1e__1t",
        "1m__0t", "1m__1t",
        "1e__0t__chi2pass", "1e__1t__chi2pass",
        "1m__0t__chi2pass", "1m__1t__chi2pass",
        "1e__0t__chi2pass__acts_0_5", "1e__1t__chi2pass__acts_0_5",
        "1m__0t__chi2pass__acts_0_5", "1m__1t__chi2pass__acts_0_5",
        "1e__0t__chi2pass__acts_5_7", "1e__1t__chi2pass__acts_5_7",
        "1m__0t__chi2pass__acts_5_7", "1m__1t__chi2pass__acts_5_7",
        "1e__0t__chi2pass__acts_7_9", "1e__1t__chi2pass__acts_7_9",
        "1m__0t__chi2pass__acts_7_9", "1m__1t__chi2pass__acts_7_9",
        "1e__0t__chi2pass__acts_9_1", "1e__1t__chi2pass__acts_9_1",
        "1m__0t__chi2pass__acts_9_1", "1m__1t__chi2pass__acts_9_1",
        "1e__0t__chi2fail", "1e__1t__chi2fail",
        "1m__0t__chi2fail", "1m__1t__chi2fail",
        "1e__0t__chi2fail__acts_0_5", "1e__1t__chi2fail__acts_0_5",
        "1m__0t__chi2fail__acts_0_5", "1m__1t__chi2fail__acts_0_5",
        "1e__0t__chi2fail__acts_5_7", "1e__1t__chi2fail__acts_5_7",
        "1m__0t__chi2fail__acts_5_7", "1m__1t__chi2fail__acts_5_7",
        "1e__0t__chi2fail__acts_7_9", "1e__1t__chi2fail__acts_7_9",
        "1m__0t__chi2fail__acts_7_9", "1m__1t__chi2fail__acts_7_9",
        "1e__0t__chi2fail__acts_9_1", "1e__1t__chi2fail__acts_9_1",
        "1m__0t__chi2fail__acts_9_1", "1m__1t__chi2fail__acts_9_1",
    ],
    "all_chi2pass": [
        "1e",
        "1m",
        "1e__0t", "1e__1t",
        "1m__0t", "1m__1t",
        "1e__0t__chi2pass", "1e__1t__chi2pass",
        "1m__0t__chi2pass", "1m__1t__chi2pass",
        "1e__0t__chi2pass__acts_0_5", "1e__1t__chi2pass__acts_0_5",
        "1m__0t__chi2pass__acts_0_5", "1m__1t__chi2pass__acts_0_5",
        "1e__0t__chi2pass__acts_5_7", "1e__1t__chi2pass__acts_5_7",
        "1m__0t__chi2pass__acts_5_7", "1m__1t__chi2pass__acts_5_7",
        "1e__0t__chi2pass__acts_7_9", "1e__1t__chi2pass__acts_7_9",
        "1m__0t__chi2pass__acts_7_9", "1m__1t__chi2pass__acts_7_9",
        "1e__0t__chi2pass__acts_9_1", "1e__1t__chi2pass__acts_9_1",
        "1m__0t__chi2pass__acts_9_1", "1m__1t__chi2pass__acts_9_1",
    ],
})

# variable groups for conveniently looping over certain variables
# (used during plotting)
config_2017.set_aux("variable_groups", {
    "default": [
        "n_jet", "n_muon", "n_electron",
        "jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt",
        "fatjet1_pt", "fatjet2_pt", "fatjet3_pt", "fatjet4_pt",
        "muon_pt", "muon_eta",
        "electron_pt", "electron_eta",
    ],
    "cutflow": [
        "cf_n_jet", "cf_n_muon", "cf_n_electron",
        "cf_jet1_pt", "cf_jet2_pt", "cf_jet3_pt", "cf_jet4_pt",
        "cf_fatjet1_pt", "cf_fatjet2_pt", "cf_fatjet3_pt", "cf_fatjet4_pt",
        "cf_muon_pt", "cf_muon_eta",
        "cf_electron_pt", "cf_electron_eta",
    ],
    "new_version_test": [
        "n_jet", "n_electron", "n_muon",
        "met_pt", "met_phi",
        "electron_pt", "electron_phi",
        "muon_pt", "muon_phi",
        "jet1_pt", "jet1_phi",
        "fatjet1_pt", "fatjet1_phi",
        "chi2_lt100",
        "top_had_mass", "gen_top_had_mass",
        "top_lep_mass", "gen_lep_mass",
        "ttbar_mass_narrow", "gen_ttbar_mass_narrow",
        "cos_theta_star", "gen_cos_theta_star",
    ],
})

# shift groups for conveniently looping over certain shifts
# (used during plotting)
config_2017.set_aux("shift_groups", {
    "jer": ["nominal", "jer_up", "jer_down"],
})

# selector step groups for conveniently looping over certain steps
# (used in cutflow tasks)
config_2017.set_aux("selector_step_groups", {
    "default": ["Lepton", "MET", "Jet", "BJet", "JetLepton2DCut", "AllHadronicVeto", "DileptonVeto", "METFilters"],
})

config_2017.set_aux("selector_step_labels", {
    "JetLepton2DCut": "2D cut",
    "AllHadronicVeto": r"all-hadr. veto",
    "DileptonVeto": r"dilep. veto",
})


# process settings groups to quickly define settings for ProcessPlots
config_2017.set_aux("process_settings_groups", {
    "default": [
        ["zprime_tt_m400_w40", "scale=2000", "unstack"],
    ],
    "unstack_all": [
        [proc, "unstack"] for proc in config_2017.processes
    ],
})

zprime_base_label = r"Z'$\rightarrow$ $t\overline{t}$"
zprime_mass_labels = {
    "zprime_tt_m500_w50": "$m$ = 0.5 TeV",
    "zprime_tt_m1000_w100": "$m$ = 1 TeV",
    "zprime_tt_m3000_w300": "$m$ = 3 TeV",
}

for proc, zprime_mass_label in zprime_mass_labels.items():
    proc_inst = config_2017.get_process(proc)
    proc_inst.label = f"{zprime_base_label} ({zprime_mass_label})"

# 2017 luminosity with values in inverse pb and uncertainties taken from
# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM?rev=176#LumiComb
config_2017.set_aux("luminosity", Number(41480, {
    "lumi_13TeV_2017": 0.02j,
    "lumi_13TeV_1718": 0.006j,
    "lumi_13TeV_correlated": 0.009j,
}))

# 2017 minimum bias cross section in mb (milli) for creating PU weights, values from
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
# Note: not used by updated JSON based pileup weights producer
# config_2017.set_aux("minbias_xs", Number(69.2, 0.046j))

# 2017 b-tag working points
# https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
config_2017.x.btag_working_points = DotDict.wrap({
    "deepjet": {
        "loose": 0.0532,
        "medium": 0.3040,
        "tight": 0.7476,
    },
    "deepcsv": {
        "loose": 0.1355,
        "medium": 0.4506,
        "tight": 0.7738,
    },
})

# 2017 top-tagging working pointsi (DeepAK8, 1% mistagging rate, )
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/DeepAK8Tagging2018WPsSFs?rev=4
# TODO (?): unify with `toptag_sf_config`?
config_2017.x.toptag_working_points = DotDict.wrap({
    "deepak8": {
        # regular tagger
        "top": 0.344,
        "w": 0.739,
        # mass-decorrelated tagger
        "top_md": 0.725,
        "w_md": 0.925,
    },
})

# chi2 tuning parameters (mean masses/widths of top quarks
# with hadronically/leptonically decaying W bosons)
# AN2019_197_v3
config_2017.x.chi2_parameters = DotDict.wrap({
    "resolved": {
        "m_had": 175.4,  # GeV
        "s_had": 20.7,  # GeV
        "m_lep": 175.0,  # GeV
        "s_lep": 23.3,  # GeV
    },
    "boosted": {
        "m_had": 182.3,  # GeV
        "s_had": 16.1,  # GeV
        "m_lep": 172.2,  # GeV
        "s_lep": 21.7,  # GeV
    },
})

# parameters to fine-tune the ttbar combinatoric
# reconstruction
config_2017.set_aux("ttbar_reco_settings", DotDict.wrap({
    # -- minimal settings (fast runtime)
    # "n_jet_max": 9,
    # "n_jet_lep_range": (1, 1),
    # "n_jet_had_range": (3, 3),
    # "n_jet_ttbar_range": (4, 4),
    # "max_chunk_size": 100000,

    # -- default settings
    "n_jet_max": 9,
    "n_jet_lep_range": (1, 2),
    "n_jet_had_range": (1, 6),
    "n_jet_ttbar_range": (2, 6),
    "max_chunk_size": (
        lambda dataset_inst:
            10000 if dataset_inst.has_tag("has_memory_intensive_reco")
            else 30000
    ),

    # -- "maxed out" settings (very slow)
    # "n_jet_max": 10,
    # "n_jet_lep_range": (1, 8),
    # "n_jet_had_range": (1, 9),
    # "n_jet_ttbar_range": (2, 10),
    # "max_chunk_size": 10000,
}))

# L1 prefiring configuration
config_2017.x.l1_prefiring = DotDict.wrap({
    "jet": {
        "value": "l1_prefiring_efficiency_value_jetpt_2017BtoF",
        "error": "l1_prefiring_efficiency_error_jetpt_2017BtoF",
    },
    "photon": {
        "value": "l1_prefiring_efficiency_value_photonpt_2017BtoF",
        "error": "l1_prefiring_efficiency_error_photonpt_2017BtoF",
    },
})

# V+jets reweighting
config_2017.x.vjets_reweighting = DotDict.wrap({
    "w": {
        "value": "wjets_kfactor_value",
        "error": "wjets_kfactor_error",
    },
    "z": {
        "value": "zjets_kfactor_value",
        "error": "zjets_kfactor_error",
    },
})

# overwrite crosssections from cmsdb to adapt to UHH2 crosssections
# https://github.com/UHH2/UHH2-datasets/blob/master/CrossSectionHelper.py#L1804C22-L1804C35
dy_xsecs = {
    "dy_m50toinf_ht70to100": 140.1,
    "dy_m50toinf_ht100to200": 140.2,
    "dy_m50toinf_ht200to400": 38.399,
    "dy_m50toinf_ht400to600": 5.21278,
    "dy_m50toinf_ht600to800": 1.26567,
    "dy_m50toinf_ht800to1200": 0.5684304,
    "dy_m50toinf_ht1200to2500": 0.1331514,
    "dy_m50toinf_ht2500toinf": 0.00297803565,
}

for ds in dy_xsecs:
    procs.n(ds).set_xsec(13, dy_xsecs[ds])

w_lnu_xsecs = {
    "w_lnu_ht70to100": 1271,
    "w_lnu_ht100to200": 1253,
    "w_lnu_ht200to400": 335.9,
    "w_lnu_ht400to600": 45.21,
    "w_lnu_ht600to800": 10.99,
    "w_lnu_ht800to1200": 4.936,
    "w_lnu_ht1200to2500": 1.156,
    "w_lnu_ht2500toinf": 0.02623,
}

for ds in w_lnu_xsecs:
    procs.n(ds).set_xsec(13, w_lnu_xsecs[ds])

# cross sections for diboson samples; taken from:
# - ww (NNLO): https://arxiv.org/abs/1408.5243
# - wz (NLO): https://arxiv.org/abs/1105.0020
# - zz (NNLO): https://www.sciencedirect.com/science/article/pii/S0370269314004614?via%3Dihub
diboson_xsecs = {
    "ww": Number(118.7, {"scale": (0.025j, 0.022j)}),
    "wz": Number(46.74, {"scale": (0.041j, 0.033j)}),
    # "wz": Number(28.55, {"scale": (0.041j, 0.032j)}) + Number(18.19, {"scale": (0.041j, 0.033j)})  # (W+Z) + (W-Z)
    "zz": Number(16.99, {"scale": (0.032j, 0.024j)}),
}

for ds in diboson_xsecs:
    procs.n(ds).set_xsec(13, diboson_xsecs[ds])


# location of JEC txt files
config_2017.set_aux("jec", DotDict.wrap({
    "campaign": "Summer19UL17",
    "version": "V5",
    "jet_type": "AK4PFchs",
    "levels": ["L1L2L3Res"],
    "levels_for_type1_met": ["L1FastJet"],
    "data_eras": sorted(filter(None, {d.x("jec_era", None) for d in config_2017.datasets if d.is_data})),
    "uncertainty_sources": [
        # comment out most for now to prevent large file sizes
        # "AbsoluteStat",
        # "AbsoluteScale",
        # "AbsoluteSample",
        # "AbsoluteFlavMap",
        # "AbsoluteMPFBias",
        # "Fragmentation",
        # "SinglePionECAL",
        # "SinglePionHCAL",
        # "FlavorQCD",
        # "TimePtEta",
        # "RelativeJEREC1",
        # "RelativeJEREC2",
        # "RelativeJERHF",
        # "RelativePtBB",
        # "RelativePtEC1",
        # "RelativePtEC2",
        # "RelativePtHF",
        # "RelativeBal",
        # "RelativeSample",
        # "RelativeFSR",
        # "RelativeStatFSR",
        # "RelativeStatEC",
        # "RelativeStatHF",
        # "PileUpDataMC",
        # "PileUpPtRef",
        # "PileUpPtBB",
        # "PileUpPtEC1",
        # "PileUpPtEC2",
        # "PileUpPtHF",
        # "PileUpMuZero",
        # "PileUpEnvelope",
        # "SubTotalPileUp",
        # "SubTotalRelative",
        # "SubTotalPt",
        # "SubTotalScale",
        # "SubTotalAbsolute",
        # "SubTotalMC",
        "Total",
        # "TotalNoFlavor",
        # "TotalNoTime",
        # "TotalNoFlavorNoTime",
        # "FlavorZJet",
        # "FlavorPhotonJet",
        # "FlavorPureGluon",
        # "FlavorPureQuark",
        # "FlavorPureCharm",
        # "FlavorPureBottom",
        # "TimeRunA",
        # "TimeRunB",
        # "TimeRunC",
        # "TimeRunD",
        "CorrelationGroupMPFInSitu",
        "CorrelationGroupIntercalibration",
        "CorrelationGroupbJES",
        "CorrelationGroupFlavor",
        "CorrelationGroupUncorrelated",
    ],
}))

config_2017.set_aux("jer", DotDict.wrap({
    "campaign": "Summer19UL17",
    "version": "JRV2",
    "jet_type": "AK4PFchs",
}))


# -- electron scale factor names
config_2017.set_aux("electron_sf_names",
    ("UL-Electron-ID-SF", "2017", "wp80iso"),
)


# register shifts
config_2017.add_shift(name="nominal", id=0)
config_2017.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
config_2017.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})
config_2017.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
config_2017.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})
config_2017.add_shift(name="minbias_xs_up", id=7, type="shape")
config_2017.add_shift(name="minbias_xs_down", id=8, type="shape")
add_shift_aliases(config_2017, "minbias_xs", {"pu_weight": "pu_weight_{name}"})
config_2017.add_shift(name="top_pt_up", id=9, type="shape")
config_2017.add_shift(name="top_pt_down", id=10, type="shape")
add_shift_aliases(config_2017, "top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})

config_2017.add_shift(name="mur_up", id=101, type="shape")
config_2017.add_shift(name="mur_down", id=102, type="shape")
config_2017.add_shift(name="muf_up", id=103, type="shape")
config_2017.add_shift(name="muf_down", id=104, type="shape")
config_2017.add_shift(name="scale_up", id=105, type="shape")
config_2017.add_shift(name="scale_down", id=106, type="shape")
config_2017.add_shift(name="pdf_up", id=107, type="shape")
config_2017.add_shift(name="pdf_down", id=108, type="shape")
config_2017.add_shift(name="alpha_up", id=109, type="shape")
config_2017.add_shift(name="alpha_down", id=110, type="shape")

config_2017.add_shift(name="muon_up", id=111, type="shape")
config_2017.add_shift(name="muon_down", id=112, type="shape")
add_shift_aliases(config_2017, "muon", {"muon_weight": "muon_weight_{direction}"})

config_2017.add_shift(name="electron_up", id=113, type="shape")
config_2017.add_shift(name="electron_down", id=114, type="shape")
add_shift_aliases(config_2017, "electron", {"electron_weight": "electron_weight_{direction}"})

# V+jets reweighting
config_2017.add_shift(name="vjets_up", id=201, type="shape")
config_2017.add_shift(name="vjets_down", id=202, type="shape")
add_shift_aliases(config_2017, "vjets", {"vjets_weight": "vjets_weight_{direction}"})

# prefiring weights
config_2017.add_shift(name="l1_ecal_prefiring_up", id=301, type="shape")
config_2017.add_shift(name="l1_ecal_prefiring_down", id=302, type="shape")
add_shift_aliases(
    config_2017,
    "l1_ecal_prefiring",
    {"l1_ecal_prefiring_weight": "l1_ecal_prefiring_weight_{direction}"},
)

for unc in ["mur", "muf", "scale", "pdf", "alpha"]:
    add_shift_aliases(config_2017, unc, {f"{unc}_weight": unc + "_weight_{direction}"})

with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
    all_jec_sources = yaml.load(f, yaml.Loader)["names"]
for jec_source in config_2017.x.jec["uncertainty_sources"]:
    idx = all_jec_sources.index(jec_source)
    config_2017.add_shift(name=f"jec_{jec_source}_up", id=5000 + 2 * idx, type="shape")
    config_2017.add_shift(name=f"jec_{jec_source}_down", id=5001 + 2 * idx, type="shape")
    add_shift_aliases(
        config_2017,
        f"jec_{jec_source}",
        {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"},
    )

config_2017.add_shift(name="jer_up", id=6000, type="shape")
config_2017.add_shift(name="jer_down", id=6001, type="shape")
add_shift_aliases(
    config_2017,
    "jer",
    {"Jet.pt": "Jet.pt_{name}", "Jet.mass": "Jet.mass_{name}"},
)


def make_jme_filename(jme_aux, sample_type, name, era=None):
    """
    Convenience function to compute paths to JEC files.
    """
    # normalize and validate sample type
    sample_type = sample_type.upper()
    if sample_type not in ("DATA", "MC"):
        raise ValueError(f"invalid sample type '{sample_type}', expected either 'DATA' or 'MC'")

    jme_full_version = "_".join(s for s in (jme_aux.campaign, era, jme_aux.version, sample_type) if s)

    return f"{jme_aux.source}/{jme_full_version}/{jme_full_version}_{name}_{jme_aux.jet_type}.txt"


# external files
sources = {
    "cert": "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV",
    "local_repo": os.getenv("MTT_ORIG_BASE"),
    "json_mirror": "/afs/cern.ch/user/d/dsavoiu/public/mirrors/jsonpog-integration-a81953b1",
    "jet": "/afs/cern.ch/user/d/dsavoiu/public/mirrors/cms-jet-JSON_Format-54860a23",
}
config_2017.x.external_files = DotDict.wrap({
    # jet energy corrections
    "jet_jerc": (f"{sources['json_mirror']}/POG/JME/2017_UL/jet_jerc.json.gz", "v1"),  # noqa

    # top-tagging scale factors
    "toptag_sf": (f"{sources['jet']}/JMAR/DeepAK8/2017_DeepAK8_Top.json", "v1"),  # noqa

    # btag scale factors
    "btag_sf_corr": (f"{sources['json_mirror']}/POG/BTV/2017_UL/btagging.json.gz", "v1"),  # noqa

    # electron scale factors
    "electron_sf": (f"{sources['json_mirror']}/POG/EGM/2017_UL/electron.json.gz", "v1"),  # noqa

    # muon scale factors
    "muon_sf": (f"{sources['json_mirror']}/POG/MUO/2017_UL/muon_Z.json.gz", "v1"),  # noqa

    # L1 prefiring corrections
    "l1_prefiring": f"{os.getenv('MTT_ORIG_BASE')}/data/json/l1_prefiring.json",

    # V+jets reweighting
    "vjets_reweighting": f"{os.getenv('MTT_ORIG_BASE')}/data/json/vjets_reweighting.json",

    # lumi files
    "lumi": {
        "golden": (f"{sources['cert']}/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
        "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    },

    # files from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
    "pu": {
        "json": (f"{sources['cert']}/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
        "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
        "data_profile": {
            "nominal": (f"{sources['cert']}/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root", "v1"),  # noqa
            "minbias_xs_up": (f"{sources['cert']}/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root", "v1"),  # noqa
            "minbias_xs_down": (f"{sources['cert']}/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root", "v1"),  # noqa
        },
    },

    # pileup weight file
    "pu_sf": f"{sources['json_mirror']}/POG/LUM/2017_UL/puWeights.json.gz",
})

# columns to keep after certain steps
config_2017.x.keep_columns = DotDict.wrap({
    "cf.MergeSelectionMasks": {
        "mc_weight", "normalization_weight", "process_id", "category_ids", "cutflow.*",
    },
    "cf.ReduceEvents": {
        #
        # NanoAOD columns
        #

        # general event info
        "run", "luminosityBlock", "event",

        # weights
        "genWeight",
        "LHEWeight.*",
        "LHEPdfWeight", "LHEScaleWeight",

        # muons
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Muon.pfRelIso04_all",
        # electrons
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Electron.deltaEtaSC",
        "Electron.pfRelIso03_all",

        # photons (for L1 prefiring)
        "Photon.pt", "Photon.eta", "Photon.phi", "Photon.mass",
        "Photon.jetIdx",

        # -- AK4 jets

        # all
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Jet.rawFactor",
        "Jet.btagDeepFlavB", "Jet.hadronFlavour",

        # with b-tag
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        "BJet.btagDeepFlavB", "BJet.hadronFlavour",

        # without b-tag
        "LightJet.pt", "LightJet.eta", "LightJet.phi", "LightJet.mass",
        "LightJet.btagDeepFlavB", "LightJet.hadronFlavour",

        # -- AK8 jets
        # all
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass",
        "FatJet.rawFactor",
        "FatJet.msoftdrop", "FatJet.deepTagMD_TvsQCD",
        "FatJet.tau1", "FatJet.tau2", "FatJet.tau3",

        # with top tag
        "FatJetTopTag.pt", "FatJetTopTag.eta", "FatJetTopTag.phi", "FatJetTopTag.mass",
        "FatJetTopTag.rawFactor",
        "FatJetTopTag.msoftdrop", "FatJetTopTag.deepTagMD_TvsQCD",
        "FatJetTopTag.tau1", "FatJetTopTag.tau2", "FatJetTopTag.tau3",

        # with top tag and well-separated from lepton
        "FatJetTopTagDeltaRLepton.pt", "FatJetTopTagDeltaRLepton.eta",
        "FatJetTopTagDeltaRLepton.phi", "FatJetTopTagDeltaRLepton.mass",
        "FatJetTopTagDeltaRLepton.rawFactor",
        "FatJetTopTagDeltaRLepton.msoftdrop", "FatJetTopTagDeltaRLepton.deepTagDeltaRLeptonMD_TvsQCD",
        "FatJetTopTagDeltaRLepton.tau1", "FatJetTopTagDeltaRLepton.tau2", "FatJetTopTagDeltaRLepton.tau3",

        # generator quantities
        "Generator.*",

        # generator particles
        "GenPart.*",

        # generator objects
        "GenMET.*",
        "GenJet.*",
        "GenJetAK8.*",

        # missing transverse momentum
        "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",

        # number of primary vertices
        "PV.npvs",

        # average number of pileup interactions
        "Pileup.nTrueInt",

        #
        # columns added during selection
        #

        # generator particle info
        "GenTopDecay.*",
        "GenPartonTop.*",
        "GenVBoson.*",

        # generic leptons (merger of Muon/Electron)
        "Lepton.*",

        # columns for PlotCutflowVariables
        "cutflow.*",

        # other columns, required by various tasks
        "channel_id", "category_ids", "process_id",
        "deterministic_seed",
        "mc_weight",
        "pt_regime",
        "pu_weight*",
    },
})

# top pt reweighting parameters
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_dat?rev=31
config_2017.x.top_pt_reweighting_params = {
    "a": 0.0615,
    "b": -0.0005,
}

# event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
get_shifts = functools.partial(get_shifts_from_sources, config_2017)
config_2017.x.event_weights = DotDict({
    "normalization_weight": [],
    "pu_weight": get_shifts("minbias_xs"),
    "muon_weight": get_shifts("muon"),
    "electron_weight": get_shifts("electron"),
    # TODO: add scale and PDF weights, where available
    # "scale_weight": ???,
    # "pdf_weight": ???,
})

# event weights only present in certain datasets
for dataset in config_2017.datasets:
    dataset.x.event_weights = DotDict()
    # if dataset.has_tag("is_sm_ttbar"):
    #     # top pt reweighting (disable for now)
    #     dataset.x.event_weights["top_pt_weight"] = get_shifts("top_pt")
    if dataset.has_tag("is_v_jets"):
        # V+jets QCD NLO reweighting
        dataset.x.event_weights["vjets_weight"] = get_shifts("vjets")
    if not dataset.is_data:
        # prefiring weights (all datasets except real data)
        dataset.x.event_weights["l1_ecal_prefiring_weight"] = get_shifts("l1_ecal_prefiring")

# names of electron correction sets and working points
# (used in the electron_sf producer)
# TODO: check that these are appropriate
config_2017.x.electron_sf_names = ("UL-Electron-ID-SF", "2017", "wp80iso")

# names of muon correction sets and working points
# (used in the muon producer)
# TODO: check that these are appropriate
config_2017.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", "2017_UL")

# JEC uncertainty sources propagated to btag scale factors
# (names derived from contents in BTV correctionlib file)
year = config_2017.campaign.x.year
config_2017.x.btag_sf_jec_sources = [
    "",  # same as "Total"
    "Absolute",
    "AbsoluteMPFBias",
    "AbsoluteScale",
    "AbsoluteStat",
    f"Absolute_{year}",
    "BBEC1",
    f"BBEC1_{year}",
    "EC2",
    f"EC2_{year}",
    "FlavorQCD",
    "Fragmentation",
    "HF",
    f"HF_{year}",
    "PileUpDataMC",
    "PileUpPtBB",
    "PileUpPtEC1",
    "PileUpPtEC2",
    "PileUpPtHF",
    "PileUpPtRef",
    "RelativeBal",
    "RelativeFSR",
    "RelativeJEREC1",
    "RelativeJEREC2",
    "RelativeJERHF",
    "RelativePtBB",
    "RelativePtEC1",
    "RelativePtEC2",
    "RelativePtHF",
    "RelativeSample",
    f"RelativeSample_{year}",
    "RelativeStatEC",
    "RelativeStatFSR",
    "RelativeStatHF",
    "SinglePionECAL",
    "SinglePionHCAL",
    "TimePtEta",
]

# name of the btag_sf correction set and jec uncertainties to propagate through
config_2017.x.btag_sf = ("deepJet_shape", config_2017.x.btag_sf_jec_sources)

# name of the top tagging scale factors correction set and working point
# TODO (?): unify with `toptag_working_points`?
config_2017.x.toptag_sf_config = DotDict.wrap({
    "name": "DeepAK8_Top_MassDecorr",
    "wp": "1p0",
})

# versions per task family and optionally also dataset and shift
# None can be used as a key to define a default value
config_2017.set_aux("versions", {
})

config_2017.add_channel("e", id=1)
config_2017.add_channel("mu", id=2)

# working points for event categorization
config_2017.x.categorization = DotDict({
    "chi2_max": 30,
})

# add categories
add_categories_selection(config_2017)

# add variables
add_variables(config_2017)

# limited config with only 10 file per dataset
config_2017_limited = config_2017.copy(name=f"{config_2017.name}_limited", id="+")
for dataset in config_2017_limited.datasets:
    for k in dataset.info.keys():
        if dataset[k].n_files > 10:
            dataset[k].n_files = 10
