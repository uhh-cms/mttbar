# coding: utf-8

"""
Configuration of the 2022 m(ttbar) analysis.
"""

from __future__ import annotations

import functools
import os

import yaml
from scinum import Number

from columnflow.util import DotDict
from columnflow.config_util import (
    add_shift_aliases,
    get_root_processes_from_campaign,
    get_shifts_from_sources,
    verify_config_processes,
)
from mtt.config.categories import add_categories_selection
from mtt.config.variables import add_variables

import order as od


thisdir = os.path.dirname(os.path.abspath(__file__))


def add_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
) -> od.Config:
    """
    Configurable function for creating a config for a run3 analysis given
    a base *analysis* object and a *campaign* (i.e. set of datasets).
    """
    # validation
    assert campaign.x.year in [2022, 2023]
    if campaign.x.year == 2022:
        assert campaign.x.EE in ["pre", "post"]
    elif campaign.x.year == 2023:
        assert campaign.x.BPix in ["pre", "post"]

    # gather campaign data
    year = campaign.x.year
    year2 = year % 100
    corr_postfix = ""
    if year == 2022:
        corr_postfix = f"{campaign.x.EE}EE"
    elif year == 2023:
        corr_postfix = f"{campaign.x.BPix}BPix"

    implemented_years = [2022]

    if year not in implemented_years:
        raise NotImplementedError("For now, only 2022 campaign is fully implemented")

    # create a config by passing the campaign
    # (if id and name are not set they will be taken from the campaign)
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)

    # add some important tags to the config
    cfg.x.cpn_tag = f"{year}{corr_postfix}"

    cfg.x.run = 3

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # add processes we are interested in
    cfg.add_process(procs.n.data)
    cfg.add_process(procs.n.tt)
    cfg.add_process(procs.n.st)
    cfg.add_process(procs.n.w_lnu)
    cfg.add_process(procs.n.dy)
    cfg.add_process(procs.n.qcd)
    cfg.add_process(procs.n.vv)
    # # ttbar signal processes
    # cfg.add_process(procs.n.zprime_tt)
    # cfg.add_process(procs.n.hscalar_tt)
    # cfg.add_process(procs.n.hpseudo_tt)
    # cfg.add_process(procs.n.rsgluon_tt)

    # # set `unstack` flag for signal processes (used when plotting)
    # for process, _, _ in cfg.walk_processes():
    #     if any(
    #         process.name.startswith(prefix)
    #         for prefix in [
    #             "zprime_tt",
    #             "hpseudo_tt",
    #             "hscalar_tt",
    #             "rsgluon_tt",
    #         ]
    #     ):
    #         process.color1 = "#aaaaaa"
    #         process.color2 = "#000000"
    #         process.x.is_mtt_signal = True
    #         process.unstack = True
    #         process.hide_errors = True
    #     else:
    #         process.x.is_mtt_signal = False

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

    for proc in cfg.processes:
        cfg.get_process(proc).color1 = colors.get(proc.name, "#aaaaaa")
        cfg.get_process(proc).color2 = colors.get(proc.name, "#000000")

    #
    # datasets
    #

    # add datasets we need to study
    # errors taken over from top sf analysis, might work in this analysis
    dataset_names = [
        # TTbar
        "tt_sl_powheg",
        "tt_dl_powheg",
        "tt_fh_powheg",
        # SingleTop 2022 v12 preEE datasets
        "st_tchannel_t_4f_powheg",
        "st_tchannel_tbar_4f_powheg",
        "st_twchannel_t_sl_powheg",
        "st_twchannel_tbar_sl_powheg",
        "st_twchannel_t_dl_powheg",
        "st_twchannel_tbar_dl_powheg",
        "st_twchannel_t_fh_powheg",
        "st_twchannel_tbar_fh_powheg",
        # DY 2022 v12 preEE datasets
        # "dy_m4to50_ht40to70_madgraph",  # FIXME AssertionError in preEE (full stat.)
        # "dy_m4to50_ht70to100_madgraph",  # FIXME AssertionError in preEE (full stat.)
        "dy_m4to50_ht100to400_madgraph",
        "dy_m4to50_ht400to800_madgraph",
        "dy_m4to50_ht800to1500_madgraph",
        "dy_m4to50_ht1500to2500_madgraph",
        "dy_m4to50_ht2500toinf_madgraph",
        "dy_m50to120_ht40to70_madgraph",  # FIXME AssertionError in preEE (full stat.)
        "dy_m50to120_ht70to100_madgraph",
        "dy_m50to120_ht100to400_madgraph",
        "dy_m50to120_ht400to800_madgraph",
        # WJets 2022 v12 preEE datasets
        "w_lnu_mlnu0to120_ht40to100_madgraph",
        "w_lnu_mlnu0to120_ht100to400_madgraph",
        "w_lnu_mlnu0to120_ht400to800_madgraph",
        "w_lnu_mlnu0to120_ht800to1500_madgraph",
        # "w_lnu_mlnu0to120_ht1500to2500_madgraph",
        "w_lnu_mlnu0to120_ht2500toinf_madgraph",
        # Diboson
        "ww_pythia",
        "wz_pythia",
        "zz_pythia",
        # QCD 2022 v12 preEE datasets
        # "qcd_ht70to100_madgraph",  # FIXME AssertionError in preEE (full stat.)
        # "qcd_ht100to200_madgraph",  # FIXME no xs for 13.6 in https://xsdb-temp.app.cern.ch/xsdb/?columns=67108863&currentPage=0&pageSize=10&searchQuery=DAS%3DQCD-4Jets_HT-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8  # noqa
        # "qcd_ht200to400_madgraph",  # FIXME AssertionError in preEE (full stat.)
        # "qcd_ht400to600_madgraph",  # FIXME AssertionError in preEE (full stat.)
        # "qcd_ht600to800_madgraph",
        "qcd_ht800to1000_madgraph",
        "qcd_ht1000to1200_madgraph",
        "qcd_ht1200to1500_madgraph",
        "qcd_ht1500to2000_madgraph",
        "qcd_ht2000toinf_madgraph",
        # # -- signals TODO: not produced yet
        # # Z prime (width/mass = 10%)
        # "zprime_tt_m400_w40_madgraph",
        # "zprime_tt_m500_w50_madgraph",
        # "zprime_tt_m600_w60_madgraph",
        # "zprime_tt_m700_w70_madgraph",
        # "zprime_tt_m800_w80_madgraph",
        # "zprime_tt_m900_w90_madgraph",
        # "zprime_tt_m1000_w100_madgraph",
        # "zprime_tt_m1200_w120_madgraph",
        # "zprime_tt_m1400_w140_madgraph",
        # "zprime_tt_m1600_w160_madgraph",
        # "zprime_tt_m1800_w180_madgraph",
        # "zprime_tt_m2000_w200_madgraph",
        # "zprime_tt_m2500_w250_madgraph",
        # "zprime_tt_m3000_w300_madgraph",
        # "zprime_tt_m3500_w350_madgraph",
        # "zprime_tt_m4000_w400_madgraph",
        # "zprime_tt_m4500_w450_madgraph",
        # "zprime_tt_m5000_w500_madgraph",
        # "zprime_tt_m6000_w600_madgraph",
        # "zprime_tt_m7000_w700_madgraph",
        # "zprime_tt_m8000_w800_madgraph",
        # "zprime_tt_m9000_w900_madgraph",
        # # Z prime (width/mass = 30%)
        # "zprime_tt_m400_w120_madgraph",
        # "zprime_tt_m500_w150_madgraph",
        # "zprime_tt_m600_w180_madgraph",
        # "zprime_tt_m700_w210_madgraph",
        # "zprime_tt_m800_w240_madgraph",
        # "zprime_tt_m900_w270_madgraph",
        # "zprime_tt_m1000_w300_madgraph",
        # "zprime_tt_m1200_w360_madgraph",
        # "zprime_tt_m1400_w420_madgraph",
        # "zprime_tt_m1600_w480_madgraph",
        # "zprime_tt_m1800_w540_madgraph",
        # "zprime_tt_m2000_w600_madgraph",
        # "zprime_tt_m2500_w750_madgraph",
        # "zprime_tt_m3000_w900_madgraph",
        # "zprime_tt_m3500_w1050_madgraph",
        # "zprime_tt_m4000_w1200_madgraph",
        # "zprime_tt_m4500_w1350_madgraph",
        # "zprime_tt_m5000_w1500_madgraph",
        # "zprime_tt_m6000_w1800_madgraph",
        # "zprime_tt_m7000_w2100_madgraph",
        # "zprime_tt_m8000_w2400_madgraph",
        # "zprime_tt_m9000_w2700_madgraph",
        # # Z prime (width/mass = 1%)
        # "zprime_tt_m400_w4_madgraph",
        # "zprime_tt_m500_w5_madgraph",
        # "zprime_tt_m600_w6_madgraph",
        # "zprime_tt_m700_w7_madgraph",
        # "zprime_tt_m800_w8_madgraph",
        # "zprime_tt_m900_w9_madgraph",
        # "zprime_tt_m1000_w10_madgraph",
        # "zprime_tt_m1200_w12_madgraph",
        # "zprime_tt_m1400_w14_madgraph",
        # "zprime_tt_m1600_w16_madgraph",
        # "zprime_tt_m1800_w18_madgraph",
        # "zprime_tt_m2000_w20_madgraph",
        # "zprime_tt_m2500_w25_madgraph",
        # "zprime_tt_m3000_w30_madgraph",
        # "zprime_tt_m3500_w35_madgraph",
        # "zprime_tt_m4000_w40_madgraph",
        # "zprime_tt_m4500_w45_madgraph",
        # "zprime_tt_m5000_w50_madgraph",
        # "zprime_tt_m6000_w60_madgraph",
        # "zprime_tt_m7000_w70_madgraph",
        # "zprime_tt_m8000_w80_madgraph",
        # "zprime_tt_m9000_w90_madgraph",
        # # pseudoscalar heavy higgs (width/mass = 25%)
        # "hpseudo_tt_sl_m365_w91p25_res_madgraph",
        # "hpseudo_tt_sl_m365_w91p25_int_madgraph",
        # "hpseudo_tt_sl_m400_w100p0_res_madgraph",
        # "hpseudo_tt_sl_m400_w100p0_int_madgraph",
        # "hpseudo_tt_sl_m500_w125p0_res_madgraph",
        # "hpseudo_tt_sl_m500_w125p0_int_madgraph",
        # "hpseudo_tt_sl_m600_w150p0_res_madgraph",
        # "hpseudo_tt_sl_m600_w150p0_int_madgraph",
        # "hpseudo_tt_sl_m800_w200p0_res_madgraph",
        # "hpseudo_tt_sl_m800_w200p0_int_madgraph",
        # "hpseudo_tt_sl_m1000_w250p0_res_madgraph",
        # "hpseudo_tt_sl_m1000_w250p0_int_madgraph",
        # # pseudoscalar heavy higgs (width/mass = 10%)
        # "hpseudo_tt_sl_m365_w36p5_res_madgraph",
        # "hpseudo_tt_sl_m365_w36p5_int_madgraph",
        # "hpseudo_tt_sl_m400_w40p0_res_madgraph",
        # "hpseudo_tt_sl_m400_w40p0_int_madgraph",
        # "hpseudo_tt_sl_m500_w50p0_res_madgraph",
        # "hpseudo_tt_sl_m500_w50p0_int_madgraph",
        # "hpseudo_tt_sl_m600_w60p0_res_madgraph",
        # "hpseudo_tt_sl_m600_w60p0_int_madgraph",
        # "hpseudo_tt_sl_m800_w80p0_res_madgraph",
        # "hpseudo_tt_sl_m800_w80p0_int_madgraph",
        # "hpseudo_tt_sl_m1000_w100p0_res_madgraph",
        # "hpseudo_tt_sl_m1000_w100p0_int_madgraph",
        # # pseudoscalar heavy higgs (width/mass = 2.5%)
        # "hpseudo_tt_sl_m365_w9p125_res_madgraph",
        # "hpseudo_tt_sl_m365_w9p125_int_madgraph",
        # "hpseudo_tt_sl_m400_w10p0_res_madgraph",
        # "hpseudo_tt_sl_m400_w10p0_int_madgraph",
        # "hpseudo_tt_sl_m500_w12p5_res_madgraph",
        # "hpseudo_tt_sl_m500_w12p5_int_madgraph",
        # "hpseudo_tt_sl_m600_w15p0_res_madgraph",
        # "hpseudo_tt_sl_m600_w15p0_int_madgraph",
        # "hpseudo_tt_sl_m800_w20p0_res_madgraph",
        # "hpseudo_tt_sl_m800_w20p0_int_madgraph",
        # "hpseudo_tt_sl_m1000_w25p0_res_madgraph",
        # "hpseudo_tt_sl_m1000_w25p0_int_madgraph",
        # # scalar heavy higgs (width/mass = 25%)
        # "hscalar_tt_sl_m365_w91p25_res_madgraph",
        # "hscalar_tt_sl_m365_w91p25_int_madgraph",
        # "hscalar_tt_sl_m400_w100p0_res_madgraph",
        # "hscalar_tt_sl_m400_w100p0_int_madgraph",
        # "hscalar_tt_sl_m500_w125p0_res_madgraph",
        # "hscalar_tt_sl_m500_w125p0_int_madgraph",
        # "hscalar_tt_sl_m600_w150p0_res_madgraph",
        # "hscalar_tt_sl_m600_w150p0_int_madgraph",
        # "hscalar_tt_sl_m800_w200p0_res_madgraph",
        # "hscalar_tt_sl_m800_w200p0_int_madgraph",
        # "hscalar_tt_sl_m1000_w250p0_res_madgraph",
        # "hscalar_tt_sl_m1000_w250p0_int_madgraph",
        # # scalar heavy higgs (width/mass = 10%)
        # "hscalar_tt_sl_m365_w36p5_res_madgraph",
        # "hscalar_tt_sl_m365_w36p5_int_madgraph",
        # "hscalar_tt_sl_m400_w40p0_res_madgraph",
        # "hscalar_tt_sl_m400_w40p0_int_madgraph",
        # "hscalar_tt_sl_m500_w50p0_res_madgraph",
        # "hscalar_tt_sl_m500_w50p0_int_madgraph",
        # "hscalar_tt_sl_m600_w60p0_res_madgraph",
        # "hscalar_tt_sl_m600_w60p0_int_madgraph",
        # "hscalar_tt_sl_m800_w80p0_res_madgraph",
        # "hscalar_tt_sl_m800_w80p0_int_madgraph",
        # "hscalar_tt_sl_m1000_w100p0_res_madgraph",
        # "hscalar_tt_sl_m1000_w100p0_int_madgraph",
        # # scalar heavy higgs (width/mass = 2.5%)
        # "hscalar_tt_sl_m365_w9p125_res_madgraph",
        # "hscalar_tt_sl_m365_w9p125_int_madgraph",
        # "hscalar_tt_sl_m400_w10p0_res_madgraph",
        # "hscalar_tt_sl_m400_w10p0_int_madgraph",
        # "hscalar_tt_sl_m500_w12p5_res_madgraph",
        # "hscalar_tt_sl_m500_w12p5_int_madgraph",
        # "hscalar_tt_sl_m600_w15p0_res_madgraph",
        # "hscalar_tt_sl_m600_w15p0_int_madgraph",
        # "hscalar_tt_sl_m800_w20p0_res_madgraph",
        # "hscalar_tt_sl_m800_w20p0_int_madgraph",
        # "hscalar_tt_sl_m1000_w25p0_res_madgraph",
        # "hscalar_tt_sl_m1000_w25p0_int_madgraph",
        # # Kaluza-Klein gluon
        # "rsgluon_tt_m500_pythia",
        # "rsgluon_tt_m1000_pythia",
        # "rsgluon_tt_m1500_pythia",
        # "rsgluon_tt_m2000_pythia",
        # "rsgluon_tt_m2500_pythia",
        # "rsgluon_tt_m3000_pythia",
        # "rsgluon_tt_m3500_pythia",
        # "rsgluon_tt_m4000_pythia",
        # "rsgluon_tt_m4500_pythia",
        # "rsgluon_tt_m5000_pythia",
        # "rsgluon_tt_m5500_pythia",
        # "rsgluon_tt_m6000_pythia",
    ]
    if campaign.x.EE == "pre":
        dataset_names += [
            "data_egamma_c",
            "data_egamma_d",
            "data_mu_c",
            "data_mu_d",
        ]
    if campaign.x.EE == "post":
        dataset_names += [
            "data_egamma_e",
            "data_egamma_f",
            "data_egamma_g",
            "data_mu_e",
            "data_mu_f",
            "data_mu_g",
        ]

    for dataset_name in dataset_names:
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

        # update JECera information
        if dataset.is_data and (dataset_name.endswith("c") or dataset_name.endswith("d")):
            dataset.x.jec_era = "RunCD"

        # add tags to datasets:
        #     has_top: any dataset containing top quarks
        #     has_ttbar: any dataset containing a ttbar pair
        #     is_sm_ttbar: standard model ttbar datasets
        #     has_memory_intensive_reco: use different settings for ttbar reco
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
        if dataset.name.startswith("data_egamma"):
            dataset.add_tag("is_egamma_data")

        # for testing purposes, limit the number of files per dataset TODO make #files variable depending on dataset
        if limit_dataset_files:
            for info in dataset.info.values():
                info.n_files = min(info.n_files, limit_dataset_files)

    # verify that the root processes of each dataset (or one of their
    # ancestor processes) are registered in the config
    verify_config_processes(cfg, warn=True)

    #
    # tagger configuration
    # (b/top taggers)
    #
    tag_key = f"2022{campaign.x.EE}EE" if year == 2022 else year

    # b-tagging working points
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22/
    # https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/
    # TODO: add correct 2022 + 2022preEE WP for deepcsv if needed
    cfg.x.btag_wp = DotDict.wrap({
        "deepjet": {
            "loose": {
                "2022preEE": 0.0583, "2022postEE": 0.0614,
            }[tag_key],
            "medium": {
                "2022preEE": 0.3086, "2022postEE": 0.3196,
            }[tag_key],
            "tight": {
                "2022preEE": 0.7183, "2022postEE": 0.7300,
            }[tag_key],
        },
        "deepcsv": {
            "loose": {
                "2022preEE": 0.1208, "2022postEE": 0.1208,
            }[tag_key],
            "medium": {
                "2022preEE": 0.4168, "2022postEE": 0.4168,
            }[tag_key],
            "tight": {
                "2022preEE": 0.7665, "2022postEE": 0.7665,
            }[tag_key],
        },
    })

    # deepak8: 2017 top tagging working points (DeepAK8, 1% mistagging rate, )
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/DeepAK8Tagging2018WPsSFs?rev=4
    # particle_net: 2022 from JMAR presentation 21.10.24 (slide 10)
    # https://indico.cern.ch/event/1459087/contributions/6173396/attachments/2951723/5188840/SF_Run3.pdf
    cfg.x.toptag_wp = {
        "deepak8": {
            # regular tagger
            "top": 0.725,
            "w": 0.925,
            # mass-decorrelated tagger
            "top_md": 0.344,
            "w_md": 0.739,
        },
        "particle_net": {
            "medium": {
                "2022preEE": 0.683, "2022postEE": 0.698,
            }[tag_key],
            "tight": {
                "2022preEE": 0.858, "2022postEE": 0.866,
            }[tag_key],
            "very tight": {
                "2022preEE": 0.979, "2022postEE": 0.980,
            }[tag_key],
        },
    }

    #
    # selector configuration
    #

    # lepton selection parameters
    cfg.x.lepton_selection = DotDict.wrap({
        "mu": {
            "column": "Muon",
            "min_pt": {
                "low_pt": 30,
                "high_pt": 55,
            },
            "max_abseta": 2.4,
            "iso": {
                "column": "pfIsoId",
                "value": 4,
            },
            "id": {
                "low_pt": {
                    "column": "tightId",
                    "value": True,
                },
                "high_pt": {
                    "column": "highPtId",
                    "value": 2,
                },
            },
            # veto events with additional leptons passing looser cuts
            "min_pt_addveto": 25,
            "id_addveto": {
                "column": "tightId",
                "value": True,
            },
            "max_abseta_addveto": 2.4,
        },
        "e": {
            "column": "Electron",
            "min_pt": {
                "low_pt": 35,
                "high_pt": 120,
            },
            "max_abseta": 2.5,
            "barrel_veto": [1.44, 1.57],
            "mva_id": {
                "low_pt": "mvaIso_WP80",
                "high_pt": "mvaNoIso_WP80",
            },
            # veto events with additional leptons passing looser cuts
            "min_pt_addveto": 25,
            "id_addveto": {
                "column": "cutBased",
                "value": 3,
            },
            "max_abseta_addveto": 2.5,
        },
    })

    # jet selection parameters
    cfg.x.jet_selection = DotDict.wrap({
        "ak4": {
            "column": "Jet",
            "max_abseta": 2.5,
            "min_pt": {
                "baseline": 30,
                "e": [50, 40],
                "mu": [50, 50],
            },
            "btagger": {
                "column": "btagDeepFlavB",
                "wp": cfg.x.btag_wp["deepjet"]["medium"],
            },
        },
        "ak8": {
            "column": "FatJet",
            "max_abseta": 2.5,
            "min_pt": 400,
            "msoftdrop": [105, 210],
            "toptagger": {
                "column": "particleNetWithMass_TvsQCD",
                "wp": cfg.x.toptag_wp["particle_net"]["tight"],
            },
            "delta_r_lep": 0.8,
        },
    })

    # MET selection parameters
    cfg.x.met_selection = DotDict.wrap({
        "column": "MET",
        "min_pt": {
            "e": 60,
            "mu": 70,
        },
    })

    # lepton jet 2D isolation parameters
    cfg.x.lepton_jet_iso = DotDict.wrap({
        "min_pt": 15,
        "delta_r": 0.4,
        "pt_rel": 25,
    })

    # trigger paths for muon/electron channels
    # TODO update to relevant Run 3 triggers
    cfg.set_aux("triggers", DotDict.wrap({
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
            "all": {
                "triggers": {
                    "muon": {
                        "Mu50",
                        "HighPtTkMu100",
                        # "OldMu100",
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

    #
    # MET filters
    #

    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#Run_3_recommendations
    cfg.x.met_filters = {
        "Flag.goodVertices",
        "Flag.globalSuperTightHalo2016Filter",
        "Flag.EcalDeadCellTriggerPrimitiveFilter",
        "Flag.BadPFMuonFilter",
        "Flag.BadPFMuonDzFilter",
        "Flag.eeBadScFilter",
        "Flag.ecalBadCalibFilter",
    }

    # default calibrator, selector, producer, ml model and inference model
    cfg.set_aux("default_calibrator", "skip_jecunc")
    cfg.set_aux("default_selector", "default")
    cfg.set_aux("default_producer", "default")
    cfg.set_aux("default_weight_producer", "all_weights")
    cfg.set_aux("default_ml_model", None)
    cfg.set_aux("default_inference_model", "simple")
    cfg.set_aux("default_categories", ["incl", "1e", "1m"])
    cfg.set_aux("default_variables", ("n_jet",))
    cfg.set_aux("default_process_settings", [
        ["zprime_tt_m400_w40", "unstack"],
    ])

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {
        "default": ["zprime_tt_m400_w40", "tt", "st", "dy", "w_lnu", "qcd", "vv"],
        "signal": ["zprime_tt_m400_w40"],
        "bkg": ["tt", "st", "w_lnu", "dy", "qcd", "vv"],
    }

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {
        "all": ["*"],
        "data": [
            "data_mu_*", "data_egamma_*",
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
        "w_lnu": ["w_lnu_*"],
    }

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {
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
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    cfg.x.variable_groups = {
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
    }

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    cfg.x.shift_groups = {
        "jer": ["nominal", "jer_up", "jer_down"],
    }

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    cfg.x.selector_step_groups = {
        "default": ["Lepton", "MET", "Jet", "BJet", "JetLepton2DCut", "AllHadronicVeto", "DileptonVeto", "METFilters"],
    }

    cfg.x.selector_step_labels = {
        "JetLepton2DCut": "2D cut",
        "AllHadronicVeto": r"all-hadr. veto",
        "DileptonVeto": r"dilep. veto",
    }

    # # process settings groups to quickly define settings for ProcessPlots
    # cfg.x.process_settings_groups = {
    #     "default": [
    #         ["zprime_tt_m400_w40", "scale=2000", "unstack"],
    #     ],
    #     "unstack_all": [
    #         [proc, "unstack"] for proc in cfg.processes
    #     ],
    # }

    # zprime_base_label = r"Z'$\rightarrow$ $t\overline{t}$"
    # zprime_mass_labels = {
    #     "zprime_tt_m500_w50": "$m$ = 0.5 TeV",
    #     "zprime_tt_m1000_w100": "$m$ = 1 TeV",
    #     "zprime_tt_m3000_w300": "$m$ = 3 TeV",
    # }

    # for proc, zprime_mass_label in zprime_mass_labels.items():
    #     proc_inst = cfg.get_process(proc)
    #     proc_inst.label = f"{zprime_base_label} ({zprime_mass_label})"

    #
    # luminosity
    #

    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis
    if year == 2022:
        if campaign.x.EE == "pre":
            cfg.x.luminosity = Number(7971, {
                "lumi_13TeV_2022": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
        elif campaign.x.EE == "post":
            cfg.x.luminosity = Number(26337, {
                "lumi_13TeV_2022": 0.01j,
                "lumi_13TeV_correlated": 0.006j,
            })
    else:
        raise NotImplementedError(f"Luminosity for year {year} is not defined.")

    #
    # pileup
    #

    # # minimum bias cross section in mb (milli) for creating PU weights, values from
    # # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData?rev=45#Recommended_cross_section
    # not used after moving to correctionlib based PU weights
    # cfg.x.minbias_xs = Number(69.2, 0.046j)

    # chi2 tuning parameters (mean masses/widths of top quarks
    # with hadronically/leptonically decaying W bosons)
    # AN2019_197_v3
    cfg.x.chi2_parameters = DotDict.wrap({
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
    cfg.x.ttbar_reco_settings = DotDict.wrap({
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
    })

    # L1 prefiring configuration
    # FIXME not needed any more? Update to 2022 if needed
    cfg.x.l1_prefiring = DotDict.wrap({
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
    # FIXME update to Run 3 k-factors
    cfg.x.vjets_reweighting = DotDict.wrap({
        "w": {
            "value": "wjets_kfactor_value",
            "error": "wjets_kfactor_error",
        },
        "z": {
            "value": "zjets_kfactor_value",
            "error": "zjets_kfactor_error",
        },
    })

    #
    # cross sections
    #

    # cross sections for diboson samples; taken from:
    # - ww (NNLO): https://arxiv.org/abs/1408.5243
    # - wz (NLO): https://arxiv.org/abs/1105.0020
    # - zz (NNLO): https://www.sciencedirect.com/science/article/pii/S0370269314004614?via%3Dihub
    diboson_xsecs_13 = {
        "ww": Number(118.7, {"scale": (0.025j, 0.022j)}),
        "wz": Number(46.74, {"scale": (0.041j, 0.033j)}),
        # "wz": Number(28.55, {"scale": (0.041j, 0.032j)}) + Number(18.19, {"scale": (0.041j, 0.033j)}),  # (W+Z) + (W-Z)  # noqa
        "zz": Number(16.99, {"scale": (0.032j, 0.024j)}),
    }
    # TODO Use 14 TeV xs for Run 3?
    diboson_xsecs_14 = {
        "ww": Number(131.1, {"scale": (0.026j, 0.022j)}),
        "wz": Number(67.06, {"scale": (0.039j, 0.031j)}),
        # "wz": Number(31.50, {"scale": (0.039j, 0.030j)}) + Number(20.32, {"scale": (0.039j, 0.031j)}),  # (W+Z) + (W-Z)  # noqa
        "zz": Number(18.77, {"scale": (0.032j, 0.024j)}),
    }

    # linear interpolation between 13 and 14 TeV
    diboson_xsecs_13_6 = {
        ds: diboson_xsecs_13[ds] + (13.6 - 13.0) * (diboson_xsecs_14[ds] - diboson_xsecs_13[ds]) / (14.0 - 13.0)
        for ds in diboson_xsecs_13.keys()  # ww: 125.8 wz: 58.932 zz: 18.058  noqa
    }

    for ds in diboson_xsecs_14:
        procs.n(ds).set_xsec(13.6, diboson_xsecs_13_6[ds])

    #
    # JEC & JER  # FIXME: Taken from HBW
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L138C5-L269C1
    #

    # jec configuration
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=2017#Jet_Energy_Corrections_in_Run2

    # jec configuration taken from HBW
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L138C5-L269C1
    # https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC?rev=201
    jerc_postfix = ""
    if year == 2022 and campaign.x.EE == "post":
        jerc_postfix = "EE"

    jerc_campaign = f"Summer{year2}{jerc_postfix}_22Sep2023"
    jet_type = "AK4PFPuppi"

    cfg.x.jec = DotDict.wrap({
        "campaign": jerc_campaign,
        "version": {2016: "V7", 2017: "V5", 2018: "V5", 2022: "V2"}[year],
        "jet_type": jet_type,
        "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
        "levels_for_type1_met": ["L1FastJet"],
        "uncertainty_sources": [
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
            # "CorrelationGroupMPFInSitu",
            # "CorrelationGroupIntercalibration",
            # "CorrelationGroupbJES",
            # "CorrelationGroupFlavor",
            # "CorrelationGroupUncorrelated",
        ],
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    # TODO: get jerc working for Run3
    cfg.x.jer = DotDict.wrap({
        "campaign": jerc_campaign,
        "version": {2022: "JRV1"}[year],
        "jet_type": jet_type,
    })

    # JEC uncertainty sources propagated to btag scale factors
    # (names derived from contents in BTV correctionlib file)
    cfg.x.btag_sf_jec_sources = [
        "",  # total
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

    #
    # producer configurations
    #

    # TODO: check that everyting is setup as intended

    # btag weight configuration
    cfg.x.btag_sf = ("deepJet_shape", cfg.x.btag_sf_jec_sources)

    # name of the top tagging scale factors correction set and working point
    # TODO (?): unify with `toptag_working_points`?
    cfg.x.toptag_sf_config = DotDict.wrap({
        "name": "DeepAK8_Top_MassDecorr",
        "wp": "1p0",
    })

    # lepton sf taken from
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L338C1-L352C85
    # names of electron correction sets and working points
    # (used in the electron_sf producer)
    if cfg.x.cpn_tag == "2022postEE":
        # TODO: we need to use different SFs for control regions
        cfg.x.electron_sf_names = ("Electron-ID-SF", "2022Re-recoE+PromptFG", "Tight")
    elif cfg.x.cpn_tag == "2022preEE":
        cfg.x.electron_sf_names = ("Electron-ID-SF", "2022Re-recoBCD", "Tight")

    # names of muon correction sets and working points
    # (used in the muon producer)
    # TODO: we need to use different SFs for control regions
    cfg.x.muon_sf_names = ("NUM_TightPFIso_DEN_TightID", f"{cfg.x.cpn_tag}")
    cfg.x.muon_id_sf_names = ("NUM_TightID_DEN_TrackerMuons", f"{cfg.x.cpn_tag}")
    cfg.x.muon_iso_sf_names = ("NUM_TightPFIso_DEN_TightID", f"{cfg.x.cpn_tag}")

    #
    # systematic shifts
    #

    # read in JEC sources from file
    with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
        all_jec_sources = yaml.load(f, yaml.Loader)["names"]

    # declare the shifts
    def add_shifts(cfg):
        # nominal shift
        cfg.add_shift(name="nominal", id=0)

        # tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
        # (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
        cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
        cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

        cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
        cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})

        # pileup / minimum bias cross section variations
        cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
        cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
        add_shift_aliases(cfg, "minbias_xs", {"pu_weight": "pu_weight_{name}"})

        # top pt reweighting
        cfg.add_shift(name="top_pt_up", id=9, type="shape")
        cfg.add_shift(name="top_pt_down", id=10, type="shape")
        add_shift_aliases(cfg, "top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})

        # renormalization scale
        cfg.add_shift(name="mur_up", id=901, type="shape")
        cfg.add_shift(name="mur_down", id=902, type="shape")

        # factorization scale
        cfg.add_shift(name="muf_up", id=903, type="shape")
        cfg.add_shift(name="muf_down", id=904, type="shape")

        # scale variation (?)
        cfg.add_shift(name="scale_up", id=905, type="shape")
        cfg.add_shift(name="scale_down", id=906, type="shape")

        # pdf variations
        cfg.add_shift(name="pdf_up", id=951, type="shape")
        cfg.add_shift(name="pdf_down", id=952, type="shape")

        # alpha_s variation
        cfg.add_shift(name="alpha_up", id=961, type="shape")
        cfg.add_shift(name="alpha_down", id=962, type="shape")

        # TODO: murf_envelope?
        for unc in ["mur", "muf", "scale", "pdf", "alpha"]:
            add_shift_aliases(cfg, unc, {
                # TODO: normalized?
                f"{unc}_weight": f"{unc}_weight_{{direction}}",
            })

        # event weights due to muon scale factors
        if not cfg.has_tag("skip_muon_weights"):
            cfg.add_shift(name="muon_up", id=111, type="shape")
            cfg.add_shift(name="muon_down", id=112, type="shape")
            add_shift_aliases(cfg, "muon", {"muon_weight": "muon_weight_{direction}"})

        # event weights due to electron scale factors
        if not cfg.has_tag("skip_electron_weights"):
            cfg.add_shift(name="electron_up", id=121, type="shape")
            cfg.add_shift(name="electron_down", id=122, type="shape")
            add_shift_aliases(cfg, "electron", {"electron_weight": "electron_weight_{direction}"})

        # V+jets reweighting
        cfg.add_shift(name="vjets_up", id=201, type="shape")
        cfg.add_shift(name="vjets_down", id=202, type="shape")
        add_shift_aliases(cfg, "vjets", {"vjets_weight": "vjets_weight_{direction}"})

        # b-tagging shifts
        btag_uncs = [
            "hf", "lf",
            f"hfstats1_{year}", f"hfstats2_{year}",
            f"lfstats1_{year}", f"lfstats2_{year}",
            "cferr1", "cferr2",
        ]
        for i, unc in enumerate(btag_uncs):
            cfg.add_shift(name=f"btag_{unc}_up", id=501 + 2 * i, type="shape")
            cfg.add_shift(name=f"btag_{unc}_down", id=502 + 2 * i, type="shape")
            add_shift_aliases(
                cfg,
                f"btag_{unc}",
                {
                    # taken from
                    # https://github.com/uhh-cms/hh2bbww/blob/c6d4ee87a5c970660497e52aed6b7ebe71125d20/hbw/config/config_run2.py#L421
                    "normalized_btag_weight": f"normalized_btag_weight_{unc}_" + "{direction}",
                    "normalized_njet_btag_weight": f"normalized_njet_btag_weight_{unc}_" + "{direction}",
                    "btag_weight": f"btag_weight_{unc}_" + "{direction}",
                    "njet_btag_weight": f"njet_btag_weight_{unc}_" + "{direction}",
                },
            )

        # jet energy scale (JEC) uncertainty variations
        for jec_source in cfg.x.jec.uncertainty_sources:
            idx = all_jec_sources.index(jec_source)
            cfg.add_shift(name=f"jec_{jec_source}_up", id=5000 + 2 * idx, type="shape", tags={"jec"})
            cfg.add_shift(name=f"jec_{jec_source}_down", id=5001 + 2 * idx, type="shape", tags={"jec"})
            add_shift_aliases(
                cfg,
                f"jec_{jec_source}",
                {
                    "Jet.pt": "Jet.pt_{name}",
                    "Jet.mass": "Jet.mass_{name}",
                    "MET.pt": "MET.pt_{name}",
                },
            )

        # jet energy resolution (JER) scale factor variations
        cfg.add_shift(name="jer_up", id=6000, type="shape")
        cfg.add_shift(name="jer_down", id=6001, type="shape")
        add_shift_aliases(
            cfg,
            "jer",
            {
                "Jet.pt": "Jet.pt_{name}",
                "Jet.mass": "Jet.mass_{name}",
                "MET.pt": "MET.pt_{name}",
            },
        )

        # PSWeight variations
        cfg.add_shift(name="ISR_up", id=7001, type="shape")  # PS weight [0] ISR=2 FSR=1
        cfg.add_shift(name="ISR_down", id=7002, type="shape")  # PS weight [2] ISR=0.5 FSR=1
        add_shift_aliases(cfg, "ISR", {"ISR": "ISR_{direction}"})
        cfg.add_shift(name="FSR_up", id=7003, type="shape")  # PS weight [1] ISR=1 FSR=2
        cfg.add_shift(name="FSR_down", id=7004, type="shape")  # PS weight [3] ISR=1 FSR=0.5
        add_shift_aliases(cfg, "FSR", {"FSR": "FSR_{direction}"})

    # add the shifts
    add_shifts(cfg)

    #
    # external files
    # taken from
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L535C7-L579C84
    #

    # external files
    json_mirror = "/afs/cern.ch/user/j/jmatthie/public/mirrors/jsonpog-integration-49ddc547"
    local_repo = "/nfs/dust/cms/user/matthiej/mttbar"  # TODO: avoid hardcoding path

    corr_tag = f"{year}_Summer22{jerc_postfix}"

    cfg.x.external_files = DotDict.wrap({
        # pileup weight corrections
        "pu_sf": (f"{json_mirror}/POG/LUM/{corr_tag}/puWeights.json.gz", "v1"),

        # jet energy correction
        "jet_jerc": (f"{json_mirror}/POG/JME/{corr_tag}/jet_jerc.json.gz", "v1"),

        # electron scale factors
        "electron_sf": (f"{json_mirror}/POG/EGM/{corr_tag}/electron.json.gz", "v1"),

        # muon scale factors
        "muon_sf": (f"{json_mirror}/POG/MUO/{corr_tag}/muon_Z.json.gz", "v1"),

        # btag scale factor
        "btag_sf_corr": (f"{json_mirror}/POG/BTV/{corr_tag}/btagging.json.gz", "v1"),

        # met phi corrector
        "met_phi_corr": (f"{json_mirror}/POG/JME/{corr_tag}/met.json.gz", "v1"),

        # V+jets reweighting
        "vjets_reweighting": f"{local_repo}/data/json/vjets_reweighting.json",
    })

    # temporary fix due to missing corrections in run 3
    cfg.x.external_files.pop("met_phi_corr")

    if year == 2022 and campaign.x.EE == "pre":
        cfg.x.external_files.update(DotDict.wrap({
            # files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
            "lumi": {
                "golden": ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },
            "pu": {
                # "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileup_JSON.txt", "v1"),  # noqa
                "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCDEFG/pileup_JSON.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/bb525104a7ddb93685f8ced6fed1ab793b2d2103/SimGeneral/MixingModule/python/Run3_2022_LHC_Simulation_10h_2h_cfi.py", "v1"),  # noqa
                "data_profile": {
                    # "nominal": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-69200ub-99bins.root", "v1"),  # noqa
                    # "minbias_xs_up": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-72400ub-99bins.root", "v1"),  # noqa
                    # "minbias_xs_down": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-66000ub-99bins.root", "v1"),  # noqa
                    "nominal": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-69200ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_up": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-72400ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_down": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-66000ub-100bins.root", "v1"),  # noqa
                },
            },
        }))
    elif year == 2022 and campaign.x.EE == "post":
        cfg.x.external_files.update(DotDict.wrap({
            # files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
            "lumi": {
                "golden": ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },
            "pu": {
                # "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileup_JSON.txt", "v1"),  # noqa
                "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCDEFG/pileup_JSON.txt", "v1"),  # noqa
                "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/bb525104a7ddb93685f8ced6fed1ab793b2d2103/SimGeneral/MixingModule/python/Run3_2022_LHC_Simulation_10h_2h_cfi.py", "v1"),  # noqa
                "data_profile": {
                    # data profiles were produced with 99 bins instead of 100 --> use custom produced data profiles instead  # noqa
                    # "nominal": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-69200ub-99bins.root", "v1"),  # noqa
                    # "minbias_xs_up": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-72400ub-99bins.root", "v1"),  # noqa
                    # "minbias_xs_down": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-66000ub-99bins.root", "v1"),  # noqa
                    "nominal": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-69200ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_up": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-72400ub-100bins.root", "v1"),  # noqa
                    "minbias_xs_down": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-66000ub-100bins.root", "v1"),  # noqa
                },
            },
        }))
    else:
        raise NotImplementedError(f"No lumi and pu files provided for year {year}")

    # columns to keep after certain steps
    cfg.x.keep_columns = DotDict.wrap({
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
            "PSWeight",

            # muons
            "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
            "VetoMuon.pt", "VetoMuon.eta", "VetoMuon.phi", "VetoMuon.mass",
            "Muon.pfRelIso04_all",
            # electrons
            "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
            "VetoElectron.pt", "VetoElectron.eta", "VetoElectron.phi", "VetoElecton.mass",
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
    cfg.x.top_pt_reweighting_params = {
        "a": 0.0615,
        "b": -0.0005,
    }

    #
    # event weights
    #

    # event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
    get_shifts = functools.partial(get_shifts_from_sources, cfg)
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "pu_weight": get_shifts("minbias_xs"),
        "muon_weight": get_shifts("muon"),
        # "ISR": get_shifts("ISR"),
        # "FSR": get_shifts("FSR"),
    })

    if not cfg.has_tag("skip_electron_weights"):
        cfg.x.event_weights["electron_weight"] = get_shifts("electron")
    for dataset in cfg.datasets:
        dataset.x.event_weights = DotDict()
        if dataset.has_tag("is_ttbar"):
            # top pt reweighting
            dataset.x.event_weights["top_pt_weight"] = get_shifts("top_pt")
        # if dataset.has_tag("is_v_jets"):
        #     # V+jets QCD NLO reweighting
        #     dataset.x.event_weights["vjets_weight"] = get_shifts("vjets")

    # versions per task family and optionally also dataset and shift
    # None can be used as a key to define a default value
    cfg.set_aux("versions", {
    })

    #
    # finalization
    #

    # add channels
    cfg.add_channel("e", id=1)
    cfg.add_channel("mu", id=2)

    # working points for event categorization
    cfg.x.categorization = DotDict({
        "chi2_max": 30,
    })

    # add categories
    add_categories_selection(cfg)

    # add variables
    add_variables(cfg)

    return cfg
