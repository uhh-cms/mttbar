# coding: utf-8

"""
Configuration for the Run 3 m(ttbar) trigger study analysis.
"""

from __future__ import annotations

import functools
import os

from scinum import Number
import order as od

from columnflow.util import DotDict
from columnflow.config_util import (
    add_shift_aliases,
    get_root_processes_from_campaign,
    get_shifts_from_sources,
    verify_config_processes,
)

from tgs.config.categories import add_categories
from tgs.config.variables import add_variables


thisdir = os.path.dirname(os.path.abspath(__file__))


def add_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
) -> od.Config:
    """
    Configurable function for creating a config for the
    run3 trigger study analysis given a base *analysis* object
    and a *campaign* (i.e. set of datasets).
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
    cfg.x.run = 3
    cfg.x.cpn_tag = f"{year}{corr_postfix}"

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # add processes we are interested in
    process_names = [
        "data",
        "tt",
        "st",
    ]
    for process_name in process_names:
        # add the process
        proc = cfg.add_process(procs.get(process_name))

    # set color of some processes
    colors = {
        "data": "#000000",  # black
        "tt": "#E04F21",  # red
        "st": "#3E00FB",  # dark purple
        "other": "#999999",  # grey
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
        # TTbar (only dilepton for now)
        # "tt_sl_powheg",
        "tt_dl_powheg",
        # "tt_fh_powheg",
    ]

    # DATA: only muon-triggered primary datasets
    if campaign.x.EE == "pre":
        dataset_names.extend([
            "data_mu_c",
            "data_mu_d",
        ])
    if campaign.x.EE == "post":
        dataset_names.extend([
            "data_mu_e",
            "data_mu_f",
            "data_mu_g",
        ])

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
        #     is_*_data: various data-related tags

        # standard model ttbar
        if dataset.name.startswith("tt"):
            dataset.add_tag({"has_top", "has_ttbar", "is_sm_ttbar"})

        if dataset.name == "tt_sl_powheg":
            dataset.add_tag("has_memory_intensive_reco")

        # single top
        if dataset.name.startswith("st"):
            dataset.add_tag("has_top")

        # various data-related tags
        if dataset.name.startswith("data_mu"):
            dataset.add_tag("is_mu_data")

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
                "min_value": 4,  # 1 = PFIsoVeryLoose, 2 = PFIsoLoose, 3 = PFIsoMedium, 4 = PFIsoTight, 5 = PFIsoVeryTight, 6 = PFIsoVeryVeryTight  # noqa
            },
            "id": {
                "low_pt": {
                    "column": "tightId",
                    "value": True,
                },
                "high_pt": {
                    "column": "highPtId",
                    "value": 2,  # 2 = global high pT, which includes tracker high pT
                },
            },
            "trigger": {
                "low_pt": "IsoMu27",
                "high_pt": "Mu50",
            },
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
        },
    })

    # jet selection parameters
    cfg.x.jet_selection = DotDict.wrap({
        "ak4": {
            "column": "Jet",
            # to be applied to all jets
            "all": {
                "max_abseta": 2.4,
                "min_pt": 25,
            },
            # to be applied on the first *n* pT-leading jets
            "leading": [
                # 1st jet
                {
                    "max_abseta": 2.4,
                    "min_pt": 50,
                },
                # 2nd jet
                {
                    "max_abseta": 2.4,
                    "min_pt": 50,
                },
            ],

            # b-tagging parameters (applied to 'all' jets above)
            "btagger": {
                "column": "btagDeepFlavB",
                "wp": cfg.x.btag_wp["deepjet"]["medium"],
            },
        },
    })

    # MET selection parameters
    cfg.x.met_selection = DotDict.wrap({
        "column": "MET",
        "min_pt": 70,
    })

    # lepton jet 2D isolation parameters
    cfg.x.lepton_jet_iso = DotDict.wrap({
        "min_pt": 15,
        "min_delta_r": 0.4,
        "min_pt_rel": 25,
    })

    # triggers paths to apply in the low
    # and high-pt muon regimes
    cfg.x.triggers = DotDict.wrap({
        "low_pt": "IsoMu27",
        "high_pt": "Mu50",
    })

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
    cfg.x.default_calibrator = "skip_jecunc"
    cfg.x.default_selector = "default_tgs"
    cfg.x.default_producer = "default_tgs"
    cfg.x.default_weight_producer = "all_weights"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = None
    cfg.x.default_categories = ["incl"]
    cfg.x.default_variables = ("electron_pt", "electron_eta")

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {}

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {}

    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {}

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    cfg.x.variable_groups = {}

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    cfg.x.shift_groups = {}

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    cfg.x.selector_step_groups = {
        "default": ["jet1", "jet2", "bjet", "muon", "electron", "muon_jet_2d_cut", "electron_jet_2d_cut", "MET"],
    }

    cfg.x.selector_step_labels = {
        "electron_jet_2d_cut": r"2D cut (ej)",
        "muon_jet_2d_cut": r"2D cut ($\mu$j)",
    }

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
        "Jet": {
            "campaign": jerc_campaign,
            "version": {2016: "V7", 2017: "V5", 2018: "V5", 2022: "V2"}[year],
            "jet_type": jet_type,
            "levels": ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"],
            "levels_for_type1_met": ["L1FastJet"],
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
                # "CorrelationGroupMPFInSitu",
                # "CorrelationGroupIntercalibration",
                # "CorrelationGroupbJES",
                # "CorrelationGroupFlavor",
                # "CorrelationGroupUncorrelated",
            ],
        },
    })

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    # TODO: get jerc working for Run3
    cfg.x.jer = DotDict.wrap({
        "Jet": {
            "campaign": jerc_campaign,
            "version": {2022: "JRV1"}[year],
            "jet_type": jet_type,
        },
    })

    # JEC uncertainty sources propagated to btag scale factors
    # (names derived from contents in BTV correctionlib file)
    cfg.x.btag_sf_jec_sources = [
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

    #
    # producer configurations
    #

    # TODO: check that everyting is setup as intended

    # name of the btag_sf correction set and jec uncertainties to propagate through
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

    # declare the shifts
    def add_shifts(cfg):
        # nominal shift
        cfg.add_shift(name="nominal", id=0)

    # add the shifts
    add_shifts(cfg)

    #
    # external files
    # taken from
    # https://github.com/uhh-cms/hh2bbww/blob/master/hbw/config/config_run2.py#L535C7-L579C84
    #

    sources = {
        "cert": "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22",
        "local_repo": "/data/dust/user/matthiej/mttbar",  # TODO: avoid hardcoding path
        "json_mirror": "/afs/cern.ch/user/j/jmatthie/public/mirrors/jsonpog-integration-49ddc547",
        # "jet": "/afs/cern.ch/user/d/dsavoiu/public/mirrors/cms-jet-JSON_Format-54860a23",
    }

    corr_tag = f"{year}_Summer22{jerc_postfix}"

    cfg.x.external_files = DotDict.wrap({
        # pileup weight corrections
        "pu_sf": (f"{sources['json_mirror']}/POG/LUM/{corr_tag}/puWeights.json.gz", "v1"),  # noqa

        # jet energy corrections
        "jet_jerc": (f"{sources['json_mirror']}/POG/JME/{corr_tag}/jet_jerc.json.gz", "v1"),  # noqa

        # top-tagging scale factors (TODO)
        # "toptag_sf": (f"{sources['jet']}/JMAR/???/???.json", "v1"),  # noqa

        # btag scale factors
        "btag_sf_corr": (f"{sources['json_mirror']}/POG/BTV/{corr_tag}/btagging.json.gz", "v1"),  # noqa

        # electron scale factors
        "electron_sf": (f"{sources['json_mirror']}/POG/EGM/{corr_tag}/electron.json.gz", "v1"),  # noqa

        # muon scale factors
        "muon_sf": (f"{sources['json_mirror']}/POG/MUO/{corr_tag}/muon_Z.json.gz", "v1"),  # noqa

        # met phi corrector
        "met_phi_corr": (f"{sources['json_mirror']}/POG/JME/{corr_tag}/met.json.gz", "v1"),

        # V+jets reweighting
        "vjets_reweighting": f"{sources['local_repo']}/data/json/vjets_reweighting.json",
    })

    # temporary fix due to missing corrections in run 3
    cfg.x.external_files.pop("met_phi_corr")

    if year == 2022 and campaign.x.EE == "pre":
        cfg.x.external_files.update(DotDict.wrap({
            # lumi files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
            "lumi": {
                "golden": (f"{sources['cert']}/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
                "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
            },

            # pileup files (for PU reweighting)
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

            # pileup files (for PU reweighting)
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
            "Muon.pfRelIso04_all", "Muon.highPtId",

            # electrons
            "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
            "Electron.deltaEtaSC",
            "Electron.pfRelIso03_all",
            "Electron.mvaFall17V2Iso_WP80",

            # -- AK4 jets

            # all
            "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
            "Jet.rawFactor",
            "Jet.btagDeepFlavB", "Jet.hadronFlavour",

            # -- AK8 jets
            # all
            "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass",
            "FatJet.rawFactor",
            "FatJet.msoftdrop", "FatJet.deepTagMD_TvsQCD",
            "FatJet.tau1", "FatJet.tau2", "FatJet.tau3",

            # generator quantities
            "Generator.*",

            # missing transverse momentum
            "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",

            # number of primary vertices
            "PV.npvs",

            # average number of pileup interactions
            "Pileup.nTrueInt",

            # trigger info (all possibly relevant triggers)
            "HLT.IsoMu24",
            "HLT.IsoMu27",
            "HLT.IsoTkMu24",
            "HLT.Mu50",
            "HLT.TkMu50",
            "HLT.TkMu100",
            "HLT.Ele27_WPTight_Gsf",
            "HLT.Ele35_WPTight_Gsf",
            # "HLT.Ele32_WPTight_Gsf",  # not present for part of run B
            "HLT.Ele115_CaloIdVT_GsfTrkIdT",
            "HLT.Photon175",
            "HLT.Photon200",

            #
            # columns added during selection
            #

            # columns for PlotCutflowVariables
            "cutflow.*",

            # other columns, required by various tasks
            "category_ids", "process_id",
            "deterministic_seed",
            "mc_weight",
            "pu_weight*",
        },
    })

    #
    # event weights
    #

    # event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
    get_shifts = functools.partial(get_shifts_from_sources, cfg)
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        # TODO: add systematics
        # "pu_weight": get_shifts("minbiax_xs"),
        # "muon_weight": get_shifts("muon"),
        # "ISR": get_shifts("ISR"),
        # "FSR": get_shifts("FSR"),
    })

    # # optional weights
    # if not cfg.has_tag("skip_electron_weights"):
    #     cfg.x.event_weights["electron_weight"] = get_shifts("electron")

    # event weights only present in certain datasets
    for dataset in cfg.datasets:
        dataset.x.event_weights = DotDict()

        # # TTbar: top pt reweighting
        # if dataset.has_tag("is_ttbar"):
        #     dataset.x.event_weights["top_pt_weight"] = get_shifts("top_pt")

        # V+jets: QCD NLO reweighting (disable for now)
        # if dataset.has_tag("is_v_jets"):
        #     dataset.x.event_weights["vjets_weight"] = get_shifts("vjets")

    # versions per task family and optionally also dataset and shift
    # None can be used as a key to define a default value
    cfg.x.versions = {}

    #
    # finalization
    #

    # add categories
    add_categories(cfg)

    # add variables
    add_variables(cfg)

    return cfg
