# coding: utf-8

"""
Configuration for the Run 3 m(ttbar) analysis.
"""

from __future__ import annotations

import functools
import os

import yaml
from scinum import Number

from columnflow.util import DotDict
from columnflow.cms_util import CATInfo, CATSnapshot
from columnflow.config_util import (
    add_shift_aliases,
    get_root_processes_from_campaign,
    get_shifts_from_sources,
    verify_config_processes,
)
from mtt.config.categories import add_categories_selection
from mtt.config.variables import add_variables

from mtt.config.datasets import (
    data_datasets,
    dy_datasets,
    w_lnu_datasets,
    qcd_datasets,
    tt_datasets,
    st_datasets,
    vv_datasets,
    # Run 3 signal samples
    zprime_datasets,
    # hscalar_datasets,
    # hpseudo_datasets,
    # rsgluon_datasets,
)
from mtt.config.taggers import btag_params, toptag_params
from mtt.config.defaults_and_groups import (
    set_defaults,
    set_process_groups,
    set_dataset_groups,
    set_category_groups,
    set_variables_groups,
    set_shift_groups,
    set_selector_steps,
)
from mtt.config.corrections import (
    vjets_reweighting_cfg,
    jerc_cfg,
    btag_sf_cfg,
    toptag_sf_cfg,
    lepton_sf_cfg,
    met_phi_cfg,
    jet_id_cfg,
)

import order as od


thisdir = os.path.dirname(os.path.abspath(__file__))


def add_new_config(
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

    # validation (TODO: why?)
    assert campaign.x.year in [2022, 2023, 2024]
    if campaign.x.year == 2022:
        assert campaign.x.EE in ["pre", "post"]
    elif campaign.x.year == 2023:
        assert campaign.x.BPix in ["pre", "post"]

    # campaign data
    year = campaign.x.year
    # year2 = year % 100
    vnano = campaign.x.version
    corr_postfix = ""
    if year == 2022:
        corr_postfix = f"{campaign.x.EE}EE"
    elif year == 2023:
        corr_postfix = f"{campaign.x.BPix}BPix"

    implemented_years = [2022, 2023, 2024]
    if year not in implemented_years:
        raise NotImplementedError(f"Only {', '.join(map(str, implemented_years))} campaigns are implemented.")

    # create a config by passing the campaign
    # (if id and name are not set they will be taken from the campaign)
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)

    # add tags to config
    cfg.x.run = 3
    cfg.x.cpn_tag = f"{year}{corr_postfix}"
    cfg.x.year = year

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # add processes and datasets we are interested in
    cfg.add_process(procs.n.data)
    data_datasets(cfg, limit_dataset_files, log=False)

    cfg.add_process(procs.n.tt)
    tt_datasets(cfg, limit_dataset_files, log=False)

    cfg.add_process(procs.n.st)
    st_datasets(cfg, limit_dataset_files, log=False)

    cfg.add_process(procs.n.vv)
    vv_datasets(cfg, limit_dataset_files, log=False)

    # # ttbar signal processes
    if year in [2023, 2024]:
        cfg.add_process(procs.n.zprime_tt)
        zprime_datasets(cfg, limit_dataset_files, log=False)
        process_insts = [
            process_inst
            for process_inst, _, _ in cfg.walk_processes()
            if process_inst.name.startswith("zprime_tt")
        ]
        for process_inst in process_insts:
            if not process_inst.xsecs.get(13.6, None):
                # print(f"Warning: cross section for process {process_inst.name} at 13.6 TeV is not set.")
                # print("Setting it to 0.1 pb.")
                process_inst.xsecs[13.6] = Number(0.1)

    if year == 2024:
        cfg.add_process(procs.n.dy)
        dy_datasets(cfg, limit_dataset_files, log=False)

        cfg.add_process(procs.n.qcd)
        qcd_datasets(cfg, limit_dataset_files, log=False)

        cfg.add_process(procs.n.w_lnu)
        w_lnu_datasets(cfg, limit_dataset_files, log=False)

        cfg.add_process(procs.n.w_lnu_1j)

        cfg.add_process(procs.n.w_lnu_2j)

        cfg.add_process(procs.n.w_lnu_3j)

        cfg.add_process(procs.n.w_lnu_4j)

    # cfg.add_process(procs.n.hscalar_tt)
    # hscalar_datasets(cfg, log=True)

    # cfg.add_process(procs.n.hpseudo_tt)
    # hpseudo_datasets(cfg, log=True)

    # cfg.add_process(procs.n.rsgluon_tt)
    # rsgluon_datasets(cfg, log=True)

    # set flags for signal processes (used when plotting)
    for process, _, _ in cfg.walk_processes():
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

    # set color of main processes
    colors = {
        "data": "#000000",  # black
        "tt": "#E04F21",  # red
        "qcd": "#5E8FFC",  # blue
        "w_lnu": "#82FF28",  # green
        "w_lnu_1j": "#006400",  # dark green
        "w_lnu_2j": "#98FB98",  # light green
        "w_lnu_3j": "#00FF7F",  # spring green
        "w_lnu_4j": "#7CFC00",  # lawn green
        "higgs": "#984ea3",  # purple
        "st": "#3E00FB",  # dark purple
        "dy": "#FBFF36",  # yellow
        "vv": "#B900FC",  # pink
        "other": "#999999",  # grey
        "zprime_m500_w5": "#000000",  # black
        "zprime_m1000_w???": "#CCCCCC",  # light gray
        "zprime_m3000_w???": "#666666",  # dark gray
    }

    # process settings groups to quickly define settings for ProcessPlots
    if year == 2024:
        cfg.x.process_settings_groups = {
            "default": [
                ["zprime_tt_m400_w40", "scale=2000", "unstack"],
            ],
            "unstack_all": [
                [proc, "unstack"] for proc in cfg.processes
            ],
        }

        # zprime_base_label = r"Z'$\rightarrow$ $t\overline{t}$"
        # zprime_mass_labels = {
        #     # "zprime_tt_m500_w50": "$m$ = 0.5 TeV",
        #     # "zprime_tt_m1000_w100": "$m$ = 1 TeV",
        #     # "zprime_tt_m3000_w300": "$m$ = 3 TeV",
        #     "zprime_tt_m7000_w70": "$m$ = 7 TeV",
        # }

        # for proc, zprime_mass_label in zprime_mass_labels.items():
        #     proc_inst = cfg.get_process(proc)
        #     proc_inst.label = f"{zprime_base_label} ({zprime_mass_label})"

        for proc in cfg.processes:
            cfg.get_process(proc).color1 = colors.get(proc.name, "#aaaaaa")
            cfg.get_process(proc).color2 = colors.get(proc.name, "#000000")

    # verify that the root processes of each dataset (or one of their
    # ancestor processes) are registered in the config
    verify_config_processes(cfg, warn=True)

    # add tagger working points
    cfg.x.btag_wp_names = btag_params(cfg)
    cfg.x.toptag_wp = toptag_params(cfg)

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
                "min_value": 3,  # 0 = fail, 1 = veto, 2 = loose, 3 = medium, 4 = tight
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
                "column": "btagDeepFlavB" if year != 2024 else "btagUParTAK4B",
                # "column": "btagDeepFlavB" if year != 2024 else "btagPNetB",
                "wp": cfg.x.btag_wp_names.deepjet.medium if year != 2024 else cfg.x.btag_wp_names.UParTAK4.medium,
                # "wp": cfg.x.btag_wp.deepjet.medium if year != 2024 else cfg.x.btag_wp.particle_net.medium,
            },
        },
        "ak8": {
            "column": "FatJet",
            "max_abseta": 2.5,
            "min_pt": {
                "baseline": 200,
                "toptagged": 400,
            },
            "msoftdrop": [105, 210],
            "toptagger": {
                "column": ["particleNetWithMass_TvsQCD"] if year != 2024 else [
                    "globalParT3_TopbWqq",
                    "globalParT3_TopbWq",
                    "globalParT3_QCD",
                ],
                "wp": cfg.x.toptag_wp.particle_net.tight if year != 2024 else cfg.x.toptag_wp.GloParTv3.tight,
            },
            "delta_r_lep": 0.8,
        },
    })

    # MET selection parameters
    cfg.x.met_selection = DotDict.wrap({
        "column": "PuppiMET",
        "raw_column": "RawPuppiMET",
        "min_pt": {
            "e": 60,
            "mu": 70,
        },
    })

    # lepton jet 2D isolation parameters
    cfg.x.lepton_jet_iso = DotDict.wrap({
        "min_pt": 15,
        "min_delta_r": 0.4,
        "min_pt_rel": 25,
    })

    # trigger paths for muon/electron channels
    # TODO update to relevant Run 3 triggers
    cfg.x.triggers = DotDict.wrap({
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
        "Flag.hfNoisyHitsFilter",
        "Flag.eeBadScFilter",
        "Flag.ecalBadCalibFilter",
    }

    #
    # luminosity
    #

    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis
    if year == 2022 and campaign.x.EE == "pre":
        cfg.x.luminosity = Number(7_980.4541, {
            "lumi_13p6TeV_2022": 0.014j,
        })
    elif year == 2022 and campaign.x.EE == "post":
        cfg.x.luminosity = Number(26_671.6097, {
            "lumi_13p6TeV_2022": 0.014j,
        })
    elif year == 2023 and campaign.x.BPix == "pre":
        cfg.x.luminosity = Number(18_062.6591, {
            "lumi_13p6TeV_2023": 0.013j,
        })
    elif year == 2023 and campaign.x.BPix == "post":
        cfg.x.luminosity = Number(9_693.1301, {
            "lumi_13p6TeV_2023": 0.013j,
        })
    elif year == 2024:
        cfg.x.luminosity = Number(109_080.0, {  # TODO: update number
            "lumi_13p6TeV_2024": 0.013j,
        })
        # processed lumi for limited configs
        # cfg.x.luminosity = Number(995.223558512, {
        #     "lumi_13p6TeV_2024": 0.013j,
        # })
    else:
        raise NotImplementedError(f"Luminosity for year {year} is not defined.")

    #
    # ttbar reconstruction parameters
    #

    # chi2 tuning parameters (mean masses/widths of top quarks
    # with hadronically/leptonically decaying W bosons)
    # AN2019_197_v3
    # TODO: update to Run 3 values
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

    # working points for event categorization
    cfg.x.categorization = DotDict({
        "chi2_max": 30,
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
        if year != 2024:
            btag_uncs = [
                "hf", "lf",
                "hfstats1", "hfstats2",
                "lfstats1", "lfstats2",
                "cferr1", "cferr2",
            ]
            for i, unc in enumerate(btag_uncs):
                cfg.add_shift(name=f"btag_{unc}_up", id=501 + 2 * i, type="shape")
                cfg.add_shift(name=f"btag_{unc}_down", id=502 + 2 * i, type="shape")
                add_shift_aliases(
                    cfg,
                    f"btag_{unc}",
                    {
                        # PREVIOUS IMPLEMENTATION (still used in some configs?)
                        # taken from
                        # https://github.com/uhh-cms/hh2bbww/blob/c6d4ee87a5c970660497e52aed6b7ebe71125d20/hbw/config/config_run2.py#L421
                        "normalized_btag_weight": f"normalized_btag_weight_{unc}_" + "{direction}",
                        "normalized_njet_btag_weight": f"normalized_njet_btag_weight_{unc}_" + "{direction}",
                        "btag_weight": f"btag_weight_{unc}_" + "{direction}",
                        "njet_btag_weight": f"njet_btag_weight_{unc}_" + "{direction}",
                    },
                )
        else:
            # https://cms-analysis-corrections.docs.cern.ch/corrections_era/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/BTV/2025-08-19/#btagging_preliminaryjsongz  # noqa
            btag_uncs = [
                "fsrdef", "isrdef",
                "hdamp", "jer", "jes",
                "mass", "statistic",
                "tune",
            ]
            for i, unc in enumerate(btag_uncs):
                cfg.add_shift(name=f"btag_{unc}_up", id=501 + 2 * i, type="shape")
                cfg.add_shift(name=f"btag_{unc}_down", id=502 + 2 * i, type="shape")
                add_shift_aliases(
                    cfg,
                    f"btag_{unc}",
                    {
                        # UPDATED FOR 2024 USING UParTAK4B for b-tagging
                        "normalized_btag_weight_upart": f"btagUParTAK4B_shape_weight_{unc}_" + "{direction}",
                        "normalized_njet_btag_weight_upart": f"btagUParTAK4B_shape_weight_{unc}_" + "{direction}",
                    },
                )

        # jet energy scale (JEC) uncertainty variations
        for jec_source in cfg.x.jec.Jet.uncertainty_sources:
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

    #
    # corrections
    #

    cfg.x.vjets_reweighting = vjets_reweighting_cfg()
    cfg.x.jec, cfg.x.jer = jerc_cfg(campaign, year)
    # add the shifts
    add_shifts(cfg)

    cfg.x.btag_sf = btag_sf_cfg(year)
    cfg.x.toptag_sf = toptag_sf_cfg()

    cfg.x.electron_sf = lepton_sf_cfg(cfg, "electron")

    cfg.x.muon_sf_names = lepton_sf_cfg(cfg, "muon")[0]
    cfg.x.muon_id_sf_names = lepton_sf_cfg(cfg, "muon")[1]
    cfg.x.muon_iso_sf_names = lepton_sf_cfg(cfg, "muon")[2]

    cfg.x.met_phi_correction = met_phi_cfg(cfg)  # METPhiConfig object
    cfg.x.jet_id = jet_id_cfg()["Jet"]  # JetIdConfig object
    cfg.x.fatjet_id = jet_id_cfg()["FatJet"]  # JetIdConfig object

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
        "electron_weight": get_shifts("electron"),
        # "ISR": get_shifts("ISR"),
        # "FSR": get_shifts("FSR"),
        # TODO: add scale and PDF weights, where available
        # "scale_weight": ???,
        # "pdf_weight": ???,
    })

    # event weights only present in certain datasets
    for dataset in cfg.datasets:
        dataset.x.event_weights = DotDict()

        # TTbar: top pt reweighting
        if dataset.has_tag("is_ttbar"):
            dataset.x.event_weights["top_pt_weight"] = get_shifts("top_pt")

        # V+jets: QCD NLO reweighting (disable for now)
        # if dataset.has_tag("is_v_jets"):
        #     dataset.x.event_weights["vjets_weight"] = get_shifts("vjets")

    #
    # external files
    # setup taken from https://github.com/uhh-cms/hh2bbtautau/blob/ed8f363ac239b0257fc7f470b96f5c09a0572c34/hbt/config/configs_hbt.py#L1574  # noqa: E501
    # https://cms-analysis-corrections.docs.cern.ch
    #

    cfg.x.external_files = DotDict()

    # helper
    def add_external(name, value):
        if isinstance(value, dict):
            value = DotDict.wrap(value)
        cfg.x.external_files[name] = value

    # prepare run/era/nano meta data info to determine files in the CAT metadata structure
    # see https://cms-analysis-corrections.docs.cern.ch
    cat_info = {
        (2022, "", 12): CATInfo(
            run=3,
            vnano=12,
            era="22CDSep23-Summer22",
            pog_directories={"dc": "Collisions22"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-04-15", jme="2025-09-23", lum="2024-01-31", muo="2025-08-14", tau="2025-10-01"),  # noqa: E501
        ),
        (2022, "EE", 12): CATInfo(
            run=3,
            vnano=12,
            era="22EFGSep23-Summer22EE",
            pog_directories={"dc": "Collisions22"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-04-15", jme="2025-10-07", lum="2024-01-31", muo="2025-08-14", tau="2025-10-01"),  # noqa: E501
        ),
        (2023, "", 12): CATInfo(
            run=3,
            vnano=12,
            era="23CSep23-Summer23",
            # pog_eras={"tau": "23CSep23-Summer22"},  # TODO: remove once typo in CAT repo is fixed
            pog_directories={"dc": "Collisions23"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-04-15", jme="2025-10-07", lum="2024-01-31", muo="2025-08-14", tau="2025-10-01"),  # noqa: E501
        ),
        (2023, "BPix", 12): CATInfo(
            run=3,
            vnano=12,
            era="23DSep23-Summer23BPix",
            pog_directories={"dc": "Collisions23"},
            snapshot=CATSnapshot(btv="2025-08-20", dc="2025-07-25", egm="2025-04-15", jme="2025-10-07", lum="2024-01-31", muo="2025-08-14", tau="2025-10-01"),  # noqa: E501
        ),
        (2024, "", 15): CATInfo(
            run=3,
            vnano=15,
            era="24CDEReprocessingFGHIPrompt-Summer24",
            pog_directories={"dc": "Collisions24"},
            snapshot=CATSnapshot(btv="2025-12-03", dc="2025-07-25", egm="2025-12-03", jme="2025-12-02", muo="2025-11-27", lum="2025-12-02"),  # noqa: E501
        ),
    }[(year, campaign.x.postfix, vnano)]
    cfg.x.cat_info = cat_info

    # common files
    # (versions in the end are for hashing in cases where file contents changed but paths did not)
    add_external("lumi", {
        "golden": {
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2022
            2022: (cat_info.get_file("dc", "Cert_Collisions2022_355100_362760_Golden.json"), "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2023
            2023: (cat_info.get_file("dc", "Cert_Collisions2023_366442_370790_Golden.json"), "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=180#Year_2024
            # not yet available at CAT space
            # 2024: (cat_info.get_file("dc", "Cert_Collisions2024_378981_386951_Golden.json"), "v1"),
            2024: ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/Cert_Collisions2024_378981_386951_Golden.json", "v1"),  # noqa: E501
        }[year],
        "normtag": {
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2022
            2022: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2023
            2023: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=180#Year_2024
            2024: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),  # TODO: correct?
        }[year],
    })

    # pileup weight corrections
    if year != 2024:  # TODO: not yet available, see https://cms-analysis-corrections.docs.cern.ch
        add_external("pu_sf", (cat_info.get_file("lum", "puWeights.json.gz"), "v1"))
    elif year == 2024:
        add_external("pu_sf", (cat_info.get_file("lum", "puWeights_BCDEFGHI.json.gz"), "v1"))

    # jet energy correction
    add_external("jet_jerc", (cat_info.get_file("jme", "jet_jerc.json.gz"), "v1"))

    # fat jet energy correction
    add_external("fat_jet_jerc", (cat_info.get_file("jme", "fat_jet_jerc.json.gz" if year != 2024 else "fatJet_jerc.json.gz"), "v1"))  # noqa: E501

    # jet veto map
    add_external("jet_veto_map", (cat_info.get_file("jme", "jetvetomaps.json.gz"), "v1"))

    # btag scale factor
    if year != 2024:
        add_external("btag_sf_corr", (cat_info.get_file("btv", "btagging.json.gz"), "v1"))
    else:
        # SF stored in preliminary file for 2024 for now?
        add_external("btag_sf_corr", (cat_info.get_file("btv", "btagging_preliminary.json.gz"), "v1"))  # noqa: E501

    # updated jet id
    add_external("jet_id", (cat_info.get_file("jme", "jetid.json.gz"), "v1"))

    # muon scale factors
    add_external("muon_sf", (cat_info.get_file("muo", "muon_Z.json.gz"), "v1"))

    # met phi correction
    if year != 2024:  # TODO: not yet available for 2024
        add_external("met_phi_corr", (cat_info.get_file("jme", f"met_xyCorrections_{year}_{year}{campaign.x.postfix}.json.gz"), "v1"))  # noqa: E501

    # electron scale factors
    add_external("electron_sf", (cat_info.get_file("egm", "electron.json.gz"), "v1"))
    # electron energy correction and smearing
    add_external("electron_ss", (cat_info.get_file("egm", "electronSS_EtDependent.json.gz"), "v1"))  # FIXME correct for us? # noqa: E501

    # # top-tagging scale factors (TODO)
    # "toptag_sf": (f"{sources['jet']}/JMAR/???/???.json", "v1"),  # noqa

    # # V+jets reweighting
    # "vjets_reweighting": f"{sources['local_repo']}/data/json/vjets_reweighting.json",

    # if year == 2022 and campaign.x.EE == "pre":
    #     cfg.x.external_files.update(DotDict.wrap({
    #         # lumi files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
    #         "lumi": {
    #             "golden": (f"{sources['cert']}/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
    #             "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    #         },

    #         # pileup files (for PU reweighting)
    #         "pu": {
    #             # "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileup_JSON.txt", "v1"),  # noqa
    #             "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCDEFG/pileup_JSON.txt", "v1"),  # noqa
    #             "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/bb525104a7ddb93685f8ced6fed1ab793b2d2103/SimGeneral/MixingModule/python/Run3_2022_LHC_Simulation_10h_2h_cfi.py", "v1"),  # noqa
    #             "data_profile": {
    #                 # "nominal": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-69200ub-99bins.root", "v1"),  # noqa
    #                 # "minbias_xs_up": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-72400ub-99bins.root", "v1"),  # noqa
    #                 # "minbias_xs_down": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCD/pileupHistogram-Cert_Collisions2022_355100_357900_eraBCD_GoldenJson-13p6TeV-66000ub-99bins.root", "v1"),  # noqa
    #                 "nominal": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-69200ub-100bins.root", "v1"),  # noqa
    #                 "minbias_xs_up": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-72400ub-100bins.root", "v1"),  # noqa
    #                 "minbias_xs_down": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-66000ub-100bins.root", "v1"),  # noqa
    #             },
    #         },
    #     }))
    # elif year == 2022 and campaign.x.EE == "post":
    #     cfg.x.external_files.update(DotDict.wrap({
    #         # files from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
    #         "lumi": {
    #             "golden": ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/Cert_Collisions2022_355100_362760_Golden.json", "v1"),  # noqa
    #             "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    #         },

    #         # pileup files (for PU reweighting)
    #         "pu": {
    #             # "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileup_JSON.txt", "v1"),  # noqa
    #             "json": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/BCDEFG/pileup_JSON.txt", "v1"),  # noqa
    #             "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/bb525104a7ddb93685f8ced6fed1ab793b2d2103/SimGeneral/MixingModule/python/Run3_2022_LHC_Simulation_10h_2h_cfi.py", "v1"),  # noqa
    #             "data_profile": {
    #                 # data profiles were produced with 99 bins instead of 100 --> use custom produced data profiles instead  # noqa
    #                 # "nominal": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-69200ub-99bins.root", "v1"),  # noqa
    #                 # "minbias_xs_up": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-72400ub-99bins.root", "v1"),  # noqa
    #                 # "minbias_xs_down": (f"https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/PileUp/EFG/pileupHistogram-Cert_Collisions2022_359022_362760_eraEFG_GoldenJson-13p6TeV-66000ub-99bins.root", "v1"),  # noqa
    #                 "nominal": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-69200ub-100bins.root", "v1"),  # noqa
    #                 "minbias_xs_up": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-72400ub-100bins.root", "v1"),  # noqa
    #                 "minbias_xs_down": (f"/afs/cern.ch/user/a/anhaddad/public/Collisions22/pileupHistogram-Cert_Collisions2022_355100_362760_GoldenJson-13p6TeV-66000ub-100bins.root", "v1"),  # noqa
    #             },
    #         },
    #     }))
    # else:
    #     raise NotImplementedError(f"No lumi and pu files provided for year {year}")

    #
    # set defaults for
    # calibrator, selector etc
    # process, dataset, category, variable, shift groups
    #

    set_defaults(cfg)
    set_process_groups(cfg)
    set_dataset_groups(cfg)
    set_category_groups(cfg)
    set_variables_groups(cfg)
    set_shift_groups(cfg)
    set_selector_steps(cfg)

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
            "LHEPdfWeight",
            "LHEScaleWeight",
            "PSWeight",

            # muons
            "{Muon,VetoMuon}.{pt,eta,phi,mass}",
            "Muon.pfRelIso04_all",
            # electrons
            "{Electron,VetoElectron}.{pt,eta,phi,mass}",
            "Electron.{deltaEtaSC,pfRelIso03_all}",

            # photons (for L1 prefiring)
            "Photon.{pt,eta,phi,mass,jetIdx}",

            # AK4 jets
            "{Jet,BJet,LightJet,LooseJet}.{pt,eta,phi,mass,btagDeepFlavB,hadronFlavour,btagUParTAK4B}",
            "Jet.rawFactor",

            # AK8 jets
            "{FatJet,FatJetTopTag,FatJetTopTagDeltaRLepton}.{pt,eta,phi,mass,rawFactor}",
            "{FatJet,FatJetTopTag}.{msoftdrop,particleNetWithMass_TvsQCD,deepTagMD_TvsQCD}",
            "FatJet.globalParT3.{TopbWqq,TopbWq,QCD}",
            "{FatJet,FatJetTopTag,FatJetTopTagDeltaRLepton}.{tau1,tau2,tau3}",
            "FatJetTopTagDeltaRLepton.msoftdrop",
            "FatJetTopTagDeltaRLepton.deepTagDeltaRLeptonMD_TvsQCD",

            # generator quantities
            "Generator.*",

            # generator particles
            "GenPart.*",

            # generator objects
            "GenMET.*",
            "GenJet.*",
            "GenJetAK8.*",

            # missing transverse momentum
            "PuppiMET.{pt,phi,significance,covXX,covXY,covYY}",
            "MET.{pt,phi,significance,covXX,covXY,covYY}",

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

    # versions per task family and optionally also dataset and shift
    # None can be used as a key to define a default value
    cfg.x.versions = {}

    #
    # finalization
    #

    # add channels
    cfg.add_channel("e", id=1)
    cfg.add_channel("mu", id=2)

    # add categories
    add_categories_selection(cfg)

    # add variables
    add_variables(cfg)

    return cfg
