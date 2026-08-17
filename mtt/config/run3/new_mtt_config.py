# coding: utf-8

"""
Configuration for the Run 3 m(ttbar) analysis.
"""

from __future__ import annotations

import functools
import os
import law

import yaml
from scinum import Number

from columnflow.util import DotDict
from columnflow.types import Callable
from columnflow.cms_util import CATInfo, CATSnapshot
from columnflow.config_util import (
    add_shift_aliases,
    get_root_processes_from_campaign,
    get_shifts_from_sources,
    verify_config_processes,
)
from columnflow.production.cms.muon import MuonSFConfig
from columnflow.production.cms.electron import ElectronSFConfig
from mtt.config.categories import add_categories_selection
from mtt.config.variables import add_variables
from mtt.config.datasets import add_datasets_from_yaml
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
    met_phi_cfg,
)
from columnflow.production.cms.jet import JetIdConfig
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")

import order as od

logger = law.logger.get_logger(__name__)


thisdir = os.path.dirname(os.path.abspath(__file__))


def add_new_config(
    analysis: od.Analysis,
    campaign: od.Campaign,
    config_name: str | None = None,
    config_id: int | None = None,
    limit_dataset_files: int | None = None,
    split_mc: bool | None = None,
) -> od.Config:
    """
    Configurable function for creating a config for a run3 analysis given
    a base *analysis* object and a *campaign* (i.e. set of datasets).
    """

    # validation (TODO: why?)
    implemented_years = [2024, 2025]
    assert campaign.x.year in implemented_years
    if campaign.x.year == 2022:
        assert campaign.x.EE in ["pre", "post"]
    elif campaign.x.year == 2023:
        assert campaign.x.BPix in ["pre", "post"]

    if split_mc is None:
        raise ValueError("The flag 'split_mc' must be specified (True/False) for the config.")

    # campaign data
    year = campaign.x.year
    # year2 = year % 100
    vnano = campaign.x.version
    corr_postfix = ""
    if year == 2022:
        corr_postfix = f"{campaign.x.EE}EE"
    elif year == 2023:
        corr_postfix = f"{campaign.x.BPix}BPix"

    if year not in implemented_years:
        raise NotImplementedError(f"Only {', '.join(map(str, implemented_years))} campaigns are implemented.")

    # create a config by passing the campaign
    # (if id and name are not set they will be taken from the campaign)
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)

    # groups for custom plot styling
    cfg.x.custom_style_config_groups = {
        "more_legend": {
            "gridspec_cfg": {
                # "left": 0.08,
                # "right": 0.98,
                "top": 0.75,
                # "bottom": 0.05,
                # "bottom": 0.1,
            },
        },
        "large_ratio": {
            "rax_cfg": {
                "ylim": (0.4, 1.6),
            },
        },
        "DEFAULT": {
            "legend_cfg": {
                "ncols": 2,
                "fontsize": 14,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "annotate_cfg": {
                "xy": (0.60, 0.60),
                "xycoords": "axes fraction",
                "fontsize": 14,
            },
            "rax_cfg": {
                "ylim": (0.4, 1.6),
            },
        },
        "for_an": {
            "legend_cfg": {
                "ncols": 3,
                "fontsize": 12,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "annotate_cfg": {
                "xy": (0.60, 0.60),
                "xycoords": "axes fraction",
                "fontsize": 12,
            },
            "rax_cfg": {
                "ylim": (0.4, 1.6),
            },
        },
        "no_cat_label": {
            "legend_cfg": {"ncols": 2, "fontsize": 20},
            "annotate_cfg": {"text": ""},
        },
        "cutflow": {
            "annotate_cfg": {
                "xy": (0.05, 0.95),
                "xycoords": "axes fraction",
                "fontsize": 14,
            },
            "legend_cfg": {
                "ncols": 2,
                "fontsize": 14,
                "bbox_to_anchor": (0., 0., 1., 1.),
            },
            "ax_cfg": {
                "ylim": (0.000001, 5.0),
            },
        },
        "example": {
            "legend_cfg": {"title": "my custom legend title", "ncols": 2},
            "ax_cfg": {"ylabel": "my ylabel", "xlim": (0, 100)},
            "rax_cfg": {"ylabel": "some other ylabel"},
            "annotate_cfg": {"text": "category label usually here"},
        },
    }

    # add tags to config
    cfg.x.run = 3
    cfg.x.cpn_tag = f"{year}{corr_postfix}"
    cfg.x.year = year
    if limit_dataset_files is not None:
        cfg.add_tag("is_limited")

    # add tags for skipping lepton weights if not available (as of 16.07.26)
    if year == 2024:
        cfg.add_tag("skip_electron_trigger_weights")  # high pt trigger SF not available yet TODO: derive them
        cfg.add_tag("skip_muon_trigger_weights")  # TODO compute high pt custom trigger SF for muons
        cfg.add_tag("skip_btag_weights")
        cfg.add_tag("skip_btag_wp_weights")
    elif year == 2025:
        cfg.add_tag("skip_electron_trigger_weights")  # no trigger SF available yet TODO: derive them
        cfg.add_tag("skip_muon_trigger_weights")  # TODO compute high pt custom trigger SF for muons
        cfg.add_tag("skip_btag_weights")
        cfg.add_tag("skip_btag_wp_weights")
    elif year == 2026:
        cfg.add_tag("skip_electron_weights")  # TODO no SFs at all available yet
        cfg.add_tag("skip_muon_weights")  # TODO no SFs at all available yet
        cfg.add_tag("skip_btag_weights")
        cfg.add_tag("skip_btag_wp_weights")
    else:
        raise NotImplementedError(f"Skipping weights for year {year} is not implemented.")

    cfg.add_tag("skip_kfactor_weights")
    logger.warning_once("Skipping (some) electron, muon, and k-factor weights for now.")

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # add processes and datasets we are interested in
    cfg.add_process(procs.n.data)

    cfg.add_process(procs.n.tt)

    cfg.add_process(procs.n.st)

    cfg.add_process(procs.n.vv)

    # ttbar signal processes
    cfg.add_process(procs.n.zprime_tt)

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

    cfg.add_process(procs.n.dy)

    cfg.add_process(procs.n.qcd)

    cfg.add_process(procs.n.w_lnu)

    # cfg.add_process(procs.n.w_lnu_1j)

    # cfg.add_process(procs.n.w_lnu_2j)

    # cfg.add_process(procs.n.w_lnu_3j)

    # cfg.add_process(procs.n.w_lnu_4j)

    # cfg.add_process(procs.n.hscalar_tt)

    # cfg.add_process(procs.n.hpseudo_tt)

    # cfg.add_process(procs.n.rsgluon_tt)

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
            for subproc in process.get_leaf_processes():
                subproc.scale = "stack"
        else:
            process.x.is_mtt_signal = False

    dataset_names = add_datasets_from_yaml(
        cfg,
        limit_dataset_files=limit_dataset_files,
        dataset_types=[
            "data_mu",
            "data_egamma",
            "tt",
            "st",
            "dy",
            "w_lnu",
            "vv",
            "qcd",
        ],
        log=False,
    )

    if year == 2024:
        dataset_names += add_datasets_from_yaml(
            cfg,
            limit_dataset_files=limit_dataset_files,
            dataset_types=[
                "zprime_full",
            ],
            log=False,
        )
    elif year == 2025:
        dataset_names += add_datasets_from_yaml(
            cfg,
            limit_dataset_files=limit_dataset_files,
            dataset_types=[
                "zprime_full",
            ],
            log=False,
        )
    logger.info_once(f"Added {len(dataset_names)} datasets to config {cfg.name} (config id: {cfg.id})")

    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    cfg.x.validate_dataset_lfns = limit_dataset_files is None

    # MC splitting settings
    # taken from https://github.com/uhh-cms/hh2bbtautau/blob/2b227967d8cc86351908ac834144a63e2d3733f1/hbt/config/configs_hbt.py#L2458-L2475  # noqa

    if split_mc:
        # for details see https://gist.github.com/riga/5aad795980919d23ea6b7df0502a999b
        if year not in {2024, 2025, 2026}:
            raise ValueError(f"MC splitting is not supported in {year}")

        _splitter = None

        def get_nano_filter_config(task, target) -> tuple[Callable[[ak.Array], np.ndarray], set[str]] | None:
            # skip for data
            if task.dataset_inst.is_data:
                return None

            # define splitting function and required column names for mc
            def nano_filter_func(events: ak.Array) -> ak.Array | np.ndarray:
                nonlocal _splitter
                if _splitter is None:
                    from columnflow.util import load_correction_set
                    splitter_path = "/data/dust/user/matthiej/mttbar/mtt/config/run3/data/mc_event_splitter.json.gz"
                    _splitter = load_correction_set(splitter_path)["mc_event_splitter"]
                return _splitter.evaluate(events.event) == year

            columns = {"event"}

            return nano_filter_func, columns

        # register on config to be picked up by ChunkedIOMixin's
        cfg.x.get_nano_filter_config = get_nano_filter_config

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
    }
    zprime_colors = {
        # 1% width signals
        "zprime_tt_m500_w5": "#abcded",  # light blue
        "zprime_tt_m1000_w10": "#a2f6c2",  # light green
        "zprime_tt_m3000_w30": "#dbbdc6",  # light mauve
        "zprime_tt_m7000_w70": "#ffcc99",  # light orange
        # 10 % width signals
        "zprime_tt_m500_w50": "#11304d",  # blue
        "zprime_tt_m1000_w100": "#11ae4d",  # green
        "zprime_tt_m3000_w300": "#8a4b5d",  # mauve
        "zprime_tt_m7000_w700": "#ff7f00",  # orange
        # 30 % width signals
        "zprime_tt_m500_w150": "#091a2a",  # dark blue
        "zprime_tt_m1000_w300": "#052e15",  # dark green
        "zprime_tt_m3000_w900": "#4a2530",  # dark mauve
        "zprime_tt_m7000_w2100": "#331900",  # dark orange
    }

    # process settings groups to quickly define settings for ProcessPlots
    cfg.x.process_settings_groups = {
        "default": [
            ["zprime_tt_m500_w50", "scale=2000", "unstack"],
        ],
        "unstack_all": [
            [proc, "unstack"] for proc in cfg.processes
        ],
        "unstack_signal": [
            [proc, "unstack"] for proc in cfg.processes
            if proc.name.startswith("zprime_tt")
        ],
        "scale_signal": [
            [proc, "scale=stack"] for proc in cfg.processes
            if proc.name.startswith("zprime_tt")
        ],
    }
    from mtt.util import build_zprime_labels
    zprime_procs_list = []
    for proc in cfg.processes:
        if proc.name.startswith("zprime_tt"):
            zprime_proc = cfg.processes.get("zprime_tt")
            for sig_proc in zprime_proc.get_leaf_processes():
                zprime_procs_list.append(sig_proc.name)

    zprime_mass_labels = build_zprime_labels(
        zprime_procs_list,
    )

    for proc, zprime_mass_label in zprime_mass_labels.items():
        proc_inst = cfg.get_process(proc)
        proc_inst.label = zprime_mass_label
        if proc in zprime_colors:
            proc_inst.color1 = zprime_colors[proc]
            proc_inst.color2 = zprime_colors[proc]

    for proc in cfg.processes:
        proc_inst = cfg.get_process(proc)
        if proc.name not in zprime_colors.keys():
            proc_inst.color1 = colors.get(proc.name, "#aaaaaa")
            proc_inst.color2 = colors.get(proc.name, "#000000")

    # verify that the root processes of each dataset (or one of their
    # ancestor processes) are registered in the config
    verify_config_processes(cfg, warn=True)
    logger.debug(f"Added {len(cfg.processes)} processes and {len(cfg.datasets)} datasets to config '{cfg.name}'")

    # add tagger working points
    # cfg.x.btag_wp = btag_params(cfg)

    # full b-tag working points dict
    cfg.x.btag_wp = btag_params(cfg, full=True)
    # store only upart wps for fixed wp sf producer
    cfg.x.btag_wp.btagUParTAK4B.fixed_wp = btag_params(cfg, full=False)
    cfg.x.toptag_wp = toptag_params(cfg)

    #
    # selector configuration
    #

    # lepton selection parameters
    logger.warning(
        "Update cfg.x.lepton_selection['e']['id_addveto']['min_value'] to 4"
        "when rerunning selection to match Run 2 analysis.",
    )
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
                "column": "btagDeepFlavB" if year not in [2024, 2025] else "btagUParTAK4B",
                "wp": cfg.x.btag_wp.btagUParTAK4B.fixed_wp.medium if year in [2024, 2025] else cfg.x.btag_wp.deepjet.medium,
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
                "column": ["particleNetWithMass_TvsQCD"] if year not in [2024, 2025] else [
                    "globalParT3_TopbWqq",
                    "globalParT3_TopbWq",
                    "globalParT3_QCD",
                ],
                "wp": cfg.x.toptag_wp.GloParTv3.tight if year in [2024, 2025] else cfg.x.toptag_wp.particle_net.tight,
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
                        # "IsoMu27",
                        "IsoMu24",  # updated Run 3 recommendation
                    },
                    "electron": {
                        # "Ele35_WPTight_Gsf",
                        "Ele30_WPTight_Gsf",  # updated Run 3 recommendation
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
                        "CascadeMu100",  # updated Run 3 recommendation
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
        cfg.x.luminosity = Number(109_950.0 - 130.0, {
            "lumi_13p6TeV_2024": 0.016j,  # CERN-CMS-DP-2026-003
        })
        # processed lumi for limited configs
        # cfg.x.luminosity = Number(995.223558512, {
        #     "lumi_13p6TeV_2024": 0.013j,
        # })
    elif year == 2025:
        cfg.x.luminosity = Number(110_840, {
            "lumi_13p6TeV_2025": 0.016j,  # FIXME placeholder atm, fill in when available
        })
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
    # corrections
    #

    cfg.x.vjets_reweighting = vjets_reweighting_cfg()
    cfg.x.jec, cfg.x.jer = jerc_cfg(campaign, year)

    if year in [2024, 2025, 2026]:
        cfg.x.jet_id = JetIdConfig(
            corrections={"AK4PUPPI_Tight": 2, "AK4PUPPI_TightLeptonVeto": 3},
        )
        cfg.x.fatjet_id = JetIdConfig(
            corrections={"AK8PUPPI_Tight": 2, "AK8PUPPI_TightLeptonVeto": 3},
        )
    else:
        logger.debug(f"(Fat)Jet ID recalculation not configured for {cfg.x.cpn_tag} campaign. Will be skipped.")
        cfg.add_tag("skip_jet_ids")

    # cfg.x.btag_sf = btag_sf_cfg(year)
    cfg.x.btag_sf_jec_sources = btag_sf_cfg(cfg, year)
    cfg.x.toptag_sf = toptag_sf_cfg()

    # electron SF configs for low and high pt regions
    # reco
    cfg.x.electron_reco_weight_low_pt_sf_config = ElectronSFConfig(
        correction="Electron-ID-SF",
        campaign=f"{year}Prompt",
        working_point={
            "RecoBelow20": (lambda variables: variables["pt"] < 20),
            "Reco20to75": (lambda variables: (variables["pt"] >= 20) & (variables["pt"] < 75.0)),
            "RecoAbove75": (lambda variables: variables["pt"] >= 75.0),
        },
    )
    cfg.x.electron_reco_weight_high_pt_sf_config = ElectronSFConfig(
        correction="Electron-ID-SF",
        campaign=f"{year}Prompt",
        working_point={
            "RecoBelow20": (lambda variables: variables["pt"] < 20),
            "Reco20to75": (lambda variables: (variables["pt"] >= 20) & (variables["pt"] < 75.0)),
            "RecoAbove75": (lambda variables: variables["pt"] >= 75.0),
        },
    )
    # id and iso
    cfg.x.electron_id_iso_weight_low_pt_sf_config = ElectronSFConfig(
        correction="Electron-ID-SF",
        campaign=f"{year}Prompt",
        working_point={
            "wp80iso": (lambda variables: variables["pt"] > 10),
        },
    )
    cfg.x.electron_id_weight_high_pt_sf_config = ElectronSFConfig(
        correction="Electron-ID-SF",
        campaign=f"{year}Prompt",
        working_point={
            "wp80noiso": (lambda variables: variables["pt"] > 10),
        },
    )
    cfg.x.electron_iso_weight_high_pt_sf_config = ElectronSFConfig(
        correction="NOT_YET_AVAILABLE",  # TODO derive high pt iso SF for custom iso definition
        campaign=f"{year}Prompt",
        working_point={
            "wp80iso": (lambda variables: variables["pt"] > 10),
        },
    )
    # trigger
    cfg.x.electron_trigger_weight_low_pt_sf_config = ElectronSFConfig(
        correction="Electron-HLT-SF",
        campaign=f"{year}Prompt",
        hlt_path="HLT_SF_Ele30_MVAiso80ID",
    )
    cfg.x.electron_trigger_weight_high_pt_sf_config = ElectronSFConfig(
        correction="NOT_YET_AVAILABLE",  # TODO derive high pt trigger SF
        campaign=f"{year}Prompt",
        hlt_path="HLT_SF_Ele30_MVAiso80ID",
    )

    # muon SF configs for low and high pt regions
    # reco
    cfg.x.muon_reco_low_pt_sf_config = MuonSFConfig(
        correction="NOT_NEEDED",
    )
    cfg.x.muon_reco_high_pt_sf_config = MuonSFConfig(
        correction="NUM_GlobalMuons_DEN_TrackerMuonProbes",
    )

    # id
    cfg.x.muon_id_weight_low_pt_sf_config = MuonSFConfig(
        correction="NUM_TightID_DEN_TrackerMuons",
    )
    cfg.x.muon_id_weight_high_pt_sf_config = MuonSFConfig(
        correction="NUM_HighPtID_DEN_GlobalMuonProbes",
    )

    # iso
    cfg.x.muon_iso_weight_low_pt_sf_config = MuonSFConfig(
        correction="NUM_TightPFIso_DEN_TightID",
    )
    cfg.x.muon_iso_weight_high_pt_sf_config = MuonSFConfig(
        correction="TO_BE_DERIVED",  # TODO derive high pt iso SF for custom iso definition
    )

    # trigger
    cfg.x.muon_trigger_weight_low_pt_sf_config = MuonSFConfig(
        correction="NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
    )
    cfg.x.muon_trigger_weight_high_pt_sf_config = MuonSFConfig(
        correction="TO_BE_DERIVED",  # TODO derive high pt trigger SF
    )

    cfg.x.met_phi_correction = met_phi_cfg(cfg)  # METPhiConfig object

    # top pt reweighting parameters
    # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_dat?rev=31
    cfg.x.top_pt_reweighting_params = {
        "a": 0.0615,
        "b": -0.0005,
    }

    #
    # systematic shifts
    #

    # read in JEC sources from file
    with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
        all_jec_sources = yaml.load(f, yaml.Loader)["names"]
    btag_uncs_bc = [
        "fsrdef", "isrdef",
        "hdamp", "jer", "jes",
        "mass", "statistic",
        "tune",
    ]
    btag_uncs_bc_full = [f"{unc}_bc" for unc in btag_uncs_bc] + ["bc"]
    btag_uncs_light = [
        "",
        "correlated", "uncorrelated",
    ]
    btag_uncs_light_full = [f"{unc}_light" for unc in btag_uncs_light] + ["light"]

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
            cfg.add_shift(name="muon_id_up", id=111, type="shape")
            cfg.add_shift(name="muon_id_down", id=112, type="shape")
            add_shift_aliases(cfg, "muon_id", {"muon_id_weight": "muon_id_weight_{direction}"})
            cfg.add_shift(name="muon_iso_up", id=113, type="shape")
            cfg.add_shift(name="muon_iso_down", id=114, type="shape")
            add_shift_aliases(cfg, "muon_iso", {"muon_iso_weight": "muon_iso_weight_{direction}"})

        # event weights due to electron scale factors
        if not cfg.has_tag("skip_electron_weights"):
            cfg.add_shift(name="electron_reco_up", id=121, type="shape")
            cfg.add_shift(name="electron_reco_down", id=122, type="shape")
            add_shift_aliases(cfg, "electron_reco", {"electron_reco_weight": "electron_reco_weight_{direction}"})
            cfg.add_shift(name="electron_id_iso_up", id=123, type="shape")
            cfg.add_shift(name="electron_id_iso_down", id=124, type="shape")
            add_shift_aliases(cfg, "electron_id_iso", {"electron_id_iso_weight": "electron_id_iso_weight_{direction}"})

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
            for i, unc in enumerate(btag_uncs_bc):
                cfg.add_shift(name=f"btag_{unc}_bc_up", id=501 + 4 * i, type="shape")
                cfg.add_shift(name=f"btag_{unc}_bc_down", id=502 + 4 * i, type="shape")
                add_shift_aliases(
                    cfg,
                    f"btag_{unc}_bc",
                    {
                        f"btag_weight": f"btag_weight_{unc}_bc_" + "{direction}",
                    },
                )
            for i, unc in enumerate(btag_uncs_light):
                cfg.add_shift(name=f"btag_{unc}_light_up", id=503 + 4 * i, type="shape")
                cfg.add_shift(name=f"btag_{unc}_light_down", id=504 + 4 * i, type="shape")
                add_shift_aliases(
                    cfg,
                    f"btag_{unc}_light",
                    {
                        f"btag_weight": f"btag_weight_{unc}_light_" + "{direction}",
                    },
                )

            cfg.add_shift(name="btag_bc_up", id=501 + 4 * len(btag_uncs_bc), type="shape")
            cfg.add_shift(name="btag_bc_down", id=502 + 4 * len(btag_uncs_bc), type="shape")
            cfg.add_shift(name="btag_light_up", id=503 + 4 * len(btag_uncs_light), type="shape")
            cfg.add_shift(name="btag_light_down", id=504 + 4 * len(btag_uncs_light), type="shape")
            add_shift_aliases(
                cfg,
                "btag_bc",
                {
                    "btag_weight": "btag_weight_bc_" + "{direction}",
                },
            )
            add_shift_aliases(
                cfg,
                "btag_light",
                {
                    "btag_weight": "btag_weight_light_" + "{direction}",
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
    # add the shifts
    add_shifts(cfg)

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
    full_btag_uncs = btag_uncs_bc_full + btag_uncs_light_full
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "pu_weight": get_shifts("minbias_xs"),
        "muon_id_weight": get_shifts("muon_id"),
        "muon_iso_weight": get_shifts("muon_iso"),
        "electron_reco_weight": get_shifts("electron_reco"),
        "electron_id_iso_weight": get_shifts("electron_id_iso"),
        "btag_weight": get_shifts(*(f"btag_{unc}" for unc in full_btag_uncs)),
        # "ISR": get_shifts("ISR"),
        # "FSR": get_shifts("FSR"),
        # TODO: add scale and PDF weights, where available
        # "scale_weight": ???,
        # "pdf_weight": ???,
    })

    for dataset in cfg.datasets:
        # event weights only present in certain datasets
        dataset.x.event_weights = DotDict()

        # group datasets together for btag WP efficiency calculation in 2024
        if year in [2024, 2025]:
            # TODO figure out which datasets should be grouped together;
            # for now, don't group any datasets together and treat each type of dataset separately
            cfg.x.btag_wp_eff_groups = [
                ["tt_*", "st_*", "zprime_tt_*", "qcd_*", "ww_*", "dy_*", "w_lnu_*", "wz_*", "zz_*"],
                # ["dy_*"],
                # ["w_lnu_*"],
                # ["ww_*", "wz_*", "zz_*"],
                # ["qcd_*"],
                # ["zprime_tt_*"],
                # ["dy_*", "w_lnu_*", "wz_*", "zz_*"],
            ]
            group_matched = False
            for i, dataset_pattern in enumerate(cfg.x.btag_wp_eff_groups):
                if law.util.multi_match(dataset.name, dataset_pattern):
                    if group_matched:
                        raise ValueError(
                            f"dataset '{dataset.name}' already has a btag WP group assigned! Cannot assign it to more "
                            "than one group",
                        )
                    group_matched = True
                    dataset.add_tag(f"btag_wp_eff_group_{i}")
            if not group_matched and dataset.is_mc:
                raise ValueError(f"no btag_wp_eff_group_* assigned to dataset '{dataset.name}'")
            if group_matched and dataset.is_data:
                raise ValueError(f"must not assign btag_wp_eff_group_* to dataset '{dataset.name}'")

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
            snapshot=CATSnapshot(btv="2026-03-10", dc="2025-07-25", egm="2025-12-15", jme="2026-07-16", muo="2025-11-27", lum="2025-12-02"),  # noqa: E501
        ),
        (2025, "", 15): CATInfo(
            run=3,
            vnano=15,
            era="25Prompt-Summer24",
            pog_directories={"dc": "Collisions25"},
            snapshot=CATSnapshot(btv="2026-06-26", dc="2025-07-25", egm="2026-06-26", jme="2026-07-14", muo="2026-04-28", lum="2026-06-05"),  # noqa: E501
        ),
        (2026, "", 15): CATInfo(
            run=3,
            vnano=15,
            era="26Prompt-Summer24",
            pog_directories={"dc": "Collisions26"},
            snapshot=CATSnapshot(jme="2026-07-15"),  # noqa: E501
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
            2025: ("https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions25/Cert_Collisions2025_391658_398903_Golden.json", "v1"),  # noqa: E501
        }[year],
        "normtag": {
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2022
            2022: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=161#Year_2023
            2023: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),
            # https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun3Analysis?rev=180#Year_2024
            2024: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),  # TODO: correct?
            2025: ("/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json", "v1"),  # TODO: correct?
        }[year],
    })

    # pileup weight corrections
    if year == 2025:
        add_external("pu_sf", (cat_info.get_file("lum", "puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz"), "v1"))
    elif year == 2024:
        add_external("pu_sf", (cat_info.get_file("lum", "puWeights_BCDEFGHI.json.gz"), "v1"))
    else:
        add_external("pu_sf", (cat_info.get_file("lum", "puWeights.json.gz"), "v1"))

    # jet energy correction
    add_external("jet_jerc", (cat_info.get_file("jme", "jet_jerc.json.gz"), "v1"))

    # fat jet energy correction
    add_external("fat_jet_jerc", (cat_info.get_file("jme", "fat_jet_jerc.json.gz" if year not in [2024, 2025, 2026] else "fatJet_jerc.json.gz"), "v1"))  # noqa: E501

    # jet veto map
    add_external("jet_veto_map", (cat_info.get_file("jme", "jetvetomaps.json.gz"), "v1"))

    # btag scale factor
    # add_external("btag_wp_sf_corr", (cat_info.get_file("btv", "btagging.json.gz"), "v1"))
    if year not in [2024, 2025]:
        add_external("btag_sf_corr", (cat_info.get_file("btv", "btagging.json.gz"), "v1"))
    else:
        # SF stored in preliminary file for 2024 for now
        # add_external("btag_sf_corr", (cat_info.get_file("btv", "btagging_preliminary.json.gz"), "v1"))  # noqa: E501
        # use custom file with merged SF for both b/c and light jets
        add_external("btag_wp_sf_corr", ("/data/dust/user/matthiej/mttbar/mtt/config/run3/btagging_preliminary_merged.json.gz", "v1"))  # noqa: E501

    # updated jet id
    add_external("jet_id", (cat_info.get_file("jme", "jetid.json.gz"), "v1"))

    # muon scale factors
    add_external("muon_low_pt_sf", (cat_info.get_file("muo", "muon_Z.json.gz"), "v1"))
    add_external("muon_high_pt_sf", (cat_info.get_file("muo", "muon_HighPt.json.gz"), "v1"))

    # met phi correction
    if year not in [2024, 2025]:  # TODO: not yet available for 2024
        add_external("met_phi_corr", (cat_info.get_file("jme", f"met_xyCorrections_{year}_{year}{campaign.x.postfix}.json.gz"), "v1"))  # noqa: E501

    # electron scale factors
    add_external("electron_sf", (cat_info.get_file("egm", "electron.json.gz"), "v1"))
    if year in {2024}:
        add_external("electron_trigger_low_pt_sf", (cat_info.get_file("egm", "electronHlt.json.gz"), "v1"))
    # electron energy correction and smearing
    add_external("electron_ss", (cat_info.get_file("egm", "electronSS_EtDependent.json.gz"), "v1"))  # FIXME correct for us? # noqa: E501

    # # top-tagging scale factors (TODO)
    # "toptag_sf": (f"{sources['jet']}/JMAR/???/???.json", "v1"),  # noqa

    # # V+jets reweighting
    # "vjets_reweighting": f"{sources['local_repo']}/data/json/vjets_reweighting.json",

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
            "pdf_weight*", "fsr_weight*", "isr_weight*",
            "muf_weight*", "mur_weight*", "murmuf_weight*", "murmuf_envelope*",
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

    logger.info_once(f"Config {cfg.name} finalized with {len(cfg.tags)} tags:\n{cfg.tags}.")

    return cfg
