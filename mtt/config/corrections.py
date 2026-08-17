# coding: utf-8

"""
Configuration of corrections for the m(ttbar) analysis.
"""

import order as od

from columnflow.util import DotDict
from columnflow.production.cms.btag import BTagSFConfig
from columnflow.production.cms.electron import ElectronSFConfig
from columnflow.production.cms.muon import MuonSFConfig


def vjets_reweighting_cfg(
) -> DotDict:

    kfactors = {
        "w": {
            "value": "wjets_kfactor_value",
            "error": "wjets_kfactor_error",
        },
        "z": {
            "value": "zjets_kfactor_value",
            "error": "zjets_kfactor_error",
        },
    }

    return DotDict.wrap(kfactors)


def jerc_cfg(
        campaign: od.Campaign,
        year: int = None,
) -> list[DotDict]:
    # https://cms-jerc.web.cern.ch/Recommendations/#jet-energy-scale

    jerc_postfix = campaign.x.postfix
    if jerc_postfix not in ("", "EE", "BPix"):
        raise ValueError(f"Invalid JERC postfix '{jerc_postfix}' for campaign {campaign.name}.")
    if year == 2022:
        jer_campaign = jec_campaign = f"Summer22{jerc_postfix}_22Sep2023"
    elif year == 2023:
        era = "Cv1234" if campaign.has_tag("preBPix") else "D"
        jer_campaign = f"Summer23{jerc_postfix}Prompt23_Run{era}"
        jec_campaign = f"Summer23{jerc_postfix}Prompt23"
    elif year == 2024:
        jec_campaign = "Summer24Prompt24"
        jer_campaign = "Summer23BPixPrompt23_RunD"  # no 2024 JER yet, use 2023 BPix: https://cms-jerc.web.cern.ch/Recommendations/#2024_1 # noqa

    jet_type = "AK4PFPuppi"
    fatjet_type = "AK8PFPuppi"
    jec_ak4_version = jec_ak8_version = {
        2022: "V3",
        2023: "V3",
        2024: "V2",
    }[year]

    jec_params = {
        "Jet": {
            "campaign": jec_campaign,
            "version": jec_ak4_version,
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
            "data_per_era": False if year in [2023, 2024] else True,  # 2022 JEC has the era in the correction set name
        },
        "FatJet": {
            "campaign": jec_campaign,
            "version": jec_ak8_version,
            "jet_type": fatjet_type,
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
            "data_per_era": False if year in [2023, 2024, 2025, 2026] else True,  # 2022 JEC has the era in the correction set name  # noqa
        },
        "SubJet": {
            "campaign": jec_campaign,
            "version": jec_ak4_version,
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
            "data_per_era": False if year in [2023, 2024, 2025, 2026] else True,  # 2022 JEC has the era in the correction set name  # noqa
        },
    }

    # JER
    # https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution?rev=107
    jer_params = {
        "Jet": {
            "campaign": jer_campaign,
            "version": {2022: "JRV1", 2023: "JRV1", 2024: "JRV1"}[year],
            "jet_type": jet_type,
        },
        "FatJet": {
            "campaign": jer_campaign,
            "version": {2022: "JRV1", 2023: "JRV1", 2024: "JRV1"}[year],
            "jet_type": fatjet_type,
        },
        "SubJet": {
            "campaign": jer_campaign,
            "version": {2022: "JRV1", 2023: "JRV1", 2024: "JRV1"}[year],
            "jet_type": jet_type,
        },
    }

    return [DotDict.wrap(jec_params), DotDict.wrap(jer_params)]


def btag_sf_cfg(
    config: od.Config,
    year: int = None,
) -> list[tuple, list]:
    name = ("deepJet_shape") if year != 2024 else ("UParTAK4_kinfit")
    discr = "btagPNetB" if year != 2024 else "btagUParTAK4B"
    jec_sources = [
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

    btag_uncs = {
        ## combined(?) uncertainties
        # uncertainties to b/c jets
        "down_bc": "bc_down",
        "up_bc": "bc_up",
        # uncertainties to light jets
        "down_light": "light_down",
        "up_light": "light_up",
        ## split uncertainties(?) (all needed?)
        # uncertainties to b/c jets
        "up_fsrdef_bc": "fsrdef_bc_up",
        "up_isrdef_bc": "isrdef_bc_up",
        "up_hdamp_bc": "hdamp_bc_up",
        "up_jer_bc": "jer_bc_up",
        "up_jes_bc": "jes_bc_up",
        "up_mass_bc": "mass_bc_up",
        "up_statistic_bc": "statistic_bc_up",
        "up_tune_bc": "tune_bc_up",
        "down_fsrdef_bc": "fsrdef_bc_down",
        "down_isrdef_bc": "isrdef_bc_down",
        "down_hdamp_bc": "hdamp_bc_down",
        "down_jer_bc": "jer_bc_down",
        "down_jes_bc": "jes_bc_down",
        "down_mass_bc": "mass_bc_down",
        "down_statistic_bc": "statistic_bc_down",
        "down_tune_bc": "tune_bc_down",
        # uncertainties to light jets
        "down_correlated_light": "correlated_light_down",
        "up_correlated_light": "correlated_light_up",
        "down_uncorrelated_light": "uncorrelated_light_down",
        "up_uncorrelated_light": "uncorrelated_light_up",
    }
    if year == 2024:
        # TODO: use shape based BTagSFConfig when available
        # currently, one fixed WP is available for b tagging SF in 2024
        # implementation from hbt analysis:
        # https://github.com/uhh-cms/hh2bbtautau/blob/4b2f1bc57a9c2ada18776e5ac6f0372269e1e26c/hbt/config/configs_hbt.py#L1410 # noqa
        from columnflow.selection.cms.btag import BTagWPCountConfig
        btag_wp_count_config = BTagWPCountConfig(
            jet_name="Jet",
            btag_column=discr,
            btag_wps=config.x.btag_wp.btagUParTAK4B,
            pt_edges=(0, 20, 30, 50, 70, 100, 140, 200, 300, 600, 10_000),
            # abs_eta_edges=(0.0, 1.0, 1.5, 2.0, 5.0),
            abs_eta_edges=(0.0, 1.5, 5.0),
        )

        from columnflow.production.cms.btag import BTagWPSFConfig

        def dataset_groups(dataset_inst: od.Dataset) -> list[od.Dataset]:
            # check which group the dataset belongs to
            for group_index in range(0, len(config.x.btag_wp_eff_groups)):
                group_tag = f"btag_wp_eff_group_{group_index}"
                if dataset_inst.has_tag(group_tag):
                    return [
                        _dataset_inst
                        for _dataset_inst in config.datasets
                        if _dataset_inst.has_tag(group_tag)
                    ]
            raise NotImplementedError(f"btag WP efficiency group not implemented for dataset {dataset_inst.name}")

        btag_wp_sf_config = BTagWPSFConfig(
            jet_name="Jet",
            btag_column=discr,
            correction_set="UParTAK4_merged",
            btag_wps=config.x.btag_wp.btagUParTAK4B,
            dataset_groups=dataset_groups,
            systs=btag_uncs,
            # further merge eta bins for sufficient statistics in each bin
            abs_eta_edges=(0.0, 5.0),
        )
    else:
        raise NotImplementedError("B-tagging SFs for 2022 and 2023 not implemented yet.")

    configs = {
        "btag_wp_count_config": btag_wp_count_config,
        "btag_wp_sf_config": btag_wp_sf_config,
    }

    return configs


def toptag_sf_cfg(
) -> DotDict:
    # TODO: use PNet!
    name = {
        "name": "DeepAK8_Top_MassDecorr",
        "wp": "1p0",
    }

    return DotDict.wrap(name)


def lepton_sf_cfg(
        config: od.Config,
        lepton: str = None,
) -> list:
    # TODO: we need to use different SFs for control regions
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    lepton_sf_dict = {
        3: {
            "2022preEE": {
                "electron": {
                    "id_sf_names": (
                        "Electron-ID-SF",
                        "2022Re-recoBCD",
                        "Tight",
                    ),
                },
                "muon": {
                    "sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2022preEE",
                    ),
                    "id_sf_names": (
                        "NUM_TightID_DEN_TrackerMuons",
                        "2022preEE",
                    ),
                    "iso_sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2022preEE",
                    ),
                },
            },
            "2022postEE": {
                "electron": {
                    "id_sf_names": (
                        "Electron-ID-SF",
                        "2022Re-recoE+PromptFG",
                        "Tight",
                    ),
                },
                "muon": {
                    "sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2022postEE",
                    ),
                    "id_sf_names": (
                        "NUM_TightID_DEN_TrackerMuons",
                        "2022postEE",
                    ),
                    "iso_sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2022postEE",
                    ),
                },
            },
            "2023preBPix": {
                "electron": {
                    "id_sf_names": (
                        "Electron-ID-SF",
                        "2023PromptC",
                        "Tight",
                    ),
                },
                "muon": {
                    "sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2023preBPix",
                    ),
                    "id_sf_names": (
                        "NUM_TightID_DEN_TrackerMuons",
                        "2023preBPix",
                    ),
                    "iso_sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2023preBPix",
                    ),
                },
            },
            "2023postBPix": {
                "electron": {
                    "id_sf_names": (
                        "Electron-ID-SF",
                        "2023PromptD",
                        "Tight",
                    ),
                },
                "muon": {
                    "sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2023postBPix",
                    ),
                    "id_sf_names": (
                        "NUM_TightID_DEN_TrackerMuons",
                        "2023postBPix",
                    ),
                    "iso_sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2023postBPix",
                    ),
                },
            },
            "2024": {
                "electron": {
                    "id_sf_names": (
                        "Electron-ID-SF",
                        "2024Prompt",
                        "Tight",
                    ),
                    "reco_sf_names": (
                        "Electron-ID-SF",
                        "2024Prompt",
                        ["RecoBelow20", "Reco20to75", "RecoAbove75"],
                    ),
                },
                "muon": {
                    "sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2024Prompt24",
                    ),
                    "id_sf_names": (
                        "NUM_TightID_DEN_TrackerMuons",
                        "2024Prompt24",
                    ),
                    "iso_sf_names": (
                        "NUM_TightPFIso_DEN_TightID",
                        "2024Prompt24",
                    ),
                },
            },
        },
    }

    if lepton == "electron":
        electron_id_iso_sf_config = ElectronSFConfig(
            correction=lepton_sf_dict[run][tag][lepton]["reco_sf_names"][0],
            campaign=lepton_sf_dict[run][tag][lepton]["reco_sf_names"][1],
            working_point={
                "wp80iso": (lambda variables: variables["pt"] > 10),
            },
        )
        electron_reco_sf_config = ElectronSFConfig(
            correction=lepton_sf_dict[run][tag][lepton]["reco_sf_names"][0],
            campaign=lepton_sf_dict[run][tag][lepton]["reco_sf_names"][1],
            working_point={
                "RecoBelow20": (lambda variables: variables["pt"] < 20),
                "Reco20to75": (lambda variables: (variables["pt"] >= 20) & (variables["pt"] < 75.0)),
                "RecoAbove75": (lambda variables: variables["pt"] >= 75.0),
            },
        )
        configs = {
            "electron_id_iso_sf_config": electron_id_iso_sf_config,
            "electron_reco_sf_config": electron_reco_sf_config,
        }
        return configs

    elif lepton == "muon":
        muon_id_config = MuonSFConfig(
            correction=lepton_sf_dict[run][tag][lepton]["id_sf_names"][0],
        )
        muon_iso_config = MuonSFConfig(
            correction=lepton_sf_dict[run][tag][lepton]["iso_sf_names"][0],
        )
        configs = {
            "muon_id_config": muon_id_config,
            "muon_iso_config": muon_iso_config,
        }
        return configs


def met_phi_cfg(
    config: od.Config,
):
    met_column = config.x.met_selection.column
    # raw_met_column = config.x.met_selection.raw_column

    from columnflow.calibration.cms.met import METPhiConfig
    met_config = METPhiConfig(
        met_name=met_column,
        met_type=met_column,
        correction_set="met_xy_corrections",
        keep_uncorrected=True,  # TODO do we need this?
        pt_phi_variations={
            "stat_xdn": "metphi_statx_down",
            "stat_xup": "metphi_statx_up",
            "stat_ydn": "metphi_staty_down",
            "stat_yup": "metphi_staty_up",
        },
        variations={
            "pu_dn": "minbias_xs_down",
            "pu_up": "minbias_xs_up",
        },
    )
    return met_config


def jet_id_cfg():
    from columnflow.production.cms.jet import JetIdConfig
    jet_id_config = JetIdConfig(
        corrections={"AK4PUPPI_Tight": 2, "AK4PUPPI_TightLeptonVeto": 3},
    )
    fatjet_id_config = JetIdConfig(
        corrections={"AK8PUPPI_Tight": 2, "AK8PUPPI_TightLeptonVeto": 3},
    )

    return DotDict.wrap({
        "Jet": jet_id_config,
        "FatJet": fatjet_id_config,
    })
