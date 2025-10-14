# coding: utf-8

"""
Configuration of corrections for the m(ttbar) analysis.
"""

import order as od

from columnflow.util import DotDict


def vjets_reweighting(
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


def jerc(
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
        jec_campaign = f"Summer24Prompt24"
        jer_campaign = f"Summer23BPixPrompt23_RunD"  # no 2024 JER yet, use 2023 BPix

    jet_type = "AK4PFPuppi"
    fatjet_type = "AK8PFPuppi"
    jec_ak4_version = jec_ak8_version = {
        2022: "V3",
        2023: "V3",
        2024: "V1",
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
            "data_per_era": False if year == 2023 else True,
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
            "data_per_era": False if year == 2023 else True,
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
            "data_per_era": False if year == 2023 else True,
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


def btag_sf(
        year: int = None,
) -> list[tuple, list]:
    name = ("deepJet_shape")
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

    return (name, jec_sources)


def toptag_sf(
) -> DotDict:
    # TODO: use PNet!
    name = {
        "name": "DeepAK8_Top_MassDecorr",
        "wp": "1p0",
    }

    return DotDict.wrap(name)


def lepton_sf(
        config: od.Config,
        lepton: str = None,
) -> list:
    # TODO: we need to use different SFs for control regions
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    lepton_sf = {
        3: {
            "2022preEE": {
                "electron": {
                    "sf_names": (
                        "Electron-ID-SF",
                        "2022Re-recoBCD",
                        "Tight",
                    )
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
                }
            },
            "2022postEE": {
                "electron": {
                    "sf_names": (
                        "Electron-ID-SF",
                        "2022Re-recoE+PromptFG",
                        "Tight",
                    )
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
                    "sf_names": (
                        "Electron-ID-SF",
                        "2023PromptC",
                        "Tight",
                    )
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
                    "sf_names": (
                        "Electron-ID-SF",
                        "2023PromptD",
                        "Tight",
                    )
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
                    "sf_names": (
                        "Electron-ID-SF",
                        "2024Prompt24",
                        "Tight",
                    )
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
        }
    }

    return DotDict.wrap(
        lepton_sf[run][tag][lepton]
    )
