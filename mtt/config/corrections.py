# coding: utf-8

"""
Configuration of corrections for the m(ttbar) analysis.
"""

import order as od
import law

from columnflow.util import DotDict
from columnflow.selection.cms.btag import BTagWPCountConfig
from columnflow.production.cms.btag import BTagSFConfig, BTagWPSFConfig

logger = law.logger.get_logger(__name__)


def vjets_reweighting_cfg(
) -> DotDict:

    kfactors = {
        "w": {
            # "value": "wjets_kfactor_value",
            "value": "NOT_YET_AVAILABLE",
            "error": "wjets_kfactor_error",
        },
        "z": {
            # "value": "zjets_kfactor_value",
            "value": "NOT_YET_AVAILABLE",
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
        jer_campaign = "Summer24Prompt24"  # no 2024 JER yet, use 2023 BPix: https://cms-jerc.web.cern.ch/Recommendations/#2024_1 # noqa
    elif year == 2025:
        jec_campaign = "Summer24Prompt25"
        jer_campaign = "Summer24Prompt25"

    jet_type = "AK4PFPuppi"
    fatjet_type = "AK8PFPuppi"
    jec_ak4_version = jec_ak8_version = {
        2022: "V3",
        2023: "V3",
        2024: "V5",
        2025: "V2",
        2026: "V1",
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
            "data_per_era": False if year in [2023, 2024, 2025, 2026] else True,  # 2022 JEC has the era in the correction set name  # noqa
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
            "version": {2022: "JRV1", 2023: "JRV1", 2024: "JRV2", 2025: "JRV2"}[year],
            "jet_type": jet_type,
        },
        "FatJet": {
            "campaign": jer_campaign,
            "version": {2022: "JRV1", 2023: "JRV1", 2024: "JRV2", 2025: "JRV2"}[year],
            "jet_type": fatjet_type,
        },
        "SubJet": {
            "campaign": jer_campaign,
            "version": {2022: "JRV1", 2023: "JRV1", 2024: "JRV2", 2025: "JRV2"}[year],
            "jet_type": jet_type,
        },
    }

    return [DotDict.wrap(jec_params), DotDict.wrap(jer_params)]


def btag_sf_setup(
    config: od.Config,
    year: int = None,
) -> list[tuple, list]:
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

    # b tagging SF configuration
    discr = config.x.jet_selection.ak4.btagger.column

    # b/c tagging uncertainties from config
    bc_uncs_list = config.x.btag_uncs_bc
    light_uncs_list = config.x.btag_uncs_light

    # build dict from these lists:
    # key: "up/down_<uncertainty_name>" (clib name), value: "<uncertainty_name>_up/_down" (column name)

    btag_uncs = {}
    for unc in bc_uncs_list:
        btag_uncs[f"up_{unc}_bc"] = f"{unc}_bc_up"
        btag_uncs[f"down_{unc}_bc"] = f"{unc}_bc_down"
    for unc in light_uncs_list:
        btag_uncs[f"up_{unc}_light"] = f"{unc}_light_up"
        btag_uncs[f"down_{unc}_light"] = f"{unc}_light_down"

    btag_uncs["up_bc"] = "bc_up"
    btag_uncs["down_bc"] = "bc_down"
    btag_uncs["up_light"] = "light_up"
    btag_uncs["down_light"] = "light_down"

    if year in [2024, 2025, 2026]:
        config.add_tag("skip_btag_weights")
        logger.debug("Setting up fixed WP based btag SFs for 2024, as shape based SFs are not yet available."
        " Please switch to shape based SFs as soon as they are available.")
        # NOTE: switch to shape based SF also for 2024 as soon as they are available; placeholder for now
        config.x.btag_sf = BTagSFConfig(
            correction_set="DO_NOT_USE",
            jec_sources=jec_sources,
            discriminator=discr,
        )
        # implementation from hbt analysis:
        # https://github.com/uhh-cms/hh2bbtautau/blob/4b2f1bc57a9c2ada18776e5ac6f0372269e1e26c/hbt/config/configs_hbt.py#L1410 # noqa
        # set up btag WP histogram to be stored in SelectEvent step
        config.x.btag_wp_count_config = BTagWPCountConfig(
            jet_name="Jet",
            btag_column=discr,
            # all five wps
            btag_wps=config.x.btag_wp.btagUParTAK4B.fixed_wp,
            # fine pt binning, can be merged later for sufficient statistics in each bin
            pt_edges=(0, 20, 30, 50, 70, 100, 140, 200, 300, 600, 10_000),
            # abs_eta_edges=(0.0, 1.0, 1.5, 2.0, 5.0),
            abs_eta_edges=(0.0, 1.5, 5),
        )

        def dataset_groups(dataset_inst: od.Dataset) -> list[od.Dataset]:
            # check which efficiency group the dataset belongs to
            for group_index in range(0, len(config.x.btag_wp_eff_groups)):
                group_tag = f"btag_wp_eff_group_{group_index}"
                if dataset_inst.has_tag(group_tag):
                    return [
                        _dataset_inst
                        for _dataset_inst in config.datasets
                        if _dataset_inst.has_tag(group_tag)
                    ]
            raise NotImplementedError(f"btag WP efficiency group not implemented for dataset {dataset_inst.name}")

        # set up btag WP SF binning and systematic variantions to be stored in ProduceColumns step
        config.x.btag_wp_sf_config = BTagWPSFConfig(
            jet_name="Jet",
            btag_column=discr,
            correction_set="UParTAK4_merged",
            btag_wps=config.x.btag_wp.btagUParTAK4B.fixed_wp,
            dataset_groups=dataset_groups,
            systs=btag_uncs,
            # further merge eta bins for sufficient statistics in each bin
            abs_eta_edges=(0.0, 1.5, 5.0),
            wp_merging={
                # remove xxtight for better stats
                "loose": ["loose"],
                "medium": ["medium"],
                "tight": ["tight"],
                "xtight": ["xtight"],
                "xxtight": ["xxtight"],
            },
            pt_edges=(0, 20, 30, 50, 70, 100, 140, 200, 300, 600, 10_000) if not config.has_tag("is_limited") else (0, 10_000),  # no pt binning for testing with limited files # noqa
        )
    else:
        config.add_tag("skip_btag_wp_weights")  # skip fixed WP based btag weights for 2022/2023, apply shape based SF
        logger.debug("Setting up shape based btag SFs for 2022/2023.")
        logger.warning_once("Evaluate used processes for normalized btag SFs for 2022/2023, set to 'tt'+'st' for now.")
        config.x.btag_sf = BTagSFConfig(
            correction_set="deepJet_shape",
            jec_sources=jec_sources,
            discriminator=discr,
        )
        config.x.btag_wp_count_config = BTagWPCountConfig(
            jet_name="Dummy",
        )
        config.x.btag_wp_sf_config = BTagWPSFConfig(
            jet_name="Dummy",
        )

    return jec_sources


def toptag_sf_cfg(
) -> DotDict:
    logger.warning_once("Top-tagging SFs are not yet implemented.")
    # TODO: use PNet!
    # name = {
    #     "name": "DeepAK8_Top_MassDecorr",
    #     "wp": "1p0",
    # }

    # return DotDict.wrap(name)


def met_phi_cfg(
    config: od.Config,
    year: int = None,
):
    met_column = config.x.met_selection.column
    # raw_met_column = config.x.met_selection.raw_column

    from columnflow.calibration.cms.met import METPhiConfig
    if year in [2024, 2025, 2026]:
        correction_set = "NOT_YET_AVAILABLE"
    else:
        correction_set = "met_xy_corrections"
    met_config = METPhiConfig(
        met_name=met_column,
        met_type=met_column,
        correction_set=correction_set,
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
