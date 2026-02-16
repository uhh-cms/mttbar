# coding: utf-8

"""
mttbar inference model
"""
from __future__ import annotations

import law

from columnflow.inference import inference_model, ParameterType, ParameterTransformation, InferenceModel, FlowStrategy

logger = law.logger.get_logger(__name__)


@inference_model
def an_2019_197(
    self: InferenceModel,
) -> None:
    """
    Inference model intended to reproduce the fits in CMS-AN-2019-197.
    """
    logger.warning(f"Requested inference model {self.cls_name} is untested. It only works for a single config.")
    config_inst = self.config_insts[0]

    year = config_inst.campaign.x.year  # noqa; not used right now

    #
    # regions/categories
    #

    # tuples of inference categories and
    # corresponding config category names
    categories = [
        # signal regions (TODO: add DNN)
        ("1e_sr_bin1_0t", "1e__0t__chi2pass__acts_0_5"),
        ("1e_sr_bin1_1t", "1e__1t__chi2pass__acts_0_5"),
        ("1e_sr_bin2_0t", "1e__0t__chi2pass__acts_5_7"),
        ("1e_sr_bin2_1t", "1e__1t__chi2pass__acts_5_7"),
        ("1e_sr_bin3", "1e__chi2pass__acts_7_9"),
        ("1e_sr_bin4", "1e__chi2pass__acts_9_1"),
        ("1m_sr_bin1_0t", "1m__0t__chi2pass__acts_0_5"),
        ("1m_sr_bin1_1t", "1m__1t__chi2pass__acts_0_5"),
        ("1m_sr_bin2_0t", "1m__0t__chi2pass__acts_5_7"),
        ("1m_sr_bin2_1t", "1m__1t__chi2pass__acts_5_7"),
        ("1m_sr_bin3", "1m__chi2pass__acts_7_9"),
        ("1m_sr_bin4", "1m__chi2pass__acts_9_1"),
        # control regions (TODO)
        # ("1m_cr_st", "1m__dnn_st"),
        # ("1m_cr_wjets", "1m__dnn_w_lnu"),
        # ("1e_cr_st", "1e__dnn_st"),
        # ("1e_cr_wjets", "1e__dnn_w_lnu"),
    ]

    # add categories to inference model
    for inference_cat, config_cat in categories:
        self.add_category(
            name=inference_cat,
            config_data={
                config_inst.name: self.category_config_spec(
                    category=config_cat,
                    variable="ttbar_mass",
                    data_datasets=[],
                )
            },
            mc_stats=True,
            flow_strategy=FlowStrategy.warn,  # FIXME look for options!
            rate_precision=5,  # FIXME default for now
            data_from_processes=[
                "tt",
            ],
        )

    #
    # processes
    #

    processes = [
        # signals (TODO: add others)
        # "zprime_tt_m400_w40",
        # "zprime_tt_m500_w50",
        # "zprime_tt_m600_w60",
        # "zprime_tt_m700_w70",
        # "zprime_tt_m800_w80",
        # "zprime_tt_m900_w90",
        "zprime_tt_m1000_w100",
        # "zprime_tt_m1200_w120",
        # "zprime_tt_m1400_w140",
        # "zprime_tt_m1600_w160",
        # "zprime_tt_m1800_w180",
        # "zprime_tt_m2000_w200",
        # "zprime_tt_m2500_w250",
        # "zprime_tt_m3000_w300",
        # "zprime_tt_m3500_w350",
        # "zprime_tt_m4000_w400",
        # "zprime_tt_m4500_w450",
        # "zprime_tt_m5000_w500",
        # "zprime_tt_m6000_w600",
        # "zprime_tt_m7000_w700",
        # "zprime_tt_m8000_w800",
        # "zprime_tt_m9000_w900",

        # backgrounds (TODO: add others)
        "tt",
        # "st",
        # "dy_lep",
        # "w_lnu",
        # "qcd",
        # "vv",
    ]

    # different naming convention for some processes
    inference_processes = {
        "w_lnu": "wjets",
    }

    for proc in processes:

        # raise if process not defined in config
        if not self.config_inst.has_process(proc):
            raise ValueError(
                f"Process {proc} requested for inference, but is not "
                f"present in the config {self.config_inst.name}.",
            )

        # determine datasets for each process
        process_insts = [
            p for p, _, _ in self.config_inst.get_process(proc).walk_processes(include_self=True)
        ]
        datasets = [
            dataset_inst.name for dataset_inst in self.config_inst.datasets
            if any(map(dataset_inst.has_process, process_insts))
        ]

        # add process to inference model
        self.add_process(
            name=inference_processes.get(proc, proc),
            config_data={
                config_inst.name: self.process_config_spec(
                    process=proc,
                    mc_datasets=datasets,
                ),
            },
            is_signal=proc.startswith("zprime_tt"),
        )

    #
    # parameters
    #

    # lumi
    lumi = self.config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    # process rates
    for proc, rate in [
        ("tt", 1.2),
        # ("st", 1.3),
        # ("wjets", 1.5),
    ]:
        self.add_parameter(
            f"{proc}_rate",
            type=ParameterType.rate_gauss,
        )

    # systematic shifts (TODO: add others)
    uncertainty_shifts = [
        # "pdf",
        # "mcscale",
        # "prefiring",
        "minbias_xs",  # pileup

        # "muon",  # TODO: split?
        # "mu_id",
        # "mu_iso",
        # "mu_reco",
        # "mu_trigger",

        # b-tagging
        # "btag_cferr1",
        # "btag_cferr2",
        # "btag_hf",
        # "btag_hfstats1_2017",
        # "btag_hfstats2_2017",
        # "btag_lf",
        # "btag_lfstats1_2017",
        # "btag_lfstats2_2017",
    ]

    # different naming convention for some parameters
    inference_pars = {
        "minbias_xs": "pu",
    }

    param_kwargs = {}

    for proc in processes:
        for unc in uncertainty_shifts:
            par = inference_pars.get(unc, unc)
            param_kwargs["config_data"] = {
                config_inst.name: self.parameter_config_spec(
                    shift_source=unc,
                )
            }
            self.add_parameter(
                f"{par}_{proc}",
                process=inference_processes.get(proc, proc),
                type=ParameterType.shape,
                **param_kwargs,
            )

    self.cleanup()


@inference_model
def an_v12_simplified(
    self: InferenceModel,
) -> None:
    """
    Inference model intended to reproduce the fits in CMS-AN-2019-197.
    """
    logger.warning(f"Requested inference model {self.cls_name} only works for a single config.")
    config_inst = self.config_insts[0]
    signal_mass = getattr(self, "signal_mass", None)
    signal_width = getattr(self, "signal_width", None)

    # Debug logging
    logger.debug(f"Model name: {self.cls_name}")
    logger.debug(f"signal_mass = {signal_mass}")
    logger.debug(f"signal_width = {signal_width}")
    logger.debug(f"Available attributes: {[attr for attr in dir(self) if not attr.startswith('_')]}")

    # year = config_inst.campaign.x.year  # noqa; not used right now

    #
    # regions/categories
    #

    # tuples of inference categories and
    # corresponding config category names
    categories = [
        # signal regions (simplified)
        ("el_sr1", "1e__1t__chi2pass__dnn_tt"),
        ("mu_sr1", "1m__1t__chi2pass__dnn_tt"),
        ("el_sr2", "1e__0t__chi2pass__dnn_tt"),
        ("mu_sr2", "1m__0t__chi2pass__dnn_tt"),
        # control regions
        ("el_cr1", "1e__dnn_st"),
        ("mu_cr1", "1m__dnn_st"),
        ("el_cr2", "1e__dnn_other"),
        ("mu_cr2", "1m__dnn_other"),
    ]

    # add categories to inference model
    for inference_cat, config_cat in categories:
        self.add_category(
            name=inference_cat,
            config_data={
                config_inst.name: self.category_config_spec(
                    category=config_cat,
                    variable="ttbar_mass_ext",
                    data_datasets=["data_*"],  # Should this be empty for now?
                )
            },
            mc_stats=True,
            flow_strategy=FlowStrategy.warn,  # FIXME look for options!
            rate_precision=5,  # FIXME default for now
            data_from_processes=[
                "tt",
                "st",
                "dy",
                "w_lnu",
                "qcd",
                # "vv",
            ],
        )

    #
    # processes
    #

    background_processes = [
        "tt",
        "st",
        "dy",
        "w_lnu",
        "qcd",
        # "vv",
    ]

    processes = background_processes.copy()
    if signal_mass is not None and signal_width is not None:
        signal_process = f"zprime_tt_m{signal_mass}_w{signal_width}"
        logger.info(f"Creating signal+background model with {signal_process}")

        # Check if signal process exists in config
        if config_inst.has_process(signal_process):
            processes.append(signal_process)
            logger.debug(f"Added signal process: {signal_process}")
        else:
            logger.warning(f"Signal process {signal_process} not found in config - using background-only")
    else:
        logger.info("Creating background-only model (no signal specified)")

    logger.debug(f"Final process list: {processes}")

    # different naming convention for some processes
    # FIXME: make renaming work with fake data
    #        currently doesn't recognize name change when calculating 'observation' in datacard
    inference_processes = {
        # "w_lnu": "wjets",
    }

    for proc in processes:
        logger.debug(f"Adding process {proc} to inference model.")

        # raise if process not defined in config
        if not config_inst.has_process(proc):
            raise ValueError(
                f"Process {proc} requested for inference, but is not "
                f"present in the config {config_inst.name}.",
            )

        # determine datasets for each process
        process_insts = [
            p for p, _, _ in config_inst.get_process(proc).walk_processes(include_self=True)
        ]
        datasets = [
            dataset_inst.name for dataset_inst in config_inst.datasets
            if any(map(dataset_inst.has_process, process_insts))
        ]

        # add process to inference model
        self.add_process(
            name=inference_processes.get(proc, proc),
            config_data={
                config_inst.name: self.process_config_spec(
                    process=proc,
                    mc_datasets=datasets,
                ),
            },
            is_signal=proc.startswith("zprime_tt"),
        )

    #
    # parameters
    #

    # lumi
    lumi = config_inst.x.luminosity
    for unc_name in lumi.uncertainties:
        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
            transformations=[ParameterTransformation.symmetrize],
        )

    # process rates
    for proc, rate in [
        ("tt", 1.2),
        ("st", 1.5),
        ("w_lnu", 1.5),
        ("dy", 1.5),
        ("qcd", 1.5),
        # ("vv", 1.5),
    ]:
        self.add_parameter(
            f"xsec_{proc}",
            type=ParameterType.rate_gauss,
            effect=rate,
            process=inference_processes.get(proc, proc),
        )

    # systematic shifts (TODO: add others)
    uncertainty_shifts = [
        # "pdf",
        # "mcscale",
        # "prefiring",
        "minbias_xs",  # pileup

        # "muon",  # TODO: split?
        # "mu_id",
        # "mu_iso",
        # "mu_reco",
        # "mu_trigger",

        # b-tagging
        # "btag_cferr1",
        # "btag_cferr2",
        # "btag_hf",
        # "btag_hfstats1_2017",
        # "btag_hfstats2_2017",
        # "btag_lf",
        # "btag_lfstats1_2017",
        # "btag_lfstats2_2017",
    ]

    # different naming convention for some parameters
    inference_pars = {
        "minbias_xs": "pu",
    }

    param_kwargs = {}

    for proc in processes:
        for unc in uncertainty_shifts:
            par = inference_pars.get(unc, unc)
            param_kwargs["config_data"] = {
                config_inst.name: self.parameter_config_spec(
                    shift_source=unc,
                )
            }
            self.add_parameter(
                f"{par}",
                process=inference_processes.get(proc, proc),
                type=ParameterType.shape,
                **param_kwargs,
            )

    self.cleanup()


# Z' models
# 1% width samples
an_v12_simplified__m500_w5 = an_v12_simplified.derive(
    "an_v12_simplified__m500_w5",
    cls_dict={
        "signal_mass": 500,
        "signal_width": 5,
    }
)

an_v12_simplified__m4000_w40 = an_v12_simplified.derive(
    "an_v12_simplified__m4000_w40",
    cls_dict={
        "signal_mass": 4000,
        "signal_width": 40,
    }
)

an_v12_simplified__m4500_w45 = an_v12_simplified.derive(
    "an_v12_simplified__m4500_w45",
    cls_dict={
        "signal_mass": 4500,
        "signal_width": 45,
    }
)

an_v12_simplified__m7000_w70 = an_v12_simplified.derive(
    "an_v12_simplified__m7000_w70",
    cls_dict={
        "signal_mass": 7000,
        "signal_width": 70,
    }
)

# 10% width samples
an_v12_simplified__m2000_w200 = an_v12_simplified.derive(
    "an_v12_simplified__m2000_w200",
    cls_dict={
        "signal_mass": 2000,
        "signal_width": 200,
    }
)
an_v12_simplified__m8000_w800 = an_v12_simplified.derive(
    "an_v12_simplified__m8000_w800",
    cls_dict={
        "signal_mass": 8000,
        "signal_width": 800,
    }
)

# 30% width samples
an_v12_simplified__m5000_w1500 = an_v12_simplified.derive(
    "an_v12_simplified__m5000_w1500",
    cls_dict={
        "signal_mass": 5000,
        "signal_width": 1500,
    }
)
