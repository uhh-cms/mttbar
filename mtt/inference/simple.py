# coding: utf-8

"""
mttbar inference model
"""
from __future__ import annotations

import law

from columnflow.inference import inference_model, ParameterType, ParameterTransformation, InferenceModel, FlowStrategy

logger = law.logger.get_logger(__name__)


@inference_model
def simple(
    self: InferenceModel,
) -> None:
    """
    Simple inference model intended as a baseline.
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
        # signal regions
        ("sr_1e_0t", "1e__0t__chi2pass"),
        ("sr_1e_1t", "1e__1t__chi2pass"),
        ("sr_1m_0t", "1m__0t__chi2pass"),
        ("sr_1m_1t", "1m__1t__chi2pass"),
        # control regions
        ("cr_1e_0t", "1e__0t__chi2fail"),
        ("cr_1e_1t", "1e__1t__chi2fail"),
        ("cr_1m_0t", "1m__0t__chi2fail"),
        ("cr_1m_1t", "1m__1t__chi2fail"),
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
        "zprime_tt_m1000_w100",

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
                )
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
        # TODO: why not -1?
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
