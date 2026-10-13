# coding: utf-8

"""
mttbar inference model
NOTE: Not maintained at the moment, but kept as reference. Use zprime model instead.
"""
from __future__ import annotations

import law
import order as od

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
                ),
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
                ),
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
    Inference model intended to reproduce the fits in CMS-AN-2019-197, extended to combine
    multiple configs (e.g. run periods/eras). No binning in cos(theta*) included atm.
    """
    logger.warning("Not fully tested. Kept as reference. Use zprime model instead.")
    config_insts = self.config_insts
    signal_mass = getattr(self, "signal_mass", None)
    signal_width = getattr(self, "signal_width", None)
    # toggle: True merges all configs into one bin per category (fewer bins/nuisances, faster
    # fit, but per-era rate-type parameters like luminosity can no longer be scoped to just their
    # own era's share of a bin and are applied as an approximation to the whole merged rate - see
    # the lumi block below). False (default) keeps the physically correct per-era split.
    merge_eras = getattr(self, "merge_eras", False)

    background_processes = [
        "tt",
        "st",
        "dy",
        "w_lnu",
        "qcd",
        "vv",
    ]

    fit_var = "ttbar_mass_ext"

    # short, 2-digit era suffix per config (e.g. "24", "25"), used for shorter category/parameter names
    def era_suffix(config_inst: od.Config) -> str:
        return str(config_inst.campaign.x.year)[-2:]

    # Debug logging
    logger.debug(f"Model name: {self.cls_name}")
    logger.debug(f"configs: {[c.name for c in config_insts]}")
    logger.debug(f"signal_mass = {signal_mass}")
    logger.debug(f"signal_width = {signal_width}")
    logger.debug(f"merge_eras = {merge_eras}")
    logger.debug(f"Available attributes: {[attr for attr in dir(self) if not attr.startswith('_')]}")

    #
    # regions/categories
    #

    # tuples of inference categories and
    # corresponding config category names
    # (assumes the same category naming scheme is used across all configs; if a config uses
    # different names for the same logical region, replace this with a dict keyed by config name)
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

    if merge_eras:
        # single bin per region, spanning all configs that have it - matches the pre-era-split
        # behavior; see the lumi block below for the resulting caveat on per-era rate parameters
        for inference_cat, config_cat in categories:
            config_data = {}
            for config_inst in config_insts:
                if not config_inst.has_category(config_cat):
                    logger.warning(
                        f"config '{config_inst.name}' does not have category '{config_cat}', skipping it for "
                        f"inference category '{inference_cat}'",
                    )
                    continue
                config_data[config_inst.name] = self.category_config_spec(
                    category=config_cat,
                    variable=fit_var,
                    data_datasets=["data_*"],
                )

            if not config_data:
                raise ValueError(
                    f"none of the configs {[c.name for c in config_insts]} have category '{config_cat}' "
                    f"required by inference category '{inference_cat}'",
                )

            self.add_category(
                name=inference_cat,
                config_data=config_data,
                mc_stats=True,
                flow_strategy=FlowStrategy.warn,  # FIXME look for options!
                rate_precision=5,  # FIXME default for now
                data_from_processes=background_processes,  # NOTE: blinded datacards for now
            )
    else:
        # add categories to inference model
        # NOTE: one inference category is created PER ERA (config), rather than merging all
        # configs into a single category. This is required because DatacardWriter sums all
        # configs listed in a category's config_data into ONE combined rate per category/process;
        # rate-type (lnN) parameters are then applied to that whole merged rate with no
        # per-config scoping mechanism. Splitting by era keeps each bin's yield tied to exactly
        # one config, so a per-era luminosity (or any other per-era rate) uncertainty only ever
        # touches the bin it should.
        for inference_cat, config_cat in categories:
            for config_inst in config_insts:
                if not config_inst.has_category(config_cat):
                    logger.warning(
                        f"config '{config_inst.name}' does not have category '{config_cat}', skipping "
                        f"inference category '{inference_cat}_{era_suffix(config_inst)}'",
                    )
                    continue

                self.add_category(
                    name=f"{inference_cat}_{era_suffix(config_inst)}",
                    config_data={
                        config_inst.name: self.category_config_spec(
                            category=config_cat,
                            variable=fit_var,
                            data_datasets=["data_*"],
                        ),
                    },
                    mc_stats=True,
                    flow_strategy=FlowStrategy.warn,  # FIXME look for options!
                    rate_precision=5,  # FIXME default for now
                    data_from_processes=background_processes,  # NOTE: blinded datacards for now
                )

    #
    # processes
    #

    processes = background_processes.copy()
    if signal_mass is not None and signal_width is not None:
        signal_process = f"zprime_tt_m{signal_mass}_w{signal_width}"
        logger.info(f"Creating signal+background model with {signal_process}")

        # signal must exist in at least one config to be added; per-config availability is
        # handled below when building each process' config_data
        if any(config_inst.has_process(signal_process) for config_inst in config_insts):
            processes.append(signal_process)
            logger.debug(f"Added signal process: {signal_process}")
        else:
            logger.warning(
                f"Signal process {signal_process} not found in any config - using background-only",
            )
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

        config_data = {}
        for config_inst in config_insts:
            if not config_inst.has_process(proc):
                # signal processes are allowed to be missing from individual configs (e.g. only
                # produced for one era); backgrounds are expected everywhere
                if proc in background_processes:
                    raise ValueError(
                        f"Process {proc} requested for inference, but is not present in the config "
                        f"{config_inst.name}.",
                    )
                logger.warning(f"config '{config_inst.name}' does not have process '{proc}', skipping it")
                continue

            # determine datasets for each process, per config
            process_insts = [
                p for p, _, _ in config_inst.get_process(proc).walk_processes(include_self=True)
            ]
            datasets = [
                dataset_inst.name for dataset_inst in config_inst.datasets
                if any(map(dataset_inst.has_process, process_insts))
            ]
            config_data[config_inst.name] = self.process_config_spec(
                process=proc,
                mc_datasets=datasets,
            )

        # add process to inference model
        self.add_process(
            name=inference_processes.get(proc, proc),
            config_data=config_data,
            is_signal=proc.startswith("zprime_tt"),
        )

    #
    # parameters
    #

    # lumi
    # each luminosity uncertainty name is already unique per era (e.g. "lumi_13p6TeV_2024" vs
    # "lumi_13p6TeV_2025" - no name is shared across configs), so there's no cross-era merging to
    # do here regardless of `merge_eras`.
    if merge_eras:
        # CAVEAT: with categories merged, a category's rate is the SUM of both eras' yields, and
        # add_parameter/DatacardWriter apply a rate_gauss effect to that whole merged rate - there
        # is no way to scope it to just the fraction of the bin coming from one era. This block
        # therefore applies each era's uncertainty to the FULL merged rate, which is an
        # approximation (mildly conservative, since it treats the whole bin as if it all came
        # from that one era) rather than the physically exact per-era treatment. Fine for a quick
        # convergence check; revisit before using merge_eras=True for a final result.
        logger.warning(
            "merge_eras is enabled: per-era luminosity uncertainties are applied to the full "
            "merged category rate as an approximation, not scoped to their own era's yield share",
        )
        for config_inst in config_insts:
            lumi = config_inst.x.luminosity
            for unc_name in lumi.uncertainties:
                self.add_parameter(
                    unc_name,
                    type=ParameterType.rate_gauss,
                    effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
                    transformations=[ParameterTransformation.symmetrize],
                )
    else:
        # scoping each parameter to only the categories belonging to its own era via `category=`
        # matters here, since add_parameter/DatacardWriter apply a rate_gauss effect directly to
        # whatever (category, process) rows match - with categories split per era, a plain
        # wildcard match would otherwise attach e.g. lumi_13p6TeV_2024 to the 2025 categories too
        # and double up the uncertainty
        for config_inst in config_insts:
            lumi = config_inst.x.luminosity
            for unc_name in lumi.uncertainties:
                self.add_parameter(
                    unc_name,
                    type=ParameterType.rate_gauss,
                    effect=lumi.get(names=unc_name, direction=("down", "up"), factor=True),
                    transformations=[ParameterTransformation.symmetrize],
                    category=f"*_{era_suffix(config_inst)}",
                )

    # process rates
    for proc, rate in [
        ("tt", 1.2),
        ("st", 1.5),
        ("w_lnu", 1.5),
        ("dy", 1.5),
        ("qcd", 1.5),
        ("vv", 1.5),
    ]:
        self.add_parameter(
            f"xsec_{proc}",
            type=ParameterType.rate_gauss,
            effect=rate,
            process=inference_processes.get(proc, proc),
        )

    # systematic shifts (TODO: add others)
    uncertainty_shifts = [
        "minbias_xs",  # pileup

        # highest impacts in ANv12, page 222
        # TODO: what uncs do we have?
        # "top_tagging",  # NOTE: not yet implemented
        # "btag_cferr1",
        # "pdf",
        # "mur",
        # "btag_lf",
        # "muf",
        # "btag_cferr2",
        # "scale_j",
        # "btag_hf",
    ]

    # different naming convention for some parameters
    inference_pars = {
        "minbias_xs": "pu",
    }

    for proc in processes:
        for unc in uncertainty_shifts:
            par = inference_pars.get(unc, unc)

            if merge_eras:
                # one parameter per (proc, unc), spanning all configs that have this shift - since
                # categories share plain names in merged mode (e.g. "el_sr1" for both eras),
                # calling add_parameter once per config with the same name/process/category would
                # collide on the second call
                config_data = {}
                for config_inst in config_insts:
                    if not config_inst.has_shift(f"{unc}_up") and not config_inst.has_shift(f"{unc}_down"):
                        logger.warning(
                            f"config '{config_inst.name}' does not have shift source '{unc}', skipping it "
                            f"for parameter '{par}'",
                        )
                        continue
                    config_data[config_inst.name] = self.parameter_config_spec(
                        shift_source=unc,
                    )

                if not config_data:
                    continue

                self.add_parameter(
                    f"{par}",
                    process=inference_processes.get(proc, proc),
                    type=ParameterType.shape,
                    config_data=config_data,
                )
            else:
                for config_inst in config_insts:
                    if not config_inst.has_shift(f"{unc}_up") and not config_inst.has_shift(f"{unc}_down"):
                        logger.warning(
                            f"config '{config_inst.name}' does not have shift source '{unc}', skipping it for "
                            f"parameter '{par}'",
                        )
                        continue

                    self.add_parameter(
                        f"{par}",
                        process=inference_processes.get(proc, proc),
                        category=f"*_{era_suffix(config_inst)}",
                        type=ParameterType.shape,
                        config_data={
                            config_inst.name: self.parameter_config_spec(
                                shift_source=unc,
                            ),
                        },
                    )

    self.cleanup()


# Z' signal mass/width grid used to derive individual inference models. `width` is the ABSOLUTE
# resonance width in GeV (matching the zprime_tt_m<mass>_w<width> dataset naming), computed from
# the mass and a relative width percentage.
#
# NOTE: .derive() only registers a lightweight subclass here - it does NOT run the
# an_v12_simplified body above (which queries configs, datasets, categories, etc.). That body
# only executes once, when law actually instantiates the one specific model requested via
# `--inference-model <name>` for a given task run. So looping over the full grid here is cheap;
# only the requested model is ever "instantiated" in the expensive sense.
ZPRIME_MASSES = [
    400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000,
    2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000,
]
ZPRIME_WIDTH_PERCENTAGES = [1, 10, 30]

for _mass in ZPRIME_MASSES:
    for _width_pct in ZPRIME_WIDTH_PERCENTAGES:
        # exact integer arithmetic (masses are always multiples of 100) avoids float rounding
        _width = (_mass // 100) * _width_pct
        _model_name = f"an_v12_simplified__m{_mass}_w{_width}"
        globals()[_model_name] = an_v12_simplified.derive(
            _model_name,
            cls_dict={
                "signal_mass": _mass,
                "signal_width": _width,
            },
        )

an_v12_simplified__m3000_w300__merged = an_v12_simplified.derive(
    "an_v12_simplified__m3000_w300__merged",
    cls_dict={"signal_mass": 3000, "signal_width": 300, "merge_eras": True},
)
