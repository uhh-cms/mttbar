# coding: utf-8

"""
mttbar inference model
"""
from __future__ import annotations

import law
import order as od

from columnflow.inference import inference_model, ParameterType, ParameterTransformation, InferenceModel, FlowStrategy

logger = law.logger.get_logger(__name__)


@inference_model(
    # mass and width to be set in derived classes
    signal_mass=None,
    signal_width=None,
    process_rates={
        "tt": 1.2,
        "st": 1.5,
        "others": 1.5,
    },
    fit_var="ttbar_mass_inf",
    merge_eras=True,
    blinded_datacard=True,
)
def zprime_base_model(
    self: InferenceModel
) -> None:
    """
    Inference model for Z' -> ttbar, based on the AN v12 inference model.
    No binning in cos(theta*) included at the moment.
    """
    #
    # setup
    #
    config_insts = self.config_insts
    signal_mass = self.signal_mass
    signal_width = self.signal_width
    process_rates = self.process_rates
    merge_eras = self.merge_eras
    background_processes = list(process_rates.keys())
    fit_var = self.fit_var
    blinded_datacard = self.blinded_datacard

    if signal_mass is None or signal_width is None:
        raise ValueError("signal_mass and signal_width must be set in derived classes")
    if merge_eras:
        logger.warning("With combined eras, no era-specific systematics can be applied.")
    if not blinded_datacard:
        logger.warning("Unblinded datacard requested. Is this intended?")

    def era_suffix(config_inst: od.Config) -> str:
        return str(config_inst.campaign.x.year)[-2:]

    # debug message
    msg = f"Creating inference model '{self.cls_name}' with the following parameters:"
    msg += f"\nSignal mass: {signal_mass} GeV"
    msg += f"\nSignal width: {signal_width} GeV"
    msg += f"\nBackground processes: {background_processes}"
    msg += f"\nFit variable: {fit_var}"
    msg += f"\nMerge eras: {merge_eras}"
    msg += f"\nBlinded datacard: {blinded_datacard}"
    logger.debug(msg)

    #
    # regions
    #

    categories = [
        # signal regions
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

            extra_kwargs = {"data_from_processes": background_processes} if blinded_datacard else {}
            self.add_category(
                name=inference_cat,
                config_data=config_data,
                mc_stats=True,
                flow_strategy=FlowStrategy.warn,
                rate_precision=5,
                **extra_kwargs,
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

                config_data = {
                    config_inst.name: self.category_config_spec(
                        category=config_cat,
                        variable=fit_var,
                        data_datasets=["data_*"],
                    ),
                }

                extra_kwargs = {"data_from_processes": background_processes} if blinded_datacard else {}
                self.add_category(
                    name=f"{inference_cat}_{era_suffix(config_inst)}",
                    config_data=config_data,
                    mc_stats=True,
                    flow_strategy=FlowStrategy.warn,
                    rate_precision=5,
                    **extra_kwargs,
                )

    #
    # processes
    #
    processes = background_processes + [f"zprime_tt_m{signal_mass}_w{signal_width}"]
    logger.debug(f"Adding processes: {processes}")

    for proc in processes:
        config_data = {}
        for config_inst in config_insts:
            if not config_inst.has_process(proc):
                if proc in background_processes:
                    raise ValueError(
                        f"config '{config_inst.name}' does not have background process '{proc}', "
                        f"required for inference process '{proc}'",
                    )
                logger.warning(
                    f"config '{config_inst.name}' does not have process '{proc}', skipping it for "
                    f"inference process '{proc}'",
                )
                continue

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
        if not config_data:
            logger.warning(f"process '{proc}' not found in any config - skipping it entirely")
            continue

        self.add_process(
            name=proc,
            config_data=config_data,
            is_signal=proc not in background_processes,
        )

    processes = sorted({proc_obj.name for cat_obj in self.categories for proc_obj in cat_obj.processes})

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
    for proc, rate in process_rates.items():
        self.add_parameter(
            f"xsec_{proc}",
            type=ParameterType.rate_gauss,
            effect=rate,
            process=proc,
        )

    # shape systematics
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

        # fixed wp uncs, simple scheme for now
        "btag_bc",
        "btag_light",
        # "btag_correlated_bc",
        # "btag_uncorrelated_bc",
        # "btag_correlated_light",
        # "btag_uncorrelated_light",

        # normalized
        "pdf",
        "mur",
        # "muf",  # FIXME: include in rerun!
        "isr",
        "fsr",

        # lepton uncs
        # merged across pt regimes
        # "muon_reco",
        # "muon_id",
        # "muon_iso",
        # "muon_trigger",
        # "electron_reco",
        # "electron_id_iso_low",  # joint low-pT id+iso SF, kept separate from "electron_id"
        # "electron_id",
        # "electron_iso",
        # "electron_trigger",

        # jec
        # TODO which ones do we need/try first?
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
                    process=proc,
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
                        process=proc,
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
# zprime_base_model body above (which queries configs, datasets, categories, etc.). That body
# only executes once, when law actually instantiates the one specific model requested via
# `--inference-model <name>` for a given task run. So looping over the full grid here is cheap;
# only the requested model is ever "instantiated" in the expensive sense.
ZPRIME_MASSES = [
    400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000,
    2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000,
]
ZPRIME_WIDTH_PERCENTAGES = [1, 10, 30]

for mass in ZPRIME_MASSES:
    for width_percentage in ZPRIME_WIDTH_PERCENTAGES:
        width = (mass // 100) * width_percentage
        model_name_base = f"zp_m{mass}_w{int(width)}"
        # zprime_base_model.derive(
        #     f"{model_name_base}__split_eras",  # not needed due to lumi correlation scheme
        #     cls_dict={
        #         "signal_mass": mass,
        #         "signal_width": width,
        #         "merge_eras": False,
        #     },
        # )
        zprime_base_model.derive(
            f"{model_name_base}__default",
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
            },
        )
