
"""
mttbar inference model
"""
from __future__ import annotations

import law
import order as od
from columnflow.inference import (
    FlowStrategy,
    InferenceModel,
    ParameterTransformation,
    ParameterType,
    inference_model,
)

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
    # each entry maps an uncertainty shift source to a correlation spec:
    #   "all"                                    -> one parameter, shared across every process and category
    #   ["tt", "st", "others"]                    -> shorthand: one INDEPENDENT parameter per listed process
    #                                                 (categories stay correlated/"all")
    #   {"processes": [...] | "all", "categories": [...] | "all"}
    #                                              -> full control over both axes independently
    # a plain list/tuple/set of shift names (the old format) is also accepted as shorthand for
    # {unc: "all" for unc in list}
    shape_uncs={
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

        "minbias_xs": "all",  # pileup

        # fixed wp uncs, simple scheme for now
        "btag_bc": "all",
        "btag_light": "all",
        # "btag_correlated_bc",
        # "btag_uncorrelated_bc",
        # "btag_correlated_light",
        # "btag_uncorrelated_light",

        # normalized
        "pdf": "all",
        "mur": "all",
        "muf": "all",  # FIXME: include in rerun!
        "isr": "all",
        "fsr": "all",

        # lepton uncs merged across pt regimes
        "muon_reco": "all",
        "muon_id": "all",
        "muon_iso": "all",
        "muon_trigger": "all",
        "electron_reco": "all",
        "electron_id_iso_low": "all",  # joint low-pT id+iso SF, kept separate from "electron_id"
        "electron_id": "all",
        "electron_iso": "all",
        "electron_trigger": "all",

        # jec
        # TODO which ones do we need/try first?
        # requires running CSR workflow
    },
    merge_eras=True,
    blinded_datacard=True,
    # connect CR to SR via process rates; currently not implemented
    # absorbed into the process_rates dict above for now
    include_rate_params=False,
    inference_regions=[
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
    ],
)
def zprime_base_model(
    self: InferenceModel,
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
    uncertainty_shifts = self.shape_uncs
    include_rate_params = self.include_rate_params
    categories = self.inference_regions

    if signal_mass is None or signal_width is None:
        raise ValueError("signal_mass and signal_width must be set in derived classes")
    if merge_eras:
        logger.warning("With combined eras, no era-specific systematics can be applied.")
    if not blinded_datacard:
        logger.warning("Unblinded datacard requested. Is this intended?")
    if include_rate_params:
        # TODO: implement this feature, which would add a rate parameter per process to connect CR to SR
        # alternative to the current approach of using fixed process rates
        raise NotImplementedError("include_rate_params is not yet implemented")

    def era_suffix(config_inst: od.Config) -> str:
        return str(config_inst.campaign.x.year)[-2:]

    # debug message
    msg = f"Creating inference model '{self.cls_name}' with the following parameters:"
    msg += f"\nSignal mass: {signal_mass} GeV"
    msg += f"\nSignal width: {signal_width} GeV"
    msg += f"\nBackground processes: {background_processes}"
    msg += f"\nFit variable: {fit_var}"
    msg += f"\nProcess rates: {process_rates}"
    msg += f"\nUncertainty shifts: {uncertainty_shifts}"
    msg += f"\nMerge eras: {merge_eras}"
    msg += f"\nBlinded datacard: {blinded_datacard}"
    msg += f"\nInclude rate parameters: {include_rate_params}"
    msg += f"\nInference regions: {categories}"
    logger.debug(msg)

    #
    # regions
    #

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
    category_names = [cat_obj.name for cat_obj in self.categories]
    signal_process_name = f"zprime_tt_m{signal_mass}_w{signal_width}"

    #
    # parameters
    #

    # lumi
    # each luminosity uncertainty name is already unique per era (e.g. "lumi_13p6TeV_2024" vs
    # "lumi_13p6TeV_2025" - no name is shared across configs), so there's no cross-era merging to
    # do here regardless of `merge_eras`.
    if merge_eras:
        logger.warning(
            "merge_eras is enabled: per-era luminosity uncertainties are applied to the full "
            "merged category rate as an approximation, not scoped to their own era's yield share; "
            "using the single largest per-era uncertainty, as no cross-era correlation scheme is "
            "implemented yet.",
        )

        # collect (config_name, unc_name, (down, up)) for every luminosity uncertainty across configs
        lumi_uncs = []
        for config_inst in config_insts:
            lumi = config_inst.x.luminosity
            if not lumi:
                logger.warning(
                    f"config '{config_inst.name}' does not have luminosity information, skipping it",
                )
                continue
            for unc_name in lumi.uncertainties:
                effect = lumi.get(names=unc_name, direction=("down", "up"), factor=True)
                lumi_uncs.append((config_inst.name, unc_name, effect))

        if not lumi_uncs:
            raise ValueError(
                f"none of the configs {[c.name for c in config_insts]} have luminosity information, "
                f"required for inference model '{self.cls_name}'",
            )

        # magnitude of a (down, up) factor pair: how far the larger side deviates from 1.0
        def magnitude(effect: tuple[float, float]) -> float:
            down, up = effect
            return max(abs(down - 1.0), abs(up - 1.0))

        config_name, unc_name, effect = max(lumi_uncs, key=lambda item: magnitude(item[2]))
        logger.warning(
            f"largest lumi uncertainty across all eras is '{unc_name}' (from config '{config_name}'), "
            f"using it for the merged category rate; effect: {effect} (down, up) factor pair",
        )

        self.add_parameter(
            unc_name,
            type=ParameterType.rate_gauss,
            effect=effect,
            # transformations=[ParameterTransformation.symmetrize],  # does weird stuff atm -> fixed in latest cf TODO: update and test  # noqa: E501
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
    # different naming convention for some parameters
    inference_pars = {
        "minbias_xs": "pu",
    }

    # accept the old list form as shorthand for {unc: "all" for unc in list}, so existing
    # .derive() calls using a plain list of shift names keep working unchanged
    if isinstance(uncertainty_shifts, (list, tuple, set)):
        uncertainty_shifts = {unc: "all" for unc in uncertainty_shifts}
    elif not isinstance(uncertainty_shifts, dict):
        raise TypeError(
            f"shape_uncs must be a dict, list, tuple, or set, got {type(uncertainty_shifts)}: "
            f"{uncertainty_shifts!r}",
        )

    def resolve_proc(p: str) -> str:
        return signal_process_name if p == "signal" else p

    def resolve_shape_spec(unc: str, spec) -> tuple:
        """
        Validates and resolves a single shape_uncs entry into (proc_targets, cat_targets), where
        each is either the string "all" or a concrete list of names. Raises ValueError/TypeError
        with a clear message for any spec that can't be satisfied by this model instance, instead
        of silently matching nothing.
        """
        # normalize the spec into (raw_procs, raw_cats)
        if spec == "all":
            raw_procs, raw_cats = "all", "all"
        elif isinstance(spec, dict):
            unknown_keys = set(spec) - {"processes", "categories"}
            if unknown_keys:
                raise ValueError(
                    f"shape_uncs['{unc}']: unknown key(s) {sorted(unknown_keys)} in spec {spec!r}; "
                    f"only 'processes' and 'categories' are allowed",
                )
            raw_procs = spec.get("processes", "all")
            raw_cats = spec.get("categories", "all")
        elif isinstance(spec, (list, tuple, set)):
            raw_procs, raw_cats = list(spec), "all"
        else:
            raise TypeError(
                f"shape_uncs['{unc}']: invalid spec {spec!r} (type {type(spec)}); expected 'all', a list of "
                f"process names, or a dict with 'processes'/'categories' keys",
            )

        # resolve and validate processes
        if raw_procs == "all":
            proc_targets = "all"
        else:
            if not raw_procs:
                raise ValueError(f"shape_uncs['{unc}']: 'processes' list is empty")
            proc_targets = []
            for p in raw_procs:
                resolved = resolve_proc(p)
                if resolved not in processes:
                    raise ValueError(
                        f"shape_uncs['{unc}']: process '{p}' (resolved to '{resolved}') is not registered in "
                        f"model '{self.cls_name}'; available processes: {processes}",
                    )
                proc_targets.append(resolved)

        # resolve and validate categories
        if raw_cats == "all":
            cat_targets = "all"
        else:
            if not raw_cats:
                raise ValueError(f"shape_uncs['{unc}']: 'categories' list is empty")
            missing = [c for c in raw_cats if c not in category_names]
            if missing:
                raise ValueError(
                    f"shape_uncs['{unc}']: categor{'y' if len(missing) == 1 else 'ies'} {missing} not found in "
                    f"model '{self.cls_name}'; available categories: {category_names}",
                )
            cat_targets = list(raw_cats)

        return proc_targets, cat_targets

    for unc, spec in uncertainty_shifts.items():
        par = inference_pars.get(unc, unc)
        proc_targets, cat_targets = resolve_shape_spec(unc, spec)

        # "all" -> a single call covering every process/category (no name suffix, fully correlated)
        # a list -> one independent, suffixed parameter per listed process/category
        proc_iter = [(None, processes)] if proc_targets == "all" else [(p, [p]) for p in proc_targets]
        cat_iter = [(None, category_names)] if cat_targets == "all" else [(c, [c]) for c in cat_targets]

        for proc_suffix, proc_list in proc_iter:
            for cat_suffix, cat_list in cat_iter:
                name = par
                if proc_suffix:
                    name += f"_{proc_suffix}"
                if cat_suffix:
                    name += f"_{cat_suffix}"

                if merge_eras:
                    # one parameter per (proc, unc, cat) combo, spanning all configs that have
                    # this shift - since categories share plain names in merged mode (e.g.
                    # "el_sr1" for both eras), calling add_parameter once per config with the
                    # same name/process/category would collide on the second call
                    config_data = {}
                    for config_inst in config_insts:
                        if not config_inst.has_shift(f"{unc}_up") and not config_inst.has_shift(f"{unc}_down"):
                            logger.warning(
                                f"config '{config_inst.name}' does not have shift source '{unc}', skipping it "
                                f"for parameter '{name}'",
                            )
                            continue
                        config_data[config_inst.name] = self.parameter_config_spec(
                            shift_source=unc,
                        )

                    if not config_data:
                        continue

                    self.add_parameter(
                        name,
                        process=proc_list,
                        category=cat_list,
                        type=ParameterType.shape,
                        effect=1.0,
                        config_data=config_data,
                    )
                else:
                    for config_inst in config_insts:
                        if not config_inst.has_shift(f"{unc}_up") and not config_inst.has_shift(f"{unc}_down"):
                            logger.warning(
                                f"config '{config_inst.name}' does not have shift source '{unc}', skipping it "
                                f"for parameter '{name}'",
                            )
                            continue

                        # combine region-level decorrelation with the era-suffix wildcard: a
                        # specific region becomes "<region>_<era>"; "all" stays "*_<era>"
                        if cat_suffix:
                            era_cats = [f"{c}_{era_suffix(config_inst)}" for c in cat_list]
                        else:
                            era_cats = [f"*_{era_suffix(config_inst)}"]

                        self.add_parameter(
                            name,
                            process=proc_list,
                            category=era_cats,
                            type=ParameterType.shape,
                            effect=1.0,
                            config_data={
                                config_inst.name: self.parameter_config_spec(shift_source=unc),
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
#
# NOTE: shape_uncs below is still given as a plain list in each variant (the old format) - this
# still works unchanged, since the model body now normalizes a list into {unc: "all" for unc in
# list} automatically. Switch any entry to a dict spec (see the decorator default above) if/when
# you want per-process or per-category decorrelation for that variant.
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
            f"{model_name_base}__default",  # used for first testing, kept here for reference
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
                "shape_uncs": [
                    "minbias_xs",
                    "btag_bc",
                    "btag_light",
                    "pdf",
                    "mur",
                    "isr",
                    "fsr",
                ],
            },
        )
        zprime_base_model.derive(
            f"{model_name_base}__v2",  # used for further studies
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
                "shape_uncs": [
                    "minbias_xs",
                    "btag_bc",
                    "btag_light",
                    "pdf",
                    "mur",
                    "muf",
                    "isr",
                    "fsr",
                    "muon_reco",
                    "muon_id",
                    "muon_iso",
                    "electron_reco",
                    "electron_id_iso_low",
                    "electron_id",
                    "electron_iso",
                ],
                "fit_var": "ttbar_mass_v2",
            },
        )
        # CRonly model
        zprime_base_model.derive(
            f"{model_name_base}__cronly",
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
                "shape_uncs": [
                    "minbias_xs",
                    "btag_bc",
                    "btag_light",
                    "pdf",
                    "mur",
                    "muf",
                    "isr",
                    "fsr",
                    "muon_reco",
                    "muon_id",
                    "muon_iso",
                    "electron_reco",
                    "electron_id_iso_low",
                    "electron_id",
                    "electron_iso",
                ],
                "fit_var": "ttbar_mass_v2",
                "inference_regions": [
                    # control regions
                    ("el_cr1", "1e__dnn_st"),
                    ("mu_cr1", "1m__dnn_st"),
                    ("el_cr2", "1e__dnn_other"),
                    ("mu_cr2", "1m__dnn_other"),
                ],
            },
        )
        # st CRonly model
        zprime_base_model.derive(
            f"{model_name_base}__stcronly",
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
                "shape_uncs": [
                    "minbias_xs",
                    "btag_bc",
                    "btag_light",
                    "pdf",
                    "mur",
                    "muf",
                    "isr",
                    "fsr",
                    "muon_reco",
                    "muon_id",
                    "muon_iso",
                    "electron_reco",
                    "electron_id_iso_low",
                    "electron_id",
                    "electron_iso",
                ],
                "fit_var": "ttbar_mass_v2",
                "inference_regions": [
                    # control regions
                    ("el_cr1", "1e__dnn_st"),
                    ("mu_cr1", "1m__dnn_st"),
                ],
            },
        )
        # others CRonly model
        zprime_base_model.derive(
            f"{model_name_base}__otherscronly",
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
                "shape_uncs": [
                    "minbias_xs",
                    "btag_bc",
                    "btag_light",
                    "pdf",
                    "mur",
                    "muf",
                    "isr",
                    "fsr",
                    "muon_reco",
                    "muon_id",
                    "muon_iso",
                    "electron_reco",
                    "electron_id_iso_low",
                    "electron_id",
                    "electron_iso",
                ],
                "fit_var": "ttbar_mass_v2",
                "inference_regions": [
                    # control regions
                    ("el_cr2", "1e__dnn_other"),
                    ("mu_cr2", "1m__dnn_other"),
                ],
            },
        )
        zprime_base_model.derive(
            f"{model_name_base}__v3",
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
                "shape_uncs": [
                    "minbias_xs",
                    "btag_bc",
                    "btag_light",
                    "pdf",
                    "mur",
                    "muf",
                    "isr",
                    "fsr",
                    "muon_reco",
                    "muon_id",
                    "muon_iso",
                    "electron_reco",
                    "electron_id_iso_low",
                    "electron_id",
                    "electron_iso",
                ],
                "fit_var": "ttbar_mass_v3",
            },
        )
        # v2 var, only tt process, no others or st procs, all regions
        zprime_base_model.derive(
            f"{model_name_base}__v2_tt_only",
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
                "shape_uncs": [
                    "minbias_xs",
                    "btag_bc",
                    "btag_light",
                    "pdf",
                    "mur",
                    "muf",
                    "isr",
                    "fsr",
                    "muon_reco",
                    "muon_id",
                    "muon_iso",
                    "electron_reco",
                    "electron_id_iso_low",
                    "electron_id",
                    "electron_iso",
                ],
                "fit_var": "ttbar_mass_v2",
                "process_rates": {
                    "tt": 1.2,
                },
            },
        )
        # v2 var, only tt and st processes, no others, all regions
        zprime_base_model.derive(
            f"{model_name_base}__v2_tt_st_only",
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
                "shape_uncs": [
                    "minbias_xs",
                    "btag_bc",
                    "btag_light",
                    "pdf",
                    "mur",
                    "muf",
                    "isr",
                    "fsr",
                    "muon_reco",
                    "muon_id",
                    "muon_iso",
                    "electron_reco",
                    "electron_id_iso_low",
                    "electron_id",
                    "electron_iso",
                ],
                "fit_var": "ttbar_mass_v2",
                "process_rates": {
                    "tt": 1.2,
                    "st": 1.5,
                },
            },
        )
        # v2 var, tt, st, vv__dy__w_lnu process, decorrelated muf/mur/isr/fsr, all regions
        zprime_base_model.derive(
            f"{model_name_base}__v2__wo_qcd__dec_th",
            cls_dict={
                "signal_mass": mass,
                "signal_width": width,
                "merge_eras": True,
                "shape_uncs": {
                    "minbias_xs": "all",
                    "btag_bc": "all",
                    "btag_light": "all",
                    "pdf": "all",
                    "mur": {"processes": ["tt", "st", "vv__dy__w_lnu"], "categories": "all"},
                    "muf": {"processes": ["tt", "st", "vv__dy__w_lnu"], "categories": "all"},
                    "isr": {"processes": ["tt", "st", "vv__dy__w_lnu"], "categories": "all"},
                    "fsr": {"processes": ["tt", "st", "vv__dy__w_lnu"], "categories": "all"},
                    # lepton uncs merged across pt regimes
                    "muon_reco": "all",
                    "muon_id": "all",
                    "muon_iso": "all",
                    "electron_reco": "all",
                    "electron_id_iso_low": "all",
                    "electron_id": "all",
                    "electron_iso": "all",
                },
                "fit_var": "ttbar_mass_v2",
                "process_rates": {
                    "tt": 1.2,
                    "st": 1.5,
                    "vv__dy__w_lnu": 1.5,
                },
            },
        )
