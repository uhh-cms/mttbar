# coding: utf-8

"""
mttbar inference model
"""

from columnflow.inference import inference_model, ParameterType, ParameterTransformation


@inference_model
def test(self):
    """
    Very basic inference model intended for testing.
    """

    year = self.config_inst.campaign.x.year  # noqa; not used right now

    #
    # regions/categories
    #

    # tuples of inference categories and
    # corresponding config category names
    categories = [
        # signal regions
        ("ch_1m", "1m"),
        ("ch_1e", "1e"),
    ]

    # add categories to inference model
    for inference_cat, config_cat in categories:
        self.add_category(
            inference_cat,
            config_category=config_cat,
            config_variable="ttbar_mass",
            mc_stats=True,
            # no real data yet, use sum of backgrounds as fake data
            config_data_datasets=[],
            data_from_processes=["tt"],  # TODO: add backgrounds
        )

    #
    # processes
    #

    processes = [
        # signals
        "zprime_tt_m3000_w300",
        # backgrounds
        "tt",
    ]

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
            proc,
            config_process=proc,
            is_signal=proc.startswith("zprime_tt"),
            config_mc_datasets=datasets,
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
    for proc in ["tt"]:
        self.add_parameter(
            f"{proc}_rate",
            type=ParameterType.rate_gauss,
        )

    # systematic shifts (none)
    uncertainty_shifts = [
         "minbias_xs",  # pileup
    ]

    # different naming convention for some parameters
    inference_pars = {
        "minbias_xs": "pu",
    }

    for proc in processes:
        for unc in uncertainty_shifts:
            par = inference_pars.get(unc, unc)
            self.add_parameter(
                f"{par}_{proc}",
                process=proc,
                type=ParameterType.shape,
                config_shift_source=unc,
            )

    self.cleanup()
