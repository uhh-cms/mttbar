# coding: utf-8

"""
mttbar inference model
"""

from columnflow.inference import inference_model, ParameterType, ParameterTransformation


@inference_model
def an_2019_197(self):
    """
    Inference model intended to reproduce the fits in CMS-AN-2019-197.
    """

    year = self.config_inst.campaign.x.year  # noqa; not used right now

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
            inference_processes.get(proc, proc),
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

    for proc in processes:
        for unc in uncertainty_shifts:
            par = inference_pars.get(unc, unc)
            self.add_parameter(
                f"{par}_{proc}",
                process=inference_processes.get(proc, proc),
                type=ParameterType.shape,
                config_shift_source=unc,
            )

    self.cleanup()
