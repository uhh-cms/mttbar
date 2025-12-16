# coding: utf-8

"""
Configuration of default values and groups for the m(ttbar) analysis.
"""

import order as od
import itertools

from columnflow.util import DotDict


def set_defaults(
        config: od.Config,
) -> None:
    """
    Set default values for the m(ttbar) analysis configuration.
    """
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    base_defaults = DotDict.wrap({
        "calibrator": "skip_jecunc",
        "selector": "default",
        "reducer": "default",
        "producer": None,
        "weight_producer": "all_weights",
        "hist_producer": "cf_default",
        "ml_model": None,
        "inference_model": "simple",
        "categories": [
            "1m", "1e", "1m__0t", "1e__0t", "1m__1t", "1e__1t"
        ],
        "variables": [
            "electron_pt", "muon_pt"
        ],
        "dataset": "tt_sl_powheg",
    })

    overrides = {
        # set custum defaults for specific runs and tags if needed
        # (3, "2022preEE"): DotDict.wrap({
        #     "reducer": "cf_default",
        # })
    }

    override = overrides.get((run, tag), DotDict())
    merged_defaults = {**base_defaults, **override}

    for key, value in merged_defaults.items():
        setattr(config.x, f"default_{key}", value)


def set_process_groups(
        config: od.Config,
) -> None:
    """
    Set groups for the m(ttbar) analysis configuration.
    """
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    base_processes = DotDict.wrap({
        "bkg": [
            "tt",
            "st",
            "w_lnu",
            "dy",
            "qcd",
            "vv",
        ],
        "sig": [
            "zprime_tt_m500_w50",
        ],
    })

    base_processes.default = base_processes.bkg + base_processes.sig
    overrides = {
        # set custom process groups for specific runs and tags if needed
        (3, "2022preEE"): DotDict.wrap({
            "sig": [],
        }),
        (3, "2022postEE"): DotDict.wrap({
            "sig": [],
        }),
    }
    override = overrides.get((run, tag), DotDict())
    merged_processes = {**base_processes, **override}

    config.x.process_groups = DotDict.wrap(merged_processes)


def set_dataset_groups(
        config: od.Config,
) -> None:
    """
    Set groups for the datasets in the m(ttbar) analysis configuration.
    """
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    base_datasets = DotDict.wrap({
        "all": ["*"],
        "data": [
            "data_mu_*", "data_egamma_*",
        ],
        "tt": ["tt_*"],
        "st": ["st_*"],
        "w": ["w_lnu_*"],
        "dy": ["dy_*"],
        "w_lnu": ["w_lnu_*"],
        "qcd": ["qcd_*"],
        "vv": ["ww_*", "wz_*", "zz_*"],
        "zprime_tt": ["zprime_tt_*"],
        "hpseudo_tt": ["hpseudo_tt_*"],
        "hscalar_tt": ["hscalar_tt_*"],
        "rsgluon_tt": ["rsgluon_tt_*"],
        "bkg": [
            "tt_*", "st_*", "w_lnu_*", "dy_*",
            "qcd_*", "ww_*", "wz_*", "zz_*",
        ],
        "zprime_default": [
            "zprime_tt_m500_w50_madgraph",
            "zprime_tt_m1000_w100_madgraph",
            "zprime_tt_m3000_w300_madgraph",
        ],
    })

    overrides = {
        # set custom dataset groups for specific runs and tags if needed
        (2, "2017"): DotDict.wrap({
            "data": [
                "data_mu_*", "data_e_*", "data_pho_*",
            ],
        }),
    }
    override = overrides.get((run, tag), DotDict())
    merged_datasets = {**base_datasets, **override}

    config.x.dataset_groups = DotDict.wrap(merged_datasets)


def set_category_groups(
        config: od.Config,
) -> None:
    """
    Set groups for the categories in the m(ttbar) analysis configuration.
    """
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    lepton_categories = [
        "1e", "1m",
    ]
    top_tag_categories = [
        "0t", "1t",
    ]
    chi2_categories = [
        "pass", "fail"
    ]
    acts_categories = [
        "0_5", "5_7", "7_9", "9_1"
    ]

    def generate_all_category_combinations(
            dimensions: tuple,
            min_depth: int = 1,
            max_depth: int = None
    ) -> list[str]:
        """
        Generate all combinations of categories from the given dimensions.

        Parameters:
            dimensions: list of (label_prefix, values) tuples
            min_depth: minimum number of dimensions to combine
            max_depth: maximum number of dimensions to combine (default: all)

        Returns:
            Sorted list of all category combinations as strings
        """
        if max_depth is None:
            max_depth = len(dimensions)

        all_combinations = set()

        for depth in range(min_depth, max_depth + 1):
            for dim_subset in itertools.combinations(dimensions, depth):
                labels, values_lists = zip(*dim_subset)
                for values in itertools.product(*values_lists):
                    parts = [
                        f"{label}{value}" if label else value
                        for label, value in zip(labels, values)
                    ]
                    all_combinations.add("__".join(parts))

        return sorted(all_combinations)

    dims = [
        ("", lepton_categories),
        ("", top_tag_categories),
        ("chi2", chi2_categories),
        ("acts_", acts_categories),
    ]
    all_categories = generate_all_category_combinations(dims, min_depth=1)

    base_categories = DotDict.wrap({
        "default": config.x.default_categories,
        "all": all_categories + ["incl"],  # careful, lots of categories
        "lepton": lepton_categories,
        "chi2p": [
            f"{lep}__{toptag}__chi2pass"
            for lep in lepton_categories
            for toptag in top_tag_categories
        ],
        "chi2f": [
            f"{lep}__{toptag}__chi2fail"
            for lep in lepton_categories
            for toptag in top_tag_categories
        ],
        "resolved": [
            f"{lep}__0t__chi2{chi2}__acts_{acts}"
            for lep in lepton_categories
            for chi2 in chi2_categories
            for acts in acts_categories
        ],
        "boosted": [
            f"{lep}__1t__chi2{chi2}__acts_{acts}"
            for lep in lepton_categories
            for chi2 in chi2_categories
            for acts in acts_categories
        ],
    })

    overrides = {
        # set custom category groups for specific runs and tags if needed
    }
    override = overrides.get((run, tag), DotDict())
    merged_categories = {**base_categories, **override}

    config.x.category_groups = DotDict.wrap(merged_categories)


def set_variables_groups(
        config: od.Config,
) -> None:
    """
    Set groups for the variables in the m(ttbar) analysis configuration.
    """
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    base_variables = DotDict.wrap({
        "default": [
            "n_jet", "n_muon", "n_electron",
            "jet1_pt", "jet2_pt", "jet3_pt", "jet4_pt",
            "fatjet1_pt", "fatjet2_pt", "fatjet3_pt", "fatjet4_pt",
            "muon_pt", "muon_eta",
            "electron_pt", "electron_eta",
        ],
        "cutflow": [
            "cf_n_jet", "cf_n_muon", "cf_n_electron",
            "cf_jet1_pt", "cf_jet2_pt", "cf_jet3_pt", "cf_jet4_pt",
            "cf_fatjet1_pt", "cf_fatjet2_pt", "cf_fatjet3_pt", "cf_fatjet4_pt",
            "cf_muon_pt", "cf_muon_eta",
            "cf_electron_pt", "cf_electron_eta",
        ],
        "new_version_test": [
            "n_jet", "n_electron", "n_muon",
            "puppi_met_pt", "puppi_met_phi",
            "electron_pt", "electron_phi", "electron_eta",
            "muon_pt", "muon_phi", "muon_eta",
            "jet1_pt", "jet1_phi", "jet1_eta",
            "fatjet1_pt", "fatjet1_phi", "fatjet1_eta",
            "chi2_lt100",
            "top_had_mass",
            "top_lep_mass",
            "cos_theta_star",
            # "gen_top_had_mass",
            # "gen_top_lep_mass",
            # "gen_cos_theta_star",
        ],
    })

    overrides = {
        # set custom variable groups for specific runs and tags if needed
    }
    override = overrides.get((run, tag), DotDict())
    merged_variables = {**base_variables, **override}

    config.x.variable_groups = DotDict.wrap(merged_variables)


def set_shift_groups(
        config: od.Config,
) -> None:
    """
    Set groups for the shifts in the m(ttbar) analysis configuration.
    """
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    base_shifts = DotDict.wrap({
        "jer": ["nominal", "jer_up", "jer_down"],
    })

    overrides = {
        # set custom shift groups for specific runs and tags if needed
    }
    override = overrides.get((run, tag), DotDict())
    merged_shifts = {**base_shifts, **override}

    config.x.shift_groups = DotDict.wrap(merged_shifts)


def set_selector_steps(
        config: od.Config,
) -> None:
    """
    Set selector steps for the m(ttbar) analysis configuration.
    """
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    base_steps = {
        "default": ["METFilters", "DiLeptonVeto", "AllHadronicVeto", "JetLepton2DCut", "BJet", "Jet", "MET", "Lepton"],
    }
    base_steps_labels = {
        "JetLepton2DCut": "2D cut",
        "AllHadronicVeto": r"all-hadr. veto",
        "DileptonVeto": r"dilep. veto",
        "jet_veto_map": "JetVetoMap",
    }

    overrides = {
        # set custom selector steps for specific runs and tags if needed
    }
    overrides_labels = {
        # set custom labels for specific selector steps for specific runs and tags if needed
    }
    override = overrides.get((run, tag), DotDict())
    override_labels = overrides_labels.get((run, tag), DotDict())
    merged_steps = {**base_steps, **override}
    merged_steps_labels = {**base_steps_labels, **override_labels}

    config.x.selector_step_groups = DotDict.wrap(merged_steps)
    config.x.selector_steps_labels = DotDict.wrap(merged_steps_labels)
