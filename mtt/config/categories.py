# coding: utf-8

"""
Definition of categories.
"""

import order as od

from columnflow.config_util import create_category_combinations


def add_categories_selection(config: od.Config) -> None:
    """
    Adds categories to a *config* that are available after the selection step.
    """
    config.add_category(
        name="incl",
        id=1,
        selection="sel_incl",
        label="inclusive",
    )

    # top category for electron channel
    config.add_category(
        name="1e",
        id=2,
        selection="sel_1e",
        label="1e",
        #channel=config.get_channel("e"),
    )

    # top category for muon channel
    config.add_category(
        name="1m",
        id=3,
        selection="sel_1m",
        label=r"1$\mu$",
        #channel=config.get_channel("mu"),
    )

    # number of top tags
    config.add_category(
        name="0t",
        id=4,
        selection="sel_0t",
        label=r"0t",
    )
    config.add_category(
        name="1t",
        id=5,
        selection="sel_1t",
        label=r"1t",
    )


    # -- combined categories

    category_groups = {
        "lepton": [
            config.get_category(name)
            for name in ["1e", "1m"]
        ],
        "n_top_tags": [
            config.get_category(name)
            for name in ["0t", "1t"]
        ],
    }

    def name_fn(**groups):
        return "__".join(
            cat_name for cat_name in groups.values()
            if cat_name is not None
        )

    def kwargs_fn(categories: dict[str, od.Category]):
        return {
            "id": "+",
            "selection": [cat.selection for cat in categories.values()],
            "label": ", ".join(
                cat.label for cat in categories.values()
            )
        }

    create_category_combinations(config, category_groups, name_fn, kwargs_fn)


def add_categories_production(config: od.Config) -> None:
    """
    Adds categories to a *config* that are available after the feature
    production step.
    """

    # -- atomic categories

    # categorization from cut on chi2
    chi2_max = config.x.categorization.chi2_max
    config.add_category(
        name="chi2pass",
        id=6,
        selection="sel_chi2pass",
        label=rf"$\chi^2 < {chi2_max}$",
    )
    config.add_category(
        name="chi2fail",
        id=7,
        selection="sel_chi2fail",
        label=rf"$\chi^2 \geq {chi2_max}$",
    )

    # categories corresponding to cos(theta*) bins
    config.add_category(
        name="acts_0_5",
        id=8,
        selection="sel_acts_0_5",
        label=r"$|{cos}(\theta^*)| < 0.5$",
    )
    config.add_category(
        name="acts_5_7",
        id=9,
        selection="sel_acts_5_7",
        label=r"$0.5 < |{cos}(\theta^*)| < 0.7$",
    )
    config.add_category(
        name="acts_7_9",
        id=10,
        selection="sel_acts_7_9",
        label=r"$0.7 < |{cos}(\theta^*)| < 0.9$",
    )
    config.add_category(
        name="acts_9_1",
        id=11,
        selection="sel_acts_9_1",
        label=r"$0.9 < |{cos}(\theta^*)| < 1.0$",
    )


    # -- combined categories

    category_groups = {
        "lepton": [
            config.get_category(name)
            for name in ["1e", "1m"]
        ],
        "n_top_tags": [
            config.get_category(name)
            for name in ["0t", "1t"]
        ],
        "chi2": [
            config.get_category(name)
            for name in ["chi2pass", "chi2fail"]
        ],
        "cos_theta_star": [
            config.get_category(name)
            for name in ["acts_0_5", "acts_5_7", "acts_7_9", "acts_9_1"]
        ],
    }

    def name_fn(**groups):
        return "__".join(
            cat_name for cat_name in groups.values()
            if cat_name is not None
        )

    def kwargs_fn(categories: dict[str, od.Category]):
        return {
            "id": "+",
            "selection": [cat.selection for cat in categories.values()],
            "label": ", ".join(
                cat.label for cat in categories.values()
            )
        }

    create_category_combinations(config, category_groups, name_fn, kwargs_fn)


