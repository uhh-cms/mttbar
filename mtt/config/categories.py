# coding: utf-8

"""
Definition of categories.

Categories are assigned a unique integer ID according to a fixed numbering
scheme, with digits/groups of digits indicating the different category groups:

                   lowest digit
                              |
    +---+---+---+---+---+---+---+
    | D | D | B | B | X | T | C |
    +---+---+---+---+---+---+---+

    C = channel       (1: electron [1e], 2: muon [1m])
    T = top-tag       (1: no top-tag [0t], 2: one top-tag [1t])
    X = chi2 cut      (1: pass [chi2pass], 2: fail [chi2fail])
    B = other binning (e.g. abs(cos(theta*)) a.k.a. 'acts')
    D = DNN category

Category groups are defined at different stages in the workflow:

    C, T: added after selection
    X, B: added after production
    D:    added after machine learning step

A digit group consisting entirely of zeroes ('0') represents the inclusive
category for the corresponding category group, i.e. no selection from that
group is applied.

This scheme encodes child/parent relations into the ID, making it easy
to check if categories overlap or are categories of each other. When applied
to a set of leaf categories, the sum of the category IDs is the ID of the
parent category.
"""

import order as od

from columnflow.ml import MLModel
from columnflow.config_util import create_category_combinations

from mtt.ml.categories import register_ml_selectors


def name_fn(categories: dict[str, od.Category]):
    """Naming function for automatically generated combined categories."""
    return "__".join(
        cat.name for cat in categories.values()
        if cat.name is not None
    )


def kwargs_fn(categories: dict[str, od.Category]):
    """Customization function for automatically generated combined categories."""
    return {
        "id": sum(cat.id for cat in categories.values()),
        "selection": [cat.selection for cat in categories.values()],
        "label": ", ".join(
            cat.label for cat in categories.values()
        ),
    }


def add_categories_selection(config: od.Config) -> None:
    """
    Adds categories to a *config* that are available after the selection step.
    """
    config.add_category(
        name="incl",
        id=0,
        selection="sel_incl",
        label="inclusive",
    )

    # top category for electron channel
    config.add_category(
        name="1e",
        id=1,
        selection="sel_1e",
        label="1e",
        #channel=config.get_channel("e"),  # noqa
    )

    # top category for muon channel
    config.add_category(
        name="1m",
        id=2,
        selection="sel_1m",
        label=r"1$\mu$",
        #channel=config.get_channel("mu"),  # noqa
    )

    # number of top tags
    config.add_category(
        name="0t",
        id=10,
        selection="sel_0t",
        label=r"0t",
    )
    config.add_category(
        name="1t",
        id=20,
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
        id=100,
        selection="sel_chi2pass",
        label=rf"$\chi^2 < {chi2_max}$",
    )
    config.add_category(
        name="chi2fail",
        id=200,
        selection="sel_chi2fail",
        label=rf"$\chi^2 \geq {chi2_max}$",
    )

    # categories corresponding to cos(theta*) bins
    config.add_category(
        name="acts_0_5",
        id=1000,
        selection="sel_acts_0_5",
        label=r"$|{cos}(\theta^*)| < 0.5$",
    )
    config.add_category(
        name="acts_5_7",
        id=2000,
        selection="sel_acts_5_7",
        label=r"$0.5 < |{cos}(\theta^*)| < 0.7$",
    )
    config.add_category(
        name="acts_7_9",
        id=3000,
        selection="sel_acts_7_9",
        label=r"$0.7 < |{cos}(\theta^*)| < 0.9$",
    )
    config.add_category(
        name="acts_9_1",
        id=4000,
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

    create_category_combinations(config, category_groups, name_fn, kwargs_fn)


def add_categories_ml(config: od.Config, ml_model_inst: MLModel) -> None:
    """
    Adds categories to a *config* that are available after the machine learning step.
    """

    # -- register category selectors

    register_ml_selectors(ml_model_inst)

    # -- atomic categories

    # get output categories from ML model
    dnn_categories = []
    for i, proc in enumerate(ml_model_inst.processes):
        cat = config.add_category(
            name=f"dnn_{proc}",
            id=(i + 1) * 100000,
            selection=f"sel_dnn_{proc}",
            label=f"dnn_{proc}",
        )
        dnn_categories.append(cat)

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
        "dnn": dnn_categories,
    }

    # fixed numbering scheme for ML categories
    def kwargs_fn_dnn(categories: dict[str, od.Category]):
        """
        Customization function for automatically generated combined ML categories,
        with fixed numbering scheme: 10000 * (1 + proc_id) + parent_category_id

        NOTE: this numbering scheme is required in order for the MLEvaluation task
        to produce the expected results.
        """

        # determine name of parent category
        name_cat_no_dnn = name_fn(
            **{
                group: category.name
                for group, category in categories.items()
                if group != "dnn"
            },
        )

        # raise if parent category has not been added yet
        if not config.has_category(name_cat_no_dnn):
            name_cat = name_fn(
                **{
                    group: category.name
                    for group, category in categories.items()
                },
            )
            raise Exception(
                f"cannot compute ID for ML category '{name_cat}': "
                f"parent category '{name_cat_no_dnn}' does not exist",
            )

        # determined ID of parent category
        id_cat_no_dnn = config.get_category(name_cat_no_dnn).id

        return {
            "id": categories["dnn"].id + id_cat_no_dnn,
            "selection": [cat.selection for cat in categories.values()],
            "label": ", ".join(
                cat.label for cat in categories.values()
            ),
        }

    create_category_combinations(
        config,
        category_groups,
        name_fn,
        kwargs_fn_dnn,
        skip_existing=True,
    )
