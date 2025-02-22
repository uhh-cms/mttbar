# coding: utf-8

"""
Definition of categories for trigger study analysis.
"""

import order as od


from columnflow.util import maybe_import
from columnflow.config_util import (
    create_category_combinations, add_category,
)


def name_fn(categories: dict[str, od.Category]):
    """Naming function for automatically generated combined categories."""
    return "__".join(
        cat.name for cat in categories.values()
        if cat.name is not None
    )


def add_categories(config: od.Config) -> None:
    """
    Adds categories specific to the trigger study analysis to a *config*.
    """

    # inclusive category, passes all events
    config.add_category(
        name="incl",
        id=0,
        selection="sel_incl",
        label="inclusive",
    )

    # helper dict for creating categories
    category_info = {
        "electron_pt": [
            {
                "name": "ele_pt_lt_120",
                "label": "$p_{T}$ < 120 GeV",
            },
            {
                "name": "ele_pt_120_200",
                "label": "120 < $p_{T}$ < 200 GeV",
            },
            {
                "name": "ele_pt_gt_200",
                "label": "$p_{T}$ > 200 GeV",
            },
        ],
        "electron_trigger": [
            {
                "name": "ele_trigger_pass",
                "label": "pass electron trigger",
            },
            {
                "name": "ele_trigger_fail",
                "label": "fail electron trigger",
            },
        ],
    }

    # create individual categories
    for category_dicts in category_info.values():
        for category_dict in category_dicts:
            add_category(
                config,
                name=category_dict["name"],
                selection=f"cat_{category_dict['name']}",
                label=category_dict["label"],
            )

    # group dict needed for combined category creation
    category_groups = {
        cat_group: [
            config.get_category(category_dict["name"])
            for category_dict in category_dicts
        ]
        for cat_group, category_dicts in category_info.items()
    }

    # create combined categories (one from each group)
    create_category_combinations(config, category_groups, name_fn)
