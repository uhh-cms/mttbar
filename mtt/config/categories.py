# coding: utf-8

"""
Definition of categories.
"""

import order as od


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    cat_incl = config.add_category(
        name="incl",
        id=1,
        selection="sel_incl",
        label="inclusive",
    )
    cat_1e = config.add_category(
        name="1e",
        id=2,
        selection="sel_1e",
        label="1e",
        channel=config.get_channel("e"),
    )
    cat_1m = config.add_category(
        name="1m",
        id=3,
        selection="sel_1m",
        label=r"1$\mu$",
        channel=config.get_channel("mu"),
    )
