# coding: utf-8

"""
Definition of categories.
"""

import order as od


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    config.add_category(
        name="all",
        id=1,
        selection="select_all",
        label="inclusive",
    )
