# coding: utf-8

"""
Configuration of taggers for the m(ttbar) analysis.
"""

import order as od

from columnflow.util import DotDict


def btag_params(
        config: od.Config,
) -> DotDict:
    """
    Returns the b-tagging parameters for the given config.
    b-tagging working points from
    2017:       https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
    2022preEE:  https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22/
    2022postEE: https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer22EE/
    2023preBPix: https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer23/
    2024:       https://btv-wiki.docs.cern.ch/ScaleFactors/Run3Summer24/
    """
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    wp_values = {
        2: {
            "2017": {
                "deepjet": {
                    "loose": 0.0532,
                    "medium": 0.3040,
                    "tight": 0.7476,
                },
                "deepcsv": {
                    "loose": 0.1355,
                    "medium": 0.4506,
                    "tight": 0.7738,
                },
            }
        },
        3: {
            "2022preEE": {
                "deepjet": {
                    "loose": 0.0583,
                    "medium": 0.3086,
                    "tight": 0.7183,
                },
                # "deepcsv": {
                #     "loose": 0.1208,
                #     "medium": 0.4168,
                #     "tight": 0.7665,
                # },
            },
            "2022postEE": {
                "deepjet": {
                    "loose": 0.0614,
                    "medium": 0.3196,
                    "tight": 0.7300,
                },
                # "deepcsv": {
                #     "loose": 0.1208,
                #     "medium": 0.4168,
                #     "tight": 0.7665,
                # },
            },
            "2023preBPix": {
                "deepjet": {
                    "loose": 0.0479,
                    "medium": 0.2431,
                    "tight": 0.6553,
                },
                # "deepcsv": {
                #     "loose": 0.1208,
                #     "medium": 0.4168,
                #     "tight": 0.7665,
                # },
            },
            "2023postBPix": {
                "deepjet": {
                    "loose": 0.048,
                    "medium": 0.2435,
                    "tight": 0.6563,
                },
                # "deepcsv": {
                #     "loose": 0.1208,
                #     "medium": 0.4168,
                #     "tight": 0.7665,
                # },
            },
            "2024": {
                "UParTAK4": {
                    "loose": 0.0246,
                    "medium": 0.1272,
                    "tight": 0.4648,
                },
                "particle_net": {
                    # FIXME: placeholder values, need to be updated when official values are available
                    "loose": 0.0246,
                    "medium": 0.1272,
                    "tight": 0.4648,
                },
            }
        }
    }

    return DotDict.wrap(wp_values[run][tag])


def toptag_params(
        config: od.Config,
) -> DotDict:
    """
    Returns the top-tagging parameters for the given config.
    Top-tagging working points from
    2017 (DeepAK8, 1% mistag rate):
        https://twiki.cern.ch/twiki/bin/viewauth/CMS/DeepAK8Tagging2018WPsSFs?rev=4
    Particle Net (2022, from JMAR presentation 21.10.24 (slide 10)):
        https://indico.cern.ch/event/1459087/contributions/6173396/attachments/2951723/5188840/SF_Run3.pdf
    Particle Net (2023, from JMAR presentation 08.09.25 (slide 3)):
        https://indico.cern.ch/event/1582638/contributions/6670184/subcontributions/569885/attachments/3130712/5553828/WP%20JMAR%20presentation.pdf  # noqa: E501
    """
    run = config.campaign.x.run
    tag = config.x.cpn_tag

    wp_values = {
        2: {
            "2017": {
                "deepak8": {
                    # regular tagger
                    "top": 0.725,
                    "w": 0.925,
                    # mass-decorrelated tagger
                    "top_md": 0.344,
                    "w_md": 0.739,
                },
            },
        },
        3: {
            "2022preEE": {
                "particle_net": {
                    "medium": 0.683,
                    "tight": 0.858,
                    "very_tight": 0.979,
                },
            },
            "2022postEE": {
                "particle_net": {
                    "medium": 0.698,
                    "tight": 0.866,
                    "very_tight": 0.980,
                },
            },
            "2023preBPix": {
                "particle_net": {
                    "medium": 0.655,
                    "tight": 0.835,
                    "very_tight": 0.976,
                },
            },
            "2023postBPix": {
                "particle_net": {
                    "medium": 0.639,
                    "tight": 0.821,
                    "very_tight": 0.974,
                },
            },
            "2024": {
                # FIXME: placeholder values, need to be updated when official values are available
                "particle_net": {
                    "medium": 0.639,
                    "tight": 0.821,
                    "very_tight": 0.974,
                },
                # FIXME: placeholder values, need to be updated when official values are available
                "GloParTv3": {
                    "medium": 0.639,
                    "tight": 0.821,
                    "very_tight": 0.974,
                }
            },
        },
    }

    return DotDict.wrap(wp_values[run][tag])
