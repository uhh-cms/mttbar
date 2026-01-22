# coding: utf-8

"""
Plot limits including comparison with CMS-AN-2019-197.
Repo with Run 2 limits:
https://gitlab.cern.ch/cms-analysis/b2g/b2g-22-006/datacards
"""

import law

import matplotlib.pyplot as plt  # matplotlib library
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl
import pandas as pd
import numpy as np

logger = law.logger.get_logger(__name__)


PATH_TO_RUN2_LIMITS = "/data/dust/user/matthiej/run2_datacards/limits/limits_{model}.csv"

# Run 2 theory lines
RUN2_THEORY_LINES = {
    "ZPrime_w1": {
        "th": np.array([
            5.83131e+01, 1.36051e+01, 4.50540e+00, 1.80866e+00,
            8.13716e-01, 3.97420e-01, 2.05510e-01, 1.10890e-01,
            6.17038e-02, 3.52336e-02, 2.05665e-02, 1.21935e-02,
            7.34662e-03, 4.46826e-03, 2.75870e-03, 1.72335e-03,
            1.09115e-03, 6.99838e-04, 4.58135e-04, 3.04742e-04,
            2.07506e-04, 1.44911e-04, 1.03407e-04, 7.60116e-05,
            5.71530e-05, 4.42244e-05, 3.49246e-05,
        ]),
        "X_th": np.array([
            0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
            3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5,
            5.75, 6, 6.25, 6.5, 6.75, 7,
        ]) * 1000,
    }
}

# currently available Run 3 limits
RUN3_LIMITS = {
    "ZPrime_w1": {
        "500": 49.8750,
        "4000": 0.0591,
        "4500": 0.0376,
        "7000": 0.0254,
    },
}


# load Run 2 limits from CSV
def load_run2_limits(model: str) -> pd.DataFrame:
    path = PATH_TO_RUN2_LIMITS.format(model=model)
    df = pd.read_csv(path)
    logger.debug(f"Loaded Run 2 limits from {path}:")
    logger.debug(f"{df}")
    return df


def plot_limits(model: str, crosssection: float = 0.1):
    # load limits
    run2_limits = load_run2_limits(model)
    run3_limits = RUN3_LIMITS[model]

    # set up plot
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(12, 8))
    hep.style.use("CMS")
    hep.cms.label(
        'Private Work',
        data=False,
        lumi="138/109",
        ax=ax,
        com="13/13.6",
    )

    # plot Run 2 theory lines
    ax.plot(
        RUN2_THEORY_LINES[model]["X_th"],
        RUN2_THEORY_LINES[model]["th"],
        color="gray",
        linestyle=":",
        label="Run 2 Theory",
    )

    # plot Run 2 limits
    ax.plot(
        run2_limits["m"],
        run2_limits["exp"],
        label="Run 2 Expected",
        color="blue",
        linestyle="--",
    )

    # plot Run 3 limits
    masses = list(float(m) for m in run3_limits.keys())
    limits = list(float(v) * crosssection for v in run3_limits.values())
    ax.scatter(
        masses,
        limits,
        label="Run 3 Expected",
        color="red",
        marker="o",
    )

    # customize plot
    ax.set_xlabel("Mass [GeV]")
    ax.set_ylabel("σ × BR (Z' → ttbar) [pb]")
    ax.set_yscale("log")
    ax.legend(title='95% CL upper limit')

    # show plot
    plt.tight_layout()
    plt.savefig(f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{model}.png")
    plt.savefig(f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{model}.pdf")


if __name__ == "__main__":
    plot_limits("ZPrime_w1", crosssection=0.1)
