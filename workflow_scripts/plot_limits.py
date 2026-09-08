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
PATH_TO_RUN3_LIMITS = "/data/dust/user/matthiej/mttbar/workflow_scripts/limits/run3_{model}_limits.csv"

PATH_TO_RUN3_OUTPUTS = "/data/dust/user/matthiej/mttbar_fits/outputs/{version}/{ml_model}/{inf_model}__m{mass}_w{width}/zzz_logs/asymptotic_limits_{fit_version}.log"
# /data/dust/user/matthiej/mttbar_fits/outputs/251209_v5/v3_AN_v12/an_v12_simplified__m7000_w70/zzz_logs/asymptotic_limits_baseline.log
# /data/dust/user/matthiej/mttbar_fits/outputs/251209_v5/v3_AN_v12/an_v12_simplified__m500_w5/zzz_logs/asymptotic_limits_baseline.log

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


def extract_limits_from_log(version, ml_model, inf_model, mass, width, fit_version) -> pd.DataFrame:
    """
    Extract all expected limits from log file and return as DataFrame.
    Returns columns: exp_m2s, exp_m1s, exp, exp_p1s, exp_p2s
    """
    path = PATH_TO_RUN3_OUTPUTS.format(version=version, ml_model=ml_model, inf_model=inf_model, mass=mass, width=width, fit_version=fit_version)
    import re

    logger.info(f"Extracting limits from log file: {path}")

    # Dictionary to store all percentiles
    limits_data = {}

    # Define the expected percentiles and their column names
    percentiles_map = {
        "2.5": "exp_m2s",    # -2 sigma (2.5%)
        "16.0": "exp_m1s",   # -1 sigma (16.0%)  
        "50.0": "exp",       # median (50.0%)
        "84.0": "exp_p1s",   # +1 sigma (84.0%)
        "97.5": "exp_p2s",   # +2 sigma (97.5%)
    }

    # Regular expression to parse limit lines
    # Matches: "Expected 50.0%: r < 0.0308"
    pattern = r"Expected\s+(\d+\.?\d*)%:\s+r\s*<\s*([0-9.e+-]+)"

    with open(path, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                percentage = match.group(1)  # e.g., "50.0"
                limit_value = float(match.group(2))  # e.g., 0.0308

                # Map to column name
                if percentage in percentiles_map:
                    col_name = percentiles_map[percentage]
                    limits_data[col_name] = limit_value
                    logger.debug(f"Extracted {percentage}% limit: {limit_value} -> {col_name}")

    # Verify we got all expected values
    expected_cols = set(percentiles_map.values())
    found_cols = set(limits_data.keys())

    if not expected_cols.issubset(found_cols):
        missing = expected_cols - found_cols
        logger.warning(f"Missing expected limit columns: {missing}")

    # Convert to DataFrame (single row)
    df = pd.DataFrame([limits_data])
    logger.debug(f"Extracted limits DataFrame:\n{df}")
    return df


def plot_limits_with_bands(model: str, crosssection: float = 1.0, width: float = 1.0) -> None:
    green = '#228b22'
    yellow = '#ffcc00'
    masses = [
        500,
        4000,
        4500,
        7000,
    ]
    # Extract limits for all masses
    limits_dfs = []
    for mass in masses:
        df = extract_limits_from_log(
            version="251209_v5",
            ml_model="v1_AN_v12",
            inf_model="an_v12_simplified",
            mass=mass,
            width=int(mass * width / 100),  # width as percentage of mass
            fit_version="baseline",
        )
        df["m"] = mass  # Add mass
        limits_dfs.append(df)

    # Concatenate all limits DataFrames
    all_limits_df = pd.concat(limits_dfs, ignore_index=True)
    logger.debug(f"All extracted limits:\n{all_limits_df}")

    # set up plot
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.text(
        200, 170,
        "Private work (CMS simulation)",
        fontsize=24,
        verticalalignment='top',
        fontproperties="Tex Gyre Heros:italic"
    )
    ax.text(
        6500, 170,
        r"109 fb$^{-1}$ (13.6 TeV)",
        fontsize=24,
        verticalalignment='top',
        fontproperties="Tex Gyre Heros"
    )

    # plot Run 3 limits with error bands
    ax.fill_between(
        all_limits_df["m"],
        all_limits_df["exp_m2s"] * crosssection,
        all_limits_df["exp_p2s"] * crosssection,
        color=green,
        label="Run 3 68% Expected",
        zorder=3,
    )
    ax.fill_between(
        all_limits_df["m"],
        all_limits_df["exp_m1s"] * crosssection,
        all_limits_df["exp_p1s"] * crosssection,
        color=yellow,
        label="Run 3 95% Expected",
        zorder=5,
    )
    ax.scatter(
        all_limits_df["m"],
        all_limits_df["exp"] * crosssection,
        color='k',
        marker='o',
        label="Run 3 Expected",
        zorder=10,
    )

    # plot Run 2 theory lines
    ax.plot(
        RUN2_THEORY_LINES[model]["X_th"],
        RUN2_THEORY_LINES[model]["th"],
        color="gray",
        linestyle=":",
        label="Run 2 Theory",
        zorder=9,
    )

    run2_limits = load_limits_simple(model, run=2)

    # plot Run 2 limits
    ax.plot(
        run2_limits["m"],
        run2_limits["exp"],
        label="Run 2 Expected\n(CMS-PAS-B2G-22-006)",
        color="blue",
        linestyle="--",
        zorder=8,
    )

    # customize plot
    ax.set_xlabel("Mass [GeV]")
    ax.set_ylabel("σ × BR (Z' → ttbar) [pb]")
    ax.set_yscale("log")
    ax.legend(
        title='95% CL upper limit',
        bbox_to_anchor=(0.98, 0.90),  # x=0.98 (right edge), y=0.75 (75% up)
        loc='upper right'
    )
    ax.set_ylim(1e-5, 300)

    # show plot
    plt.tight_layout()
    plt.savefig(f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{model}__full.png")
    plt.savefig(f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{model}__full.pdf")


def plot_limits_simple_with_bands(model: str, crosssection: float = 1.0, width: float = 1.0) -> None:
    """
    Plot limits for a given model and assumed signal cross-section, including error bands.
    """
    green = '#228b22'
    yellow = '#ffcc00'
    masses = [
        500,
        4000,
        4500,
        7000,
    ]
    # Extract limits for all masses
    limits_dfs = []
    for mass in masses:
        df = extract_limits_from_log(
            version="251209_v5",
            ml_model="v1_AN_v12",
            inf_model="an_v12_simplified",
            mass=mass,
            width=int(mass * width / 100),  # width as percentage of mass
            fit_version="baseline",
        )
        df["m"] = mass  # Add mass
        limits_dfs.append(df)

    # Concatenate all limits DataFrames
    all_limits_df = pd.concat(limits_dfs, ignore_index=True)
    logger.debug(f"All extracted limits:\n{all_limits_df}")
    # load limits
    run2_limits = load_limits_simple(model, run=2)
    logger.debug(f"Loaded Run 2 limits for model '{model}':\n{run2_limits}")

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
    theory_line, = ax.plot(
        RUN2_THEORY_LINES[model]["X_th"] / 1000,  # convert to TeV
        RUN2_THEORY_LINES[model]["th"],
        color="gray",
        linestyle=":",
        zorder=5,
    )
    logger.debug(f"Loaded Run 2 limits for model '{model}':\n{run2_limits}")

    # plot Run 2 limits
    ax.plot(
        run2_limits["m"] / 1000,  # convert to TeV
        run2_limits["exp"],
        label="Run 2 Expected\n(CMS-PAS-B2G-22-006)",
        color="blue",
        linestyle="--",
        zorder=7,
    )
    logger.debug(f"Plotted Run 2 limits for model '{model}'")

    # plot Run 3 limits with error bands
    ax.fill_between(
        run2_limits["m"] / 1000,  # convert to TeV
        run2_limits["m2"],
        run2_limits["p2"],
        color=green,
        label="Run 2 68% Expected",
        zorder=1,
    )
    ax.fill_between(
        run2_limits["m"] / 1000,  # convert to TeV
        run2_limits["m1"],
        run2_limits["p1"],
        color=yellow,
        label="Run 2 95% Expected",
        zorder=3,
    )
    ax.errorbar(
        all_limits_df["m"] / 1000,  # convert to TeV
        all_limits_df["exp"] * crosssection,
        yerr=[
            (all_limits_df["exp"] - all_limits_df["exp_m2s"]) * crosssection,
            (all_limits_df["exp_p2s"] - all_limits_df["exp"]) * crosssection,
        ],
        label="Run 3 68% Expected",
        color='k',
        marker='o',
        zorder=10,
        linestyle='none',
        markersize=10,
        elinewidth=3,
        capsize=6,
        capthick=2.5,
        markerfacecolor='black',
        # markeredgecolor='white',   # white outline
        markeredgewidth=2.0,       # thick edge
        ecolor='black',
    )
    logger.debug(f"Plotted Run 3 limits with error bars for model '{model}'")

    # customize plot
    ax.set_xlabel("Mass [TeV]")
    ax.set_ylabel("σ × BR (Z' → ttbar) [pb]")
    ax.set_yscale("log")
    # ax.legend(title='95% CL upper limit')
    # --- Theory legend ---
    theory_legend = ax.legend(
        handles=[theory_line],
        labels=["Run 2 Theory"],
        title="Theory prediction",
        loc="upper center",
        frameon=False,
        fontsize=18,
        title_fontsize=20,
    )

    ax.add_artist(theory_legend)  # keep it

    # --- Main legend ---
    ax.legend(
        title="95% CL upper limit",
        loc="upper right",
        frameon=False,
        fontsize=18,
        title_fontsize=20,
    )

    # show plot
    plt.tight_layout()
    plt.savefig(f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{model}__bands.png")
    plt.savefig(f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{model}__bands.pdf")

    # close plot to free memory
    plt.close()


# load limits from CSV
def load_limits_simple(model: str, run: int) -> pd.DataFrame:
    """
    Load limits for a given model and Run from CSV file.
    """
    if run == 2:
        path = PATH_TO_RUN2_LIMITS.format(model=model)
    elif run == 3:
        path = PATH_TO_RUN3_LIMITS.format(model=model)
        logger.debug(f"Loading Run 3 limits for model '{model}': {path}")
    else:
        raise ValueError(f"Unknown run: {run}")
    df = pd.read_csv(path)
    logger.debug(f"Loaded Run {run} limits from {path}:")
    logger.debug(f"{df}")
    return df


def plot_limits_simple(model: str, crosssection: float = 1.0) -> None:
    """
    Plot limits for a given model and assumed signal cross-section.
    """
    # load limits
    run2_limits = load_limits_simple(model, run=2)
    run3_limits = load_limits_simple(model, run=3)

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
    ax.scatter(
        run3_limits["m"],
        run3_limits["exp"] * crosssection,
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
    # plot_limits_simple("ZPrime_w1", crosssection=0.1)
    # plot_limits_with_bands("ZPrime_w1", crosssection=0.1, width=1.0)
    plot_limits_simple_with_bands("ZPrime_w1", crosssection=0.1, width=1.0)
