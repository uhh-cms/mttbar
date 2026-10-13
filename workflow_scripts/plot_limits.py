
"""
Plot limits including comparison with CMS-AN-2019-197.
Repo with Run 2 limits:
https://gitlab.cern.ch/cms-analysis/b2g/b2g-22-006/datacards
"""

import os
import re

import law
import matplotlib.pyplot as plt  # matplotlib library
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl
import numpy as np
import pandas as pd

logger = law.logger.get_logger(__name__)


PATH_TO_RUN2_LIMITS = "/data/dust/user/matthiej/run2_datacards/limits/limits_{model}.csv"  # model in RSGluon, ZprimeDM, ZPrime_w10, ZPrime_w1, ZPrime_w30

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
    },
    "ZPrime_w10": {
        "th": np.array([
            5.36522e+02, 1.31954e+02, 4.48526e+01, 1.83735e+01,
            8.47610e+00, 4.24656e+00, 2.26215e+00, 1.26395e+00,
            7.34314e-01, 4.41032e-01, 2.72788e-01, 1.73249e-01,
            1.12874e-01, 7.53710e-02, 5.15542e-02, 3.61230e-02,
            2.59114e-02, 1.90265e-02, 1.42839e-02, 1.09496e-02,
            8.55804e-03, 6.80722e-03, 5.50276e-03, 4.50943e-03,
            3.74190e-03, 3.13889e-03, 2.65864e-03
        ]),
        "X_th": np.array([
            0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
            3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5,
            5.75, 6, 6.25, 6.5, 6.75, 7,
        ]) * 1000,
    },
    "ZPrime_w30": {
        "th": np.array([
            1.32811e+03, 3.61926e+02, 1.29361e+02, 5.52047e+01,
            2.65207e+01, 1.38696e+01, 7.74166e+00, 4.55142e+00,
            2.79430e+00, 1.78062e+00, 1.17252e+00, 7.95149e-01,
            5.53916e-01, 3.95485e-01, 2.88839e-01, 2.15406e-01,
            1.63747e-01, 1.26704e-01, 9.95997e-02, 7.94496e-02,
            6.42139e-02, 5.25241e-02, 4.34250e-02, 3.62486e-02,
            3.05223e-02, 2.59064e-02, 2.21476e-02
        ]),
        "X_th": np.array([
            0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
            3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5,
            5.75, 6, 6.25, 6.5, 6.75, 7,
        ]) * 1000,
    },
}

# /data/dust/user/matthiej/mttbar_fits/outputs/260804_v8/v4_2425/zp_m400_w4__default/zzz_logs/asymptotic_limits_expected_baseline__merged.log
# BASE_PATH_RUN3_ASYMPTOTIC_LIMITS = "/data/dust/user/matthiej/mttbar_fits/outputs/{version}/{ml_model}/zp_m{mass}_w{width}__{mode}/zzz_logs/asymptotic_limits_expected_{fit_version}__{eras}.log"
BASE_PATH_RUN3_ASYMPTOTIC_LIMITS_CF = "/data/dust/user/matthiej/mttbar/data/mtt_store/new_analysis_mtt/mtt.AsymptoticLimits/run3_mtt_2024_nano_v15_new__run3_mtt_2025_nano_v15_new/calib__default_nominal/sel__ttbar_res_sel/red__default/prod__add_prod_cats__ttbar__features__7f6991e303/hist__no_trigger_weights/inf__zp_m{mass}_w{width}__v2/{version}/asymptotic_limits/asymptotic_limits.log"

# Combined CSV holding all extracted Run 3 limits, written by store_limits()
# and read back in by plot_limits_simple_with_bands().
# RUN3_LIMITS_CSV = "/data/dust/user/matthiej/mttbar/workflow_scripts/limits/run3_limits.csv"  # master file with all extracted Run 3 limits
RUN3_LIMITS_CSV_CF = "/data/dust/user/matthiej/mttbar/workflow_scripts/limits/run3_limits_cf.csv"  # master file with all extracted Run 3 limits (columnflow)


def extract_info_from_log(version, ml_model, mass, width, mode, fit_version) -> pd.DataFrame:
    """
    Extract information from log file and return as DataFrame.
    Returns columns: exp_m2s, exp_m1s, exp, exp_p1s, exp_p2s, time_elapsed
    """
    if not all([version, ml_model, mass, width, mode, fit_version]):
        raise ValueError("All parameters must be provided to extract info from log.")
    if mode == "default":
        eras = "merged"
    elif mode == "split_eras":
        eras = "2425"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # path = BASE_PATH_RUN3_ASYMPTOTIC_LIMITS.format(version=version, ml_model=ml_model, mass=mass, width=width, mode=mode, fit_version=fit_version, eras=eras)
    path = BASE_PATH_RUN3_ASYMPTOTIC_LIMITS_CF.format(version=version, mass=mass, width=width)

    logger.info(f"Extracting info from log file: {path}")
    # give warning if file does not exist
    if not os.path.exists(path):
        logger.warning(f"Log file does not exist: {path}")
        return pd.DataFrame()  # return empty DataFrame

    # Dictionary to store all percentiles
    data = {}
    data["version"] = version
    data["ml_model"] = ml_model
    data["mass"] = mass
    data["width"] = width
    data["mode"] = mode
    data["fit_version"] = fit_version

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
    elapsed_time_pattern = r"END\s+\(elapsed:\s+([0-9.]+)s\)"

    with open(path, "r") as f:
        for line in f:
            match_limit = re.search(pattern, line)
            match_elapsed = re.search(elapsed_time_pattern, line)
            if match_limit:
                percentage = match_limit.group(1)  # e.g., "50.0"
                limit_value = float(match_limit.group(2))  # e.g., 0.0308

                # Map to column name
                if percentage in percentiles_map:
                    col_name = percentiles_map[percentage]
                    data[col_name] = limit_value
                    logger.debug(f"Extracted {percentage}% limit: {limit_value} -> {col_name}")
            elif match_elapsed:
                elapsed_time = float(match_elapsed.group(1))  # e.g., 123.45
                data["time_elapsed"] = elapsed_time
                logger.debug(f"Extracted elapsed time: {elapsed_time} seconds")

    # Verify we got all expected values
    expected_cols = set(percentiles_map.values())
    found_cols = set(data.keys())

    if not expected_cols.issubset(found_cols):
        missing = expected_cols - found_cols
        logger.warning(f"Missing expected limit columns: {missing}")
    if "time_elapsed" not in found_cols:
        logger.warning("Missing 'time_elapsed' (no END line matched).")

    # Convert to DataFrame (single row)
    df = pd.DataFrame([data])
    logger.debug(f"Extracted limits DataFrame:\n{df}")

    return df


def store_limits(
    versions,
    ml_models,
    masses,
    width_percents,
    modes,
    fit_versions,
    output_path=RUN3_LIMITS_CSV_CF,
    refresh=False,
):
    """
    Extract limits for all combinations of parameters and store them
    together in a single combined CSV file at `output_path`.

    `width_percents` are widths expressed as a percentage of mass (e.g. 1, 10, 30),
    matching the directory naming convention zp_m{mass}_w{round(mass * pct / 100)}.
    Not every mass has every width percentage available on disk; missing
    combinations are skipped (extract_info_from_log logs a warning and returns
    an empty DataFrame for those).
    """
    if not refresh and os.path.exists(output_path):
        logger.info(f"Limits CSV already exists at {output_path}; skipping extraction.")
        return
    all_limits = []

    for version in versions:
        for ml_model in ml_models:
            for mass in masses:
                for width_pct in width_percents:
                    width = int(round(mass * width_pct / 100))
                    for mode in modes:
                        for fit_version in fit_versions:
                            df = extract_info_from_log(
                                version=version,
                                ml_model=ml_model,
                                mass=mass,
                                width=width,
                                mode=mode,
                                fit_version=fit_version,
                            )
                            if not df.empty:
                                df["width_pct"] = width_pct
                                all_limits.append(df)

    if not all_limits:
        logger.warning("No limits were extracted; no CSV file created.")
        return

    result_df = pd.concat(all_limits, ignore_index=True)
    result_df.to_csv(output_path, index=False)
    logger.info(f"Stored {len(result_df)} rows of limits to {output_path}")


def load_run3_limits(path=RUN3_LIMITS_CSV_CF) -> pd.DataFrame:
    """
    Load the combined Run 3 limits CSV written by store_limits().
    """
    logger.debug(f"Loading Run 3 limits from {path}")
    df = pd.read_csv(path)
    logger.debug(f"Loaded Run 3 limits:\n{df}")
    return df


def load_run2_limits(model: str) -> pd.DataFrame:
    """
    Load Run 2 limits for a given model from CSV file.
    """
    path = PATH_TO_RUN2_LIMITS.format(model=model)
    df = pd.read_csv(path)
    logger.debug(f"Loaded Run 2 limits from {path}:\n{df}")
    return df


def plot_limits_simple_with_bands(
    masses: list,
    params: dict,
    model: str,
    crosssection: float = 1.0,
    width: float = 1.0,
    run3_limits_path: str = RUN3_LIMITS_CSV_CF,
) -> None:
    """
    Plot limits for a given model and assumed signal cross-section, including error bands.
    Reads pre-extracted Run 3 limits from `run3_limits_path` (written by store_limits())
    instead of re-parsing log files.
    """
    green = '#228b22'
    yellow = '#ffcc00'
    version = params.get("version", None)
    ml_model = params.get("ml_model", None)
    mode = params.get("mode", None)
    fit_version = params.get("fit_version", None)
    # output_dir = f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{version}__{ml_model}__{mode}__{fit_version}"
    output_dir = f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{version}__cf"
    # make output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-extracted Run 3 limits and filter down to this configuration.
    # `width` here is a width *percentage* of mass (e.g. 1, 10, 30), matching
    # the width_pct column written by store_limits().
    run3_df = load_run3_limits(run3_limits_path)

    all_limits_df = run3_df[
        (run3_df["version"] == version) &
        (run3_df["ml_model"] == ml_model) &
        (run3_df["mode"] == mode) &
        (run3_df["fit_version"] == fit_version) &
        (run3_df["mass"].isin(masses)) &
        (run3_df["width_pct"] == int(round(width)))
    ].sort_values("mass").reset_index(drop=True)

    if all_limits_df.empty:
        logger.warning(
            f"No Run 3 limits found in {run3_limits_path} for "
            f"version={version}, ml_model={ml_model}, mode={mode}, fit_version={fit_version}"
        )
        return

    all_limits_df["m"] = all_limits_df["mass"]  # keep "m" column name used below
    logger.debug(f"Filtered Run 3 limits:\n{all_limits_df}")

    # load Run 2 limits
    run2_limits = load_run2_limits(model)
    logger.debug(f"Loaded Run 2 limits for model '{model}':\n{run2_limits}")

    # set up plot
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(12, 8))
    hep.style.use("CMS")
    hep.cms.label(
        'Private Work',
        data=False,
        lumi="138/220",
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

    # plot Run 2 error bands
    ax.fill_between(
        run2_limits["m"] / 1000,  # convert to TeV
        run2_limits["m2"],
        run2_limits["p2"],
        color=yellow,
        label="Run 2 68% Expected",
        zorder=1,
    )
    ax.fill_between(
        run2_limits["m"] / 1000,  # convert to TeV
        run2_limits["m1"],
        run2_limits["p1"],
        color=green,
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
        label="Run 3 95% Expected",
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
    plt.savefig(f"{output_dir}/{model}__bands.png")
    plt.savefig(f"{output_dir}/{model}__bands.pdf")
    logger.info(f"Saved plot to {output_dir}/{model}__bands.png and .pdf")

    # close plot to free memory
    plt.close()


def plot_run3_limits_only(
    masses: list,
    params: dict,
    model: str,
    crosssection: float = 1.0,
    width: float = 1.0,
    run3_limits_path: str = RUN3_LIMITS_CSV_CF,
) -> None:
    """
    Plot Run 3 limits only (no Run 2 comparison), for a given model and assumed signal cross-section.
    Reads pre-extracted Run 3 limits from `run3_limits_path` (written by store_limits())
    instead of re-parsing log files.
    """
    green = '#228b22'
    yellow = '#ffcc00'
    version = params.get("version", None)
    ml_model = params.get("ml_model", None)
    mode = params.get("mode", None)
    fit_version = params.get("fit_version", None)
    # output_dir = f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{version}__{ml_model}__{mode}__{fit_version}"
    output_dir = f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{version}__cf"
    # make output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-extracted Run 3 limits and filter down to this configuration.
    # `width` here is a width *percentage* of mass (e.g. 1, 10, 30), matching
    # the width_pct column written by store_limits().
    run3_df = load_run3_limits(run3_limits_path)

    all_limits_df = run3_df[
        (run3_df["version"] == version) &
        (run3_df["ml_model"] == ml_model) &
        (run3_df["mode"] == mode) &
        (run3_df["fit_version"] == fit_version) &
        (run3_df["mass"].isin(masses)) &
        (run3_df["width_pct"] == int(round(width)))
    ].sort_values("mass").reset_index(drop=True)

    if all_limits_df.empty:
        logger.warning(
            f"No Run 3 limits found in {run3_limits_path} for "
            f"version={version}, ml_model={ml_model}, mode={mode}, fit_version={fit_version}"
        )
        return

    all_limits_df["m"] = all_limits_df["mass"]  # keep "m" column name used below
    logger.debug(f"Filtered Run 3 limits:\n{all_limits_df}")

    # set up plot
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(12, 8))
    hep.style.use("CMS")
    hep.cms.label(
        'Private Work',
        data=False,
        lumi="138/220",
        ax=ax,
        com="13/13.6",
    )

    # plot Run 2 theory lines TODO: find Run 3 theory lines for comparison
    theory_line, = ax.plot(
        RUN2_THEORY_LINES[model]["X_th"] / 1000,  # convert to TeV
        RUN2_THEORY_LINES[model]["th"],
        color="gray",
        linestyle=":",
        zorder=5,
    )

    # plot Run 3 limits
    ax.plot(
        all_limits_df["mass"] / 1000,  # convert to TeV
        all_limits_df["exp"] * crosssection,
        label="Run 3 Expected",
        color="blue",
        linestyle="--",
        zorder=7,
    )
    logger.debug(f"Plotted Run 3 limits for model '{model}'")

    # plot Run 2 error bands
    ax.fill_between(
        all_limits_df["mass"] / 1000,  # convert to TeV
        all_limits_df["exp_m2s"] * crosssection,
        all_limits_df["exp_p2s"] * crosssection,
        color=yellow,
        label="Run 3 68% Expected",
        zorder=1,
    )
    ax.fill_between(
        all_limits_df["mass"] / 1000,  # convert to TeV
        all_limits_df["exp_m1s"] * crosssection,
        all_limits_df["exp_p1s"] * crosssection,
        color=green,
        label="Run 3 95% Expected",
        zorder=3,
    )

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
    plt.savefig(f"{output_dir}/run3__{model}__bands.png")
    plt.savefig(f"{output_dir}/run3__{model}__bands.pdf")
    logger.info(f"Saved plot to {output_dir}/run3__{model}__bands.png and .pdf")

    # close plot to free memory
    plt.close()


def plot_elapsed_time(params: dict, run3_limits_path: str = RUN3_LIMITS_CSV_CF) -> None:
    """
    Plot elapsed time for all masses in the given configuration.
    Reads pre-extracted Run 3 limits from `run3_limits_path` (written by store_limits())
    instead of re-parsing log files.
    """
    version = params.get("version", None)
    ml_model = params.get("ml_model", None)
    mode = params.get("mode", None)
    fit_version = params.get("fit_version", None)
    output_dir = f"/data/dust/user/matthiej/mttbar/workflow_scripts/limits/{version}__{ml_model}__{mode}__{fit_version}"
    # make output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-extracted Run 3 limits and filter down to this configuration.
    run3_df = load_run3_limits(run3_limits_path)

    all_limits_df = run3_df[
        (run3_df["version"] == version) &
        (run3_df["ml_model"] == ml_model) &
        (run3_df["mode"] == mode) &
        (run3_df["fit_version"] == fit_version)
    ].sort_values("mass").reset_index(drop=True)

    if all_limits_df.empty:
        logger.warning(
            f"No Run 3 limits found in {run3_limits_path} for "
            f"version={version}, ml_model={ml_model}, mode={mode}, fit_version={fit_version}"
        )
        return

    # set up plot
    plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(15, 8))
    hep.style.use("CMS")
    hep.cms.label(
        'Private Work',
        data=False,
        lumi="220",
        ax=ax,
        com="13.6",
    )

    # `all_limits_df` typically holds all three width_pct values (1/10/30)
    # together, so plot each width as its own series/color rather than
    # trying to pick one color for the whole (mixed) DataFrame.
    color_map = {1: "blue", 10: "orange", 30: "green"}
    symbols = {1: "v", 10: "s", 30: "^"}
    for width_pct, group_df in all_limits_df.groupby("width_pct"):
        group_df = group_df.sort_values("mass")
        color = color_map.get(width_pct, "black")
        marker = symbols.get(width_pct, "o")
        ax.scatter(
            group_df["mass"],
            group_df["time_elapsed"],
            label=f"width = {width_pct}%",
            color=color,
            marker=marker,
            zorder=10,
            s=100,
            linewidths=2.0,
        )

    # thin vertical line at each mass
    for mass in all_limits_df["mass"].unique():
        ax.axvline(x=mass, color="gray", linestyle=":", linewidth=0.5, zorder=1)

    ax.set_xlabel("Mass [GeV]")
    ax.set_ylabel("Elapsed time [s]")
    ax.legend(title="Signal width", loc="best", frameon=False, fontsize=18, title_fontsize=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/elapsed_time.png")
    plt.savefig(f"{output_dir}/elapsed_time.pdf")
    logger.info(f"Saved elapsed time plot to {output_dir}/elapsed_time.png and .pdf")

    # close plot to free memory
    plt.close()


if __name__ == "__main__":
    # Example usage
    # versions = ["260804_v8", "260804_v8_simplified"]
    versions = ["260804_v8"]
    ml_models = ["v4_2425"]
    # masses actually available on disk (zp_m* directories); 1800 has only
    # 1%/10% widths and 3000 only has 10% — store_limits() will simply skip
    # the combinations that don't exist rather than erroring.
    masses = [
        400, 500, 600, 700, 800, 900,
        1000, 1200, 1400, 1600, 1800,
        2000, 2500, 3000, 3500, 4000,
        4500, 5000, 6000, 7000, 8000,
        9000,
    ]
    width_percents = [1, 10, 30]  # width as percentage of mass
    modes = [
        "default",
        "split_eras",
    ]
    fit_versions = ["baseline"]

    # Step 1: parse all log files once and store the combined result in a CSV
    store_limits(versions, ml_models, masses, width_percents, modes, fit_versions, refresh=True)

    # Step 2: read the CSV back in and make plots
    for mode in [
        "default",
        # "split_eras",
    ]:
        params = {
            "version": "260804_v8",
            "ml_model": "v4_2425",
            "mode": mode,
            "fit_version": "baseline",
        }
        # plot_elapsed_time(params)
        for model in ["ZPrime_w1", "ZPrime_w10", "ZPrime_w30"]:
            if model == "ZPrime_w1":
                width = 1
            elif model == "ZPrime_w10":
                width = 10
            elif model == "ZPrime_w30":
                width = 30
            crosssection = 0.1  # assumed signal cross-section in pb
            plot_limits_simple_with_bands(masses, params, model, crosssection, width)
            plot_run3_limits_only(masses, params, model, crosssection, width)
