
"""
# impacts_exp <label> <expectSignal> <rMin> <rMax> [job_mode]
#
# job_mode: "local" (default) runs --doFits interactively with --parallel 4.
# "condor" submits --doFits to HTCondor instead (see CONDOR_* config above).
# Unless CONDOR_DRY_RUN=true, this function BLOCKS until the condor jobs finish
# (polling via wait_for_condor_cluster) and then automatically runs the JSON
# collection step -- safe to leave running unattended either way. With
# CONDOR_DRY_RUN=true, it only writes the .sub/.sh files and returns without
# collecting; you'd then submit and collect manually (see impacts_collect_json).
# --doInitialFit always runs locally either way (it's a single fast fit).
#
# NOTE: for the "sens" use case, call get_r_range FIRST to fill R_50/R_MIN/R_MAX --
# impacts_exp no longer calls it internally. See EXECUTION FLOW at the bottom for
# the required call order; `set -u` at the top of this script will abort loudly
# with "R_50: unbound variable" (or similar) if you forget.
#
# Two intended uses:
#   impacts_exp bkgonly 0        -10       10                  # background-only expected impacts
#   impacts_exp sens    ${R_50}  ${R_MIN}  ${R_MAX}  condor     # impacts at expected-sensitivity signal strength
#
# NOTE: the -n tag is deliberately IDENTICAL across the initial fit, per-nuisance
# fits, and json collection steps for a given label -- combineTool.py -M Impacts
# ties those three calls together via that shared name, so they must match.
impacts_exp() {
    local label=$1
    local expect_signal=$2
    local r_min=$3
    local r_max=$4
    local job_mode=${5:-local}

    if [[ -z "$label" || -z "$expect_signal" || -z "$r_min" || -z "$r_max" ]]; then
        echo "Usage: impacts_exp <label> <expectSignal> <rMin> <rMax> [local|condor]"
        exit 1
    fi
    if [[ "$job_mode" != "local" && "$job_mode" != "condor" ]]; then
        echo "Invalid job_mode: $job_mode. Must be 'local' or 'condor'."
        exit 1
    fi

    echo "Run Impacts (${label}) with expectSignal=${expect_signal}, ${r_min} < r < ${r_max}, job_mode=${job_mode}"

    mkdir -p "${OUTPUT_DIR}/impacts"
    cd "${OUTPUT_DIR}/impacts"

    local shared_tag=".impacts_exp_${label}_${FIT_VERSION}__${INF_MODEL}"

    local common_args=(
        -v 2
        -m ${MASS}
        -t -1
        --expectSignal ${expect_signal}
        --robustFit 1
        --cminSingleNuisFit
        --cminFallbackAlgo Minuit2,Simplex,0:0.1
        --rMin ${r_min}
        --rMax ${r_max}
    )

    local initial_args=(
        "${common_args[@]}"
        --doInitialFit
        -n ${shared_tag}
    )

    local impacts_args=(
        "${common_args[@]}"
        --doFits
        -n ${shared_tag}
    )
    if [[ "$job_mode" == "condor" ]]; then
        local \
            sub_opts="+RequestRuntime=${CONDOR_RUNTIME_SEC}\nRequestMemory=${CONDOR_REQUEST_MEMORY_MB}\nRequestCpus=${CONDOR_REQUEST_CPUS}"
        if [[ -n "$CONDOR_MYPROJECT" ]]; then
            sub_opts="${sub_opts}\n+MyProject=\"${CONDOR_MYPROJECT}\""
        fi
        impacts_args+=(
            --job-mode condor
            --task-name "impacts_${label}_${FIT_VERSION}__${YEAR}"
            --merge ${CONDOR_MERGE}
            "--sub-opts='${sub_opts}'"
        )
        if [[ "$CONDOR_DRY_RUN" == "true" ]]; then
            impacts_args+=(--dry-run)
        fi
    else
        impacts_args+=(--parallel 4)
    fi

    local initial_cmd="combineTool.py -M Impacts -d ${WORKSPACE} ${initial_args[@]}"
    local impacts_cmd="combineTool.py -M Impacts -d ${WORKSPACE} ${impacts_args[@]}"
    local impacts_log="${LOG_DIR}/impacts_exp_${label}_${FIT_VERSION}__${YEAR}.log"
    run_cmd "$initial_cmd" "${LOG_DIR}/impacts_exp_${label}_initial_${FIT_VERSION}__${YEAR}.log"
    run_cmd "$impacts_cmd" "$impacts_log"

    if [[ "$job_mode" == "condor" ]]; then
        if [[ "$CONDOR_DRY_RUN" == "true" ]]; then
            echo "CONDOR_DRY_RUN=true: condor_impacts_${label}_${FIT_VERSION}__${YEAR}.sub was written but \
                NOT submitted (in ${OUTPUT_DIR}/impacts)."
            echo "Inspect it, submit manually (condor_submit ...), then once done run: impacts_collect_json ${label}"
            cd "${BASE_DIR}"
            return
        fi

        local cluster_id
        cluster_id=$(grep -oP 'submitted to cluster \\K[0-9]+' "$impacts_log" | tail -1 || true)
        if [[ -z "$cluster_id" ]]; then
            echo "ERROR: could not determine condor cluster ID from ${impacts_log}. Check it manually, then \
                run: impacts_collect_json ${label}" >&2
            exit 1
        fi
        wait_for_condor_cluster "$cluster_id" "$label"
    fi

    impacts_collect_json "$label"
}

# impacts_collect_json <label> -- run once doFits (local or condor) has finished for that label.
impacts_collect_json() {
    local label=$1
    if [[ -z "$label" ]]; then
        echo "Usage: impacts_collect_json <label>"
        exit 1
    fi

    mkdir -p "${OUTPUT_DIR}/impacts"
    cd "${OUTPUT_DIR}/impacts"

    local shared_tag=".impacts_exp_${label}_${FIT_VERSION}__${INF_MODEL}"
    local json_file="${OUTPUT_DIR}/impacts_exp_${label}_${FIT_VERSION}__${INF_MODEL}.json"
    local json_args=(
        -v 2
        -m ${MASS}
        -t -1
        -o ${json_file}
        -n ${shared_tag}
    )
    local json_cmd="combineTool.py -M Impacts -d ${WORKSPACE} ${json_args[@]}"
    run_cmd "$json_cmd" "${LOG_DIR}/impacts_exp_${label}_json_${FIT_VERSION}__${YEAR}.log"

    if [[ -f "$json_file" ]]; then
        cleanup_impacts_files "$label"
    else
        echo "WARNING: ${json_file} was not created -- skipping cleanup so the raw fit outputs are still \
            there to debug." >&2
    fi

    cd "${BASE_DIR}"
}

# cleanup_impacts_files <label> -- deletes the per-nuisance .root outputs and condor
# .out/.err for that label (their contents are already folded into the collected
# JSON by this point). Keeps: the JSON, the plots, and the condor .log/.sub/.sh
# (small, and .log is what CONDOR_EST_SEC_PER_FIT timing gets read from later).
# Gated by CLEANUP_IMPACTS_FILES -- set false to inspect a specific fit's output.
cleanup_impacts_files() {
    local label=$1
    local dir="${OUTPUT_DIR}/impacts"
    if [[ "$CLEANUP_IMPACTS_FILES" != "true" ]]; then
        echo "CLEANUP_IMPACTS_FILES=false: leaving intermediate files for '${label}' in ${dir}"
        return
    fi
    echo "Cleaning up intermediate .root/.out/.err files for '${label}' in ${dir}"
    rm -f "${dir}"/*"${label}_${FIT_VERSION}__${INF_MODEL}"*.root
    rm -f "${dir}"/impacts_"${label}_${FIT_VERSION}__${YEAR}".*.out
    rm -f "${dir}"/impacts_"${label}_${FIT_VERSION}__${YEAR}".*.err
}

# Plot impacts from json file. plot_impacts_exp <label> -- label must match a prior impacts_exp <label> ... call
plot_impacts_exp() {
    local label=$1
    if [[ -z "$label" ]]; then
        echo "Usage: plot_impacts_exp <label>"
        exit 1
    fi

    mkdir -p ${OUTPUT_DIR}/impacts
    cd ${OUTPUT_DIR}/impacts
    local args=(
        -o exp_${label}_${FIT_VERSION}
        -i ${OUTPUT_DIR}/impacts_exp_${label}_${FIT_VERSION}__${INF_MODEL}.json
        --summary
        # --cms-label 'Private Work'
        # --per-page 12
        --height 600
        --left-margin 0.4
        --label-size 0.03
        # -t ${BASE_DIR}/rename_impact_params.json  # optional, to rename parameters in the plot (adjust as needed)
    )
    local cmd="plotImpacts.py ${args[@]}"
    run_cmd "$cmd" "${LOG_DIR}/impacts_exp_${label}_plot_${FIT_VERSION}.log"
    cd ${BASE_DIR}
}
"""
from __future__ import annotations

import os

import law
import luigi
from columnflow.tasks.framework.base import Requirements

from mtt.tasks.inference.combine_base import CombineBaseTask
from mtt.tasks.inference.inference_base import InferenceBaseTask
from mtt.tasks.inference.limits import AsymptoticLimits
from mtt.tasks.inference.mixins import (
    CascadeMinimizerOptionsMixin,
    CommonIOOptionsMixin,
    CommonMiscOptionsMixin,
    CommonStatsOptionsMixin,
    FitterAlgoOptionsMixin,
)

logger = law.logger.get_logger(__name__)


# background-only impacts use the same signal strength and range as the bash workflow
# (impacts_exp bkgonly 0 -10 10)
BKGONLY_EXPECT_SIGNAL = 0.0
BKGONLY_R_RANGE = (-10.0, 10.0)

# fallback for the "sens" mode when the limits to derive the range from are not available
FALLBACK_EXPECT_SIGNAL = 1.0
FALLBACK_R_RANGE = (0.0, 10.0)


def read_expected_limits(limits_csv: str) -> dict[str, float]:
    """
    Reads a limits CSV (lines of ``limit_type,limit_value``, see ``AsymptoticLimits.store_limits``)
    into a dictionary, e.g. ``{"expected_50": 3.2, "expected_97.5": 7.9}``.
    """
    limits = {}
    with open(limits_csv) as f:
        for line in f:
            key, _, value = line.strip().partition(",")
            try:
                limits[key] = float(value)
            except ValueError:
                continue  # header or malformed line
    return limits


def get_fit_settings(
    impacts_mode: str,
    limits: dict[str, float],
    default_settings: tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Returns the tuple ``(expectSignal, rMin, rMax)`` to use for the impacts fits in *impacts_mode*.

    - ``default``: the values given in *default_settings*
    - ``bkgonly``: background-only fits in a fixed range, as in the bash workflow
    - ``sens``: signal strength at the median expected limit, with a range derived from the 97.5% limit

    When the limits needed for ``sens`` are missing, a warning is emitted and a fallback is used.
    """
    if impacts_mode == "default":
        return default_settings
    if impacts_mode == "bkgonly":
        return (BKGONLY_EXPECT_SIGNAL, *BKGONLY_R_RANGE)
    if impacts_mode == "sens":
        if "expected_50" not in limits or "expected_97.5" not in limits:
            logger.warning(
                "could not find expected_50 and expected_97.5 limits, using "
                f"r={FALLBACK_EXPECT_SIGNAL} and the range {FALLBACK_R_RANGE} as a fallback",
            )
            return (FALLBACK_EXPECT_SIGNAL, *FALLBACK_R_RANGE)
        r_max = 4 * limits["expected_97.5"]
        return (limits["expected_50"], -0.3 * r_max, r_max)
    raise ValueError(f"Invalid impacts_mode: {impacts_mode} ['bkgonly', 'sens', 'default'].")


class _Impacts(
    CombineBaseTask,
    CommonStatsOptionsMixin,
    CommonIOOptionsMixin,
    CascadeMinimizerOptionsMixin,
    CommonMiscOptionsMixin,
    FitterAlgoOptionsMixin,
):
    """
    Base class for the Impacts task.
    """
    impacts_mode = luigi.ChoiceParameter(
        choices=["bkgonly", "sens", "default"],
        default="bkgonly",
        description="Impacts mode: background-only or sensitivity or use combine's defaults",
    )


class Impacts(
    _Impacts,
):
    """
    Run combine tool with Impacts method.
    """

    cleanup_root_files = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Cleanup intermediate .root files after impacts calculation (default: True)",
    )

    what_to_run = luigi.ChoiceParameter(
        choices=["initial_fit", "fits", "json", "all"],
        default="all",
        description="What to run: initial fit, fits, json collection, or all (default: all)",
    )

    # upstream requirements
    reqs = Requirements(
        CombineBaseTask.reqs,
        AsymptoticLimits=AsymptoticLimits,
    )

    def requires(self):
        reqs = {"workspace": self.reqs.CreateWorkspace.req(self)}
        if self.impacts_mode == "sens":
            reqs["asymptotic_limits"] = self.reqs.AsymptoticLimits.req(self)
        return reqs

    def impacts_name(self):
        name = f"impacts__{self.impacts_mode}"
        if self.impacts_mode == "default":
            name += f"__s{self.expectSignal}_r{self.rMin}_{self.rMax}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("asymptotic_limits", "impacts", self.impacts_name())
        return parts

    def output(self):
        # only declare what the selected steps actually write, otherwise the task could never be complete
        # when only some of the steps are run
        output_dict = {}
        if self.what_to_run in ["initial_fit", "all"]:
            output_dict["initial_impacts_log"] = self.target("initial_impacts.log")
        if self.what_to_run in ["fits", "all"]:
            output_dict["impacts_log"] = self.target("impacts.log")
        if self.what_to_run in ["json", "all"]:
            output_dict["impacts_json"] = self.target("impacts.json")
        return output_dict

    def build_commands(
        self,
        input_workspace: str,
        expect_signal: float,
        r_min: float,
        r_max: float,
        json_path: str | None = None,
    ) -> dict[str, str]:
        """
        Builds the combineTool.py commands of the individual steps, keyed by "initial_fit", "fits" and
        (when *json_path* is given) "json".
        """
        command_to_run = "combineTool.py -M Impacts"
        command_to_run += f" -d {input_workspace}"
        command_to_run += f" --expectSignal {expect_signal} --rMin {r_min} --rMax {r_max}"
        command_to_run += f" -n {self.impacts_mode}_m{self.mass}_{self.inference_model}"
        command_to_run += " --robustFit 1" if self.robustFit else ""
        command_to_run += " --cminSingleNuisFit" if self.cminSingleNuisFit else ""
        command_to_run += f" --cminFallbackAlgo {self.cminFallbackAlgo}" if self.cminFallbackAlgo else ""
        command_to_run += " -t -1" if self.asimov_data else ""
        command_to_run += f" -v {self.verbosity}"
        command_to_run += " -h" if self.print_help else ""
        command_to_run += f" -m {self.mass}"
        command_to_run += " --dry-run" if self.dry_run else ""

        commands = {
            "initial_fit": command_to_run + " --doInitialFit",
            "fits": command_to_run + " --doFits --parallel 4",
        }
        if json_path:
            commands["json"] = command_to_run + f" -o {json_path}"
        return commands

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        input_workspace = self.input()["workspace"]["workspace"].path
        limits = {}
        if "asymptotic_limits" in self.input():
            limits = read_expected_limits(self.input()["asymptotic_limits"]["asymptotic_limits_csv"].path)
        outputs = self.output()
        for target in outputs.values():
            target.parent.touch()

        # signal strength and range of the fits, based on the limits in the limits CSV (if needed)
        expect_signal, r_min, r_max = get_fit_settings(
            self.impacts_mode,
            limits,
            default_settings=(self.expectSignal, self.rMin, self.rMax),
        )
        logger.info(
            f"Using expected signal strength r={expect_signal}, rMin={r_min}, rMax={r_max} "
            "for impacts calculation.",
        )

        commands = self.build_commands(
            input_workspace,
            expect_signal,
            r_min,
            r_max,
            json_path=outputs["impacts_json"].path if "impacts_json" in outputs else None,
        )

        logger.info("Running impacts commands...")
        if "initial_impacts_log" in outputs:
            logger.info("Running initial fit...")
            p, outp = self.run_command(commands["initial_fit"], echo=True)
            outputs["initial_impacts_log"].dump(outp, formatter="text")
        if "impacts_log" in outputs:
            logger.info("Running fits...")
            p, outp = self.run_command(commands["fits"], echo=False)
            outputs["impacts_log"].dump(outp, formatter="text")
        if "impacts_json" in outputs:
            logger.info("Running JSON collection...")
            p, outp = self.run_command(commands["json"], echo=True)

        logger.info("Finished running impacts commands.")

        # optionally cleanup intermediate .root files
        if self.cleanup_root_files:
            workdir = os.getcwd()
            logger.info(f"Cleaning up intermediate .root files in {workdir}")
            for file in os.listdir(workdir):
                if file.endswith(".root") and file.startswith("higgsCombine_"):
                    file_path = os.path.join(workdir, file)
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")


class PlotImpacts(
    _Impacts,
    InferenceBaseTask,
):
    summary_plot = luigi.BoolParameter(
        default=True,
        description="Whether to produce a summary plot of the impacts (default: True)",
    )

    per_page = luigi.IntParameter(
        default=30,
        description="Number of impacts to plot per page (default: 30)",
    )

    height = luigi.IntParameter(
        default=600,
        description="Height of the impacts plot in pixels (default: 600)",
    )

    left_margin = luigi.FloatParameter(
        default=0.4,
        description="Left margin of the impacts plot (default: 0.4)",
    )

    label_size = luigi.FloatParameter(
        default=0.03,
        description="Label size of the impacts plot (default: 0.03)",
    )

    cms_label = luigi.Parameter(
        default="Private Work",
        description="CMS label to put on the impacts plot (default: 'Private Work')",
    )

    reqs = Requirements(
        Impacts.reqs,
        Impacts=Impacts,
    )

    def requires(self):
        reqs = {
            "impacts": self.reqs.Impacts.req(self),
        }
        return reqs

    def impacts_plot_name(self):
        name = f"impacts_plot__{self.impacts_mode}"
        if self.impacts_mode == "default":
            name += f"__s{self.expectSignal}_r{self.rMin}_{self.rMax}"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("impacts", "impacts_plot", self.impacts_plot_name())
        return parts

    def output(self):
        output_dict = {
            "impacts_plot_log": self.target("impacts_plot.log"),
            "impacts_plot_pdf": self.target("impacts_plot.pdf"),
        }
        if self.summary_plot:
            output_dict["impacts_plot_summary_pdf"] = self.target("impacts_plot_summary.pdf")
        return output_dict

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        input_json = self.input()["impacts"]["impacts_json"].path
        output_pdf = self.output()["impacts_plot_pdf"].path
        # remove the .pdf extension from the output_pdf for plotImpacts.py
        output_pdf_no_ext = os.path.splitext(output_pdf)[0]
        output_dirname = os.path.dirname(output_pdf)
        if not os.path.exists(output_dirname):
            logger.warning("Output path didn't exist yet.")
            os.makedirs(output_dirname)
            logger.info(f"Created output directory {output_dirname}.")

        command_to_run = "plotImpacts.py"
        command_to_run += f" -i {input_json}"
        command_to_run += f" -o {output_pdf_no_ext}"
        command_to_run += " --summary" if self.summary_plot else ""
        command_to_run += f" --height {self.height}"
        command_to_run += f" --left-margin {self.left_margin}"
        command_to_run += f" --label-size {self.label_size}"
        command_to_run += f" --per-page {self.per_page}"
        command_to_run += f" --cms-label '{self.cms_label}'"
        command_to_run += " -h" if self.print_help else ""

        logger.info("Running impacts plot command...")
        p, outp = self.run_command(command_to_run, echo=True)
        self.output()["impacts_plot_log"].dump(outp, formatter="text")
