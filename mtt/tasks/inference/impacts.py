
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


class Impacts(
    CombineBaseTask,
    CommonStatsOptionsMixin,
    CommonIOOptionsMixin,
    CascadeMinimizerOptionsMixin,
    CommonMiscOptionsMixin,
    FitterAlgoOptionsMixin,
):
    """
    Run combine tool with Impacts method.
    """
    impacts_mode = luigi.ChoiceParameter(
        choices=["bkgonly", "sens", "default"],
        default="bkgonly",
        description="Impacts mode: background-only or sensitivity or use combine's defaults",
    )

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
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
            "asymptotic_limits": self.reqs.AsymptoticLimits.req(self),
        }
        return reqs

    def impacts_name(self):
        name = "impacts"
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("asymptotic_limits", "impacts", self.impacts_name())
        return parts

    def output(self):
        output_dict = {
            "initial_impacts_log": self.target("initial_impacts.log"),
            "impacts_log": self.target("impacts.log"),
            "impacts_json": self.target("impacts.json"),
        }
        return output_dict

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        input_workspace = self.input()["workspace"]["workspace"].path
        input_limits = self.input()["asymptotic_limits"]["asymptotic_limits_csv"].path
        output_dirname = os.path.dirname(self.output()["impacts_json"].path)
        if not os.path.exists(output_dirname):
            logger.warning("Output path didn't exist yet.")
            os.makedirs(output_dirname)
            logger.info(f"Created output directory {output_dirname}.")

        # read the limits CSV to get the expected signal strength (r) for the impacts calculation
        expected_signal = None
        r_min = None
        r_max = None
        with open(input_limits, "r") as f:
            for line in f:
                if line.startswith("expected_50"):
                    expected_signal = float(line.split(",")[1])
                elif line.startswith("expected_97.5"):
                    r_max = 4 * float(line.split(",")[1])
                    r_min = -0.3 * r_max
        if expected_signal is None:
            logger.warning(f"Could not find expected_50 limit in {input_limits}. Using r=1 as default.")
            expected_signal = 1.0
            r_min = 0.0
            r_max = 10.0
        if self.impacts_mode == "default":
            expected_signal = self.expectSignal
            r_min = self.rMin
            r_max = self.rMax
        elif self.impacts_mode == "bkgonly":
            expected_signal = 0.0
            r_min = r_min
            r_max = r_max
        elif self.impacts_mode == "sens":
            expected_signal = expected_signal
            r_min = r_min
            r_max = r_max
        else:
            raise ValueError(f"Invalid impacts_mode: {self.impacts_mode} ['bkgonly', 'sens', 'default'].")

        logger.info(
            f"Using expected signal strength r={expected_signal}, rMin={r_min}, rMax={r_max} for impacts calculation."
        )

        command_to_run = "combineTool.py -M Impacts"
        command_to_run += f" -d {input_workspace}"
        command_to_run += f" --expectSignal {expected_signal} --rMin {r_min} --rMax {r_max}"
        command_to_run += f" -n {self.impacts_mode}_m{self.mass}_{self.inference_model}"
        command_to_run += " --robustFit 1" if self.robustFit else ""
        command_to_run += " --cminSingleNuisFit" if self.cminSingleNuisFit else ""
        command_to_run += f" --cminFallbackAlgo {self.cminFallbackAlgo}" if self.cminFallbackAlgo else ""
        command_to_run += " -t -1" if self.asimov_data else ""
        command_to_run += f" -v {self.verbosity}"
        command_to_run += " -h" if self.print_help else ""
        command_to_run += f" -m {self.mass}"
        command_to_run += " --dry-run" if self.dry_run else ""

        initial_impacts_command = command_to_run + " --doInitialFit"

        impacts_command = command_to_run + " --doFits --parallel 4"

        impact_json_command = command_to_run + f" -o {self.output()['impacts_json'].path}"

        # impact_json_command = f"combineTool.py -M Impacts -d {input_workspace}"
        # impact_json_command += f" -v {self.verbosity} -m {self.mass}"
        # impact_json_command += f" -o {self.output()['impacts_json'].path}"
        # impact_json_command += f" -n {self.impacts_mode}"
        # impact_json_command += " -h" if self.print_help else ""
        # impact_json_command += " --dry-run" if self.dry_run else ""

        logger.info("Running impacts commands...")
        if self.what_to_run in ["initial_fit", "all"]:
            logger.info("Running initial fit...")
            p, outp = self.run_command(initial_impacts_command, echo=True)
            self.output()["initial_impacts_log"].dump(outp, formatter="text")
        if self.what_to_run in ["fits", "all"]:
            logger.info("Running fits...")
            p, outp = self.run_command(impacts_command, echo=False)
            self.output()["impacts_log"].dump(outp, formatter="text")
        if self.what_to_run in ["json", "all"]:
            logger.info("Running JSON collection...")
            p, outp = self.run_command(impact_json_command, echo=True)

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
        name = "impacts_plot"  # indicates that this is the impacts task
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
