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
) -> tuple[float, float, float]:
    """
    Returns the tuple ``(expectSignal, rMin, rMax)`` to use for the impacts fits in *impacts_mode*.

    - ``bkgonly``: background-only fits in a fixed range, as in the bash workflow
    - ``sens``: signal strength at the median expected limit, with a range derived from the 97.5% limit
    """
    if impacts_mode == "bkgonly":
        return (BKGONLY_EXPECT_SIGNAL, *BKGONLY_R_RANGE)
    if impacts_mode == "sens":
        if "expected_50" not in limits or "expected_97.5" not in limits:
            raise RuntimeError("limits csv is missing expected_50 / expected_97.5, cannot derive the sens fit range")
        r_max = 4 * limits["expected_97.5"]
        return (limits["expected_50"], -0.3 * r_max, r_max)
    raise ValueError(f"Invalid impacts_mode: {impacts_mode} ['bkgonly', 'sens'].")


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
    implemented_options = frozenset({
        "mass", "robustFit", "cminSingleNuisFit", "cminFallbackAlgo",
        "rMin", "rMax", "expectSignal",
    })

    impacts_mode = luigi.ChoiceParameter(
        choices=["bkgonly", "sens"],
        default="bkgonly",
        description="Impacts mode: background-only or sensitivity.",
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.impacts_mode == "bkgonly" and (self.rMin, self.rMax, self.expectSignal) != (0.0, 20.0, 0.0):
            raise ValueError("rMin, rMax and expectSignal only apply in impacts_mode 'sens'")


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

    dry_run = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Print commands to the screen but do not run them.",
    )

    # upstream requirements
    reqs = Requirements(
        CombineBaseTask.reqs,
        AsymptoticLimits=AsymptoticLimits,
    )

    def requires(self):
        reqs = {"workspace": self.reqs.CreateWorkspace.req(self)}
        if self.impacts_mode == "sens":
            # the limits do not implement the fit options of the impacts, so do not forward them
            reset = {n: getattr(type(self), n)._default for n in ("cminSingleNuisFit", "cminFallbackAlgo")}
            reqs["asymptotic_limits"] = self.reqs.AsymptoticLimits.req(self, **reset)
        return reqs

    def impacts_name(self):
        name = f"impacts__{self.impacts_mode}"
        name += f"__s{self.expectSignal}_r{self.rMin}_{self.rMax}"
        if not self.asimov_data:
            name += "__data"
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
        # dedicated working directory next to the outputs, shared by the three commands
        workdir = os.path.join(os.path.dirname(next(iter(outputs.values())).path), "work")
        os.makedirs(workdir, exist_ok=True)

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
            p, outp = self.run_command(commands["initial_fit"], echo=True, cwd=workdir)
            outputs["initial_impacts_log"].dump(outp, formatter="text")
        if "impacts_log" in outputs:
            logger.info("Running fits...")
            # echo=False here because the fits are run in parallel and the output is not very readable, and too large
            p, outp = self.run_command(commands["fits"], echo=False, cwd=workdir)
            outputs["impacts_log"].dump(outp, formatter="text")
        if "impacts_json" in outputs:
            logger.info("Running JSON collection...")
            p, outp = self.run_command(commands["json"], echo=True, cwd=workdir)

        logger.info("Finished running impacts commands.")

        # optionally clean up the intermediate .root files, but only after the json step has read them
        if self.cleanup_root_files and "impacts_json" in outputs:
            logger.info(f"Cleaning up intermediate .root files in {workdir}")
            for file in os.listdir(workdir):
                if file.endswith(".root") and file.startswith("higgsCombine_"):
                    try:
                        os.remove(os.path.join(workdir, file))
                    except Exception as e:
                        logger.warning(f"Failed to delete {file}: {e}")
            try:
                os.rmdir(workdir)  # only succeeds if it is empty
            except OSError:
                pass


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
        name += f"__s{self.expectSignal}_r{self.rMin}_{self.rMax}"
        if not self.asimov_data:
            name += "__data"
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
