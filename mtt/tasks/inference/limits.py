# coding: utf-8

"""
asymptotic_limits() {
    local args=(
        -v 2
        -m ${MASS}
        --validateModel
        --saveWorkspace
    )
    local cmd_blind="combine -M AsymptoticLimits ${WORKSPACE} ${args[@]} --run blind -n .asymptotic_limits_blind_${FIT_VERSION}__${INF_MODEL}"
    local cmd_expected="combine -M AsymptoticLimits ${WORKSPACE} ${args[@]} --run expected -n .asymptotic_limits_expected_${FIT_VERSION}__${INF_MODEL}"
    local cmd_observed="combine -M AsymptoticLimits ${WORKSPACE} ${args[@]} --run observed -n .asymptotic_limits_observed_${FIT_VERSION}__${INF_MODEL}"
    # run_cmd "$cmd_blind" "${LOG_DIR}/asymptotic_limits_blind_${FIT_VERSION}__${YEAR}.log"  # NOTE: takes 5x longer than expected, so don't run by default, limits don't change due to blinded datacards
    run_cmd "$cmd_expected" "${LOG_DIR}/asymptotic_limits_expected_${FIT_VERSION}__${YEAR}.log"
    # run_cmd "$cmd_observed" "${LOG_DIR}/asymptotic_limits_observed_${FIT_VERSION}__${YEAR}.log"  # NOTE: don't use!
}
"""
import law
import os
import shutil

from mtt.tasks.inference.combine_base import CombineBaseTask
from mtt.tasks.inference.mixins import (
    CommonStatsOptionsMixin,
    CommonIOOptionsMixin,
    CascadeMinimizerOptionsMixin,
    AsymptoticLimitsMixin,
)
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeShiftedHistograms
from mtt.tasks.inference.datacards import CombineDatacards

from mtt.tasks.inference.workspace import CreateWorkspace

logger = law.logger.get_logger(__name__)


class AsymptoticLimits(
    CombineBaseTask,
    CommonStatsOptionsMixin,
    CommonIOOptionsMixin,
    CascadeMinimizerOptionsMixin,
    AsymptoticLimitsMixin,
):
    """
    Run combine tool with AsymptoticLimits method.
    """

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
        }
        return reqs

    def asymptotic_limits_name(self):
        name = "asymptotic_limits"  # indicates that this is the asymptotic limits task
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("workspace", "asymptotic_limits", self.asymptotic_limits_name())
        return parts

    def output(self):
        output_dict = {
            "asymptotic_limits_root": self.target("asymptotic_limits.root"),
            "asymptotic_limits_log": self.target("asymptotic_limits.log"),
            "asymptotic_limits_csv": self.target("asymptotic_limits.csv"),
        }
        return output_dict

    def store_limits(self):
        """
        Read the limits from log file and store them in a CSV file for later plotting.
        Structure:
         -- AsymptoticLimits ( CLs ) --
        Observed Limit: r < XXX  (optional line, only if observed limits are computed)
        Expected  2.5%: r < XXX
        Expected 16.0%: r < XXX
        Expected 50.0%: r < XXX
        Expected 84.0%: r < XXX
        Expected 97.5%: r < XXX
        """
        log_path = self.output()["asymptotic_limits_log"].path
        csv_path = self.output()["asymptotic_limits_csv"].path

        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Log file {log_path} not found.")

        limits_dict = {}
        with open(log_path, "r") as f:
            for line in f:
                if "Observed Limit:" in line:
                    limits_dict["observed"] = float(line.split("<")[1].strip())
                elif "Expected  2.5%:" in line:
                    limits_dict["expected_2.5"] = float(line.split("<")[1].strip())
                elif "Expected 16.0%:" in line:
                    limits_dict["expected_16"] = float(line.split("<")[1].strip())
                elif "Expected 50.0%:" in line:
                    limits_dict["expected_50"] = float(line.split("<")[1].strip())
                elif "Expected 84.0%:" in line:
                    limits_dict["expected_84"] = float(line.split("<")[1].strip())
                elif "Expected 97.5%:" in line:
                    limits_dict["expected_97.5"] = float(line.split("<")[1].strip())

        # Write to CSV
        with open(csv_path, "w") as f:
            f.write("limit_type,limit_value\n")
            for key, value in limits_dict.items():
                f.write(f"{key},{value}\n")

        logger.info(f"Stored limits to CSV file {csv_path}.")

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        input_workspace = self.input()["workspace"]["workspace"].path
        output_root = self.output()["asymptotic_limits_root"].path
        output_dirname = os.path.dirname(output_root)
        # touch the output_dirname
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname, exist_ok=True)

        command_to_run = "combine -M AsymptoticLimits"
        command_to_run += f" {input_workspace}"
        command_to_run += f" -v {self.verbosity}"
        command_to_run += " -h" if self.print_help else ""
        command_to_run += f" --run {self.run_mode}"
        command_to_run += f" -m {self.mass}"
        command_to_run += " --saveWorkspace" if self.saveWorkspace else ""
        command_to_run += " --validateModel" if self.validateModel else ""
        command_to_run += f" --name .{self.name}" if self.name else ""
        command_to_run += " -t -1" if self.asimov_data else ""

        p, outp = self.run_command(command_to_run, echo=True)

        self.output()["asymptotic_limits_log"].dump(outp, formatter="text")
        # move the output root file to the output location
        # /data/dust/user/matthiej/mttbar/higgsCombineTest.AsymptoticLimits.mH120.root
        name_extension = f".{self.name}" if self.name else ""
        raw_output_root = f"higgsCombine{name_extension}.AsymptoticLimits.mH{self.mass}.root"
        full_raw_output_root = os.path.join(os.getcwd(), raw_output_root)
        if not os.path.exists(full_raw_output_root):
            raise FileNotFoundError(f"Expected output root file {full_raw_output_root} not found.")
        shutil.move(full_raw_output_root, output_root)

        logger.info(f"Moved output root file {full_raw_output_root} to {output_root}.")
        self.store_limits()
