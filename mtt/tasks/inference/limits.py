from __future__ import annotations

import os
import shutil

import law

from mtt.tasks.inference.combine_base import CombineBaseTask
from mtt.tasks.inference.mixins import (
    AsymptoticLimitsMixin,
    CascadeMinimizerOptionsMixin,
    CommonIOOptionsMixin,
    CommonStatsOptionsMixin,
)

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
    implemented_options = frozenset({
        "name", "mass", "saveWorkspace", "validateModel", "run_mode",
        "rMin", "rMax", "expectSignal",
    })

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
        }
        return reqs

    def asymptotic_limits_name(self):
        name = f"asymptotic_limits__{self.run_mode}__{'asimov' if self.asimov_data else 'data'}"
        if (self.rMin, self.rMax) != (0.0, 20.0):
            name += f"__r{self.rMin}_{self.rMax}"
        if self.expectSignal:
            name += f"__s{self.expectSignal}"
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

        expected_keys = ("expected_2.5", "expected_16", "expected_50", "expected_84", "expected_97.5")
        missing = [k for k in expected_keys if k not in limits_dict]
        if missing:
            raise RuntimeError(f"limits log is missing {', '.join(missing)} in {log_path}.")

        # Write to CSV
        with open(csv_path, "w") as f:
            f.write("limit_type,limit_value\n")
            f.writelines(f"{key},{value}\n" for key, value in limits_dict.items())

        logger.info(f"Stored limits to CSV file {csv_path}.")

    def build_command(self) -> str:
        """
        Builds the combine command that runs the AsymptoticLimits method.
        """
        input_workspace = self.input()["workspace"]["workspace"].path
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
        command_to_run += f" --rMin {self.rMin} --rMax {self.rMax}"
        command_to_run += f" --expectSignal {self.expectSignal}" if self.expectSignal else ""
        return command_to_run

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        if not self.name:
            raise ValueError("name must not be empty (combine would use 'Test' and the output lookup would fail)")
        output_root = self.output()["asymptotic_limits_root"].path
        output_dirname = os.path.dirname(output_root)
        # touch the output_dirname
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname, exist_ok=True)
        workdir = os.path.join(output_dirname, "work")
        os.makedirs(workdir, exist_ok=True)

        command_to_run = self.build_command()

        p, outp = self.run_command(command_to_run, echo=True, cwd=workdir)

        self.output()["asymptotic_limits_log"].dump(outp, formatter="text")
        # move the output root file to the output location
        # /data/dust/user/matthiej/mttbar/higgsCombineTest.AsymptoticLimits.mH120.root
        name_extension = f".{self.name}" if self.name else ""
        raw_output_root = f"higgsCombine{name_extension}.AsymptoticLimits.mH{self.mass}.root"
        full_raw_output_root = os.path.join(workdir, raw_output_root)
        if not os.path.exists(full_raw_output_root):
            raise FileNotFoundError(f"Expected output root file {full_raw_output_root} not found.")
        shutil.move(full_raw_output_root, output_root)
        shutil.rmtree(workdir, ignore_errors=True)

        logger.info(f"Moved output root file {full_raw_output_root} to {output_root}.")
        self.store_limits()
