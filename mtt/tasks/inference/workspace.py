
"""
A script that calls the text2workspace combine command.

# from workflow.sh script
make_workspace() {
    local args=(
        -v 2
        -m ${MASS}
        # -o ${WORKSPACE}  # already passed explicitly in the cmd string below
        # -h
    )
    local cmd="text2workspace.py ${DATACARD} -o ${WORKSPACE} ${args[@]}"
    run_cmd "$cmd" "${LOG_DIR}/workspace_${FIT_VERSION}__${YEAR}.log"
}
"""

import os

import law
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow

from mtt.tasks.inference.datacards import CombineDatacards
from mtt.tasks.inference.inference_base import InferenceBaseTask


class CreateWorkspace(
    InferenceBaseTask,
):

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        CombineDatacards=CombineDatacards,
    )

    def requires(self):
        reqs = {
            "combined_datacard": self.reqs.CombineDatacards.req(self),
        }
        return reqs

    @property
    def workspace_name(self):
        name = "workspace__initial"  # indicates that this is the initial workspace
        return name

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts.insert_after("combined_datacard", "workspace", self.workspace_name)
        return parts

    def output(self):
        output_dict = {
            "workspace": self.target("workspace.root"),
            "t2w_log": self.target("text2workspace.log"),
        }
        return output_dict

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):

        input_datacards = self.input()["combined_datacard"]["combined_card"].path
        output_workspace = self.output()["workspace"].path
        output_dirname = os.path.dirname(output_workspace)
        # touch the output_dirname
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname, exist_ok=True)

        command_to_run = "text2workspace.py"
        command_to_run += f" {input_datacards}"
        command_to_run += f" -o {output_workspace}"
        command_to_run += f" -v {self.verbosity}"
        command_to_run += " -h" if self.print_help else ""

        p, outp = self.run_command(command_to_run, echo=True)

        self.output()["t2w_log"].dump(outp, formatter="text")
