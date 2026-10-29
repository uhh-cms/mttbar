
import luigi
from columnflow.tasks.framework.base import Requirements

from mtt.tasks.inference.datacards import CombineDatacards
from mtt.tasks.inference.inference_base import InferenceBaseTask
from mtt.tasks.inference.workspace import CreateWorkspace


class CombineBaseTask(
    InferenceBaseTask,
):
    # set general parameters available for all combine commands of the form
    # 'combine -M method -t -l'
    combine_method = luigi.Parameter(
        default=None,
        significant=False,
        description="Method to run with combine tool",
    )

    asimov_data = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Use Asimov data",
    )

    job_dir = luigi.Parameter(
        default=None,
        significant=False,
        description="Path to directory containing job scripts and logs.",
    )

    merge = luigi.IntParameter(
        default=1,
        significant=False,
        description="Number of jobs to run in a single script [only affects batch submission].",
    )

    dry_run = luigi.BoolParameter(
        default=False,
        significant=False,
        description="Print commands to the screen but do not run them.",
    )

    # upstream requirements
    reqs = Requirements(
        InferenceBaseTask.reqs,      # carries CreateDatacards + MergeShiftedHistograms forward
        CombineDatacards=CombineDatacards,
        CreateWorkspace=CreateWorkspace,
    )

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
        }
        return reqs
