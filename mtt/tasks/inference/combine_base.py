
import luigi
from columnflow.tasks.framework.base import Requirements

from mtt.tasks.inference.datacards import CombineDatacards
from mtt.tasks.inference.inference_base import InferenceBaseTask
from mtt.tasks.inference.mixins import COMBINE_OPTION_MIXINS, IMPLEMENTED_COMBINE_OPTIONS
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_unimplemented_options()

    def check_unimplemented_options(self):
        unimplemented = {}
        for cls in type(self).__mro__:
            if cls not in COMBINE_OPTION_MIXINS:
                continue
            for name, param in vars(cls).items():
                if (
                    isinstance(param, luigi.Parameter) and
                    name not in IMPLEMENTED_COMBINE_OPTIONS and
                    param._default is not luigi.parameter._no_value and
                    param.normalize(getattr(self, name)) != param.normalize(param._default)
                ):
                    unimplemented[name] = getattr(self, name)
        if unimplemented:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement the combine option(s) "
                f"{', '.join(f'{k}={v!r}' for k, v in sorted(unimplemented.items()))}; add them to the command "
                "construction and to IMPLEMENTED_COMBINE_OPTIONS in mixins.py",
            )
