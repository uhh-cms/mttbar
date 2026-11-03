
import luigi
from columnflow.tasks.framework.base import Requirements

from mtt.tasks.inference.datacards import CombineDatacards
from mtt.tasks.inference.inference_base import InferenceBaseTask
from mtt.tasks.inference.mixins import COMBINE_OPTION_MIXINS
from mtt.tasks.inference.workspace import CreateWorkspace


class CombineBaseTask(
    InferenceBaseTask,
):
    # set general parameters available for all combine commands of the form
    # 'combine -M method -t -l'

    asimov_data = luigi.BoolParameter(
        default=True,
        significant=False,
        description="Use Asimov data",
    )

    unblind = luigi.BoolParameter(
       default=False,
       significant=False,
       description="Allow fits to real data (--asimov-data False, or run mode 'observed'/'both').",
   )

    # upstream requirements
    reqs = Requirements(
        InferenceBaseTask.reqs,
        CombineDatacards=CombineDatacards,
        CreateWorkspace=CreateWorkspace,
    )

    implemented_options = frozenset()

    def requires(self):
        reqs = {
            "workspace": self.reqs.CreateWorkspace.req(self),
        }
        return reqs

    def check_unimplemented_options(self):
        unimplemented = {}
        for cls in type(self).__mro__:
            if cls not in COMBINE_OPTION_MIXINS:
                continue
            for name, param in vars(cls).items():
                if (
                    isinstance(param, luigi.Parameter) and
                    name not in self.implemented_options and
                    param._default is not luigi.parameter._no_value and
                    param.normalize(getattr(self, name)) != param.normalize(param._default)
                ):
                    unimplemented[name] = getattr(self, name)
        if unimplemented:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement the combine option(s) "
                f"{', '.join(f'{k}={v!r}' for k, v in sorted(unimplemented.items()))}; add them to the command "
                f"construction and to implemented_options of {self.__class__.__name__}",
            )
    
    def check_blinding(self):
        uses_data = not self.asimov_data or getattr(self, "run_mode", "expected") in ("observed", "both")
        if uses_data and not self.unblind:
            raise ValueError(
                f"{self.__class__.__name__} would use real data (asimov_data={self.asimov_data}, "
                f"run_mode={getattr(self, 'run_mode', None)}); pass --unblind if this is intended",
            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_unimplemented_options()
        self.check_blinding()
