"""
Wrapper tasks that run an inference task for several inference models at once, e.g.

    law run mtt.PlotImpactsWrapper --inference-models zp_m1000_w10__v2,zp_m1000_w100__v2 --impacts-mode sens ...
"""

import law
import luigi

from mtt.tasks.base import MTTTask
from mtt.tasks.inference.datacards import CombineDatacards
from mtt.tasks.inference.impacts import Impacts, PlotImpacts
from mtt.tasks.inference.limits import AsymptoticLimits
from mtt.tasks.inference.workspace import CreateWorkspace


def inference_models_wrapper(require_cls, cls_name=None):
    """
    Creates a wrapper task that requires one *require_cls* instance per entry of ``--inference-models``. All other
    parameters of *require_cls* are ported to the wrapper and passed on unchanged.
    """
    class Wrapper(MTTTask, law.WrapperTask):

        inference_models = law.CSVParameter(
            default=(),
            description="names of the inference models to run, one task each; use instead of --inference-model",
            brace_expand=True,
        )

        def requires(self):
            if not self.inference_models:
                raise ValueError("no inference models given, set --inference-models")
            return {
                model: require_cls.req(self, inference_model=model)
                for model in self.inference_models
            }

    Wrapper.__name__ = cls_name or f"{require_cls.__name__}Wrapper"
    Wrapper.__module__ = __name__
    Wrapper.task_namespace = require_cls.task_namespace
    Wrapper.__doc__ = f"Runs {require_cls.__name__} for several inference models."

    # port the parameters of the wrapped task, except for the ones that are per model or per workflow branch
    skip = (
        set(dict(Wrapper.get_params())) |
        set(require_cls.interactive_params) |
        {"inference_model", "effective_workflow", "branch", "branches", "start_branch", "end_branch"}
    )
    for name, param in dict(require_cls.get_params()).items():
        if name in skip or param.visibility == luigi.parameter.ParameterVisibility.PRIVATE:
            continue
        setattr(Wrapper, name, param)

    return Wrapper


CombineDatacardsWrapper = inference_models_wrapper(CombineDatacards)
CreateWorkspaceWrapper = inference_models_wrapper(CreateWorkspace)
AsymptoticLimitsWrapper = inference_models_wrapper(AsymptoticLimits)
ImpactsWrapper = inference_models_wrapper(Impacts)
PlotImpactsWrapper = inference_models_wrapper(PlotImpacts)
