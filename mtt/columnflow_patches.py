# coding: utf-8

"""
Collection of patches of underlying columnflow tasks.
"""
import law

from columnflow.util import memoize
from columnflow.tasks.ml import MLTraining

logger = law.logger.get_logger(__name__)


@memoize
def patch_mltraining():
    from columnflow.tasks.framework.remote import RemoteWorkflow

    # patch the MLTraining output collection
    MLTraining.output_collection_cls = law.NestedSiblingFileCollection

    # remove unnceessary requires from MLTraining
    def workflow_requires(self):
        # cannot call super() here; hopefully we only need the workflow_requires from RemoteWorkflow
        reqs = RemoteWorkflow.workflow_requires(self)
        reqs["model"] = self.ml_model_inst.requires(self)
        return reqs

    def requires(self):
        reqs = {}
        reqs["model"] = self.ml_model_inst.requires(self)
        return reqs

    MLTraining.requires = requires
    MLTraining.workflow_requires = workflow_requires
    logger.info("patched MLTraining to use NestedSiblingFileCollection and remove unnecessary requires")


@memoize
def patch_all():

    patch_mltraining()
    logger.info("applied all columnflow patches")
