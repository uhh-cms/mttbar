# coding: utf-8

"""
Selection methods for ML categorization
"""
import law

from columnflow.ml import MLModel
from columnflow.util import maybe_import
from columnflow.columnar_util import Route, set_ak_column
from columnflow.selection import Selector, selector
from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.categorization import categorizer

# from mtt.config.categories import add_categories_ml
from mtt.util import get_subclasses_deep

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def register_ml_selectors(ml_model_inst: MLModel) -> None:
    """
    Register selector functions for ML categorization.
    """

    ml_output_columns = {
        f"{ml_model_inst.cls_name}.score_{proc}"
        for proc in ml_model_inst.train_nodes.keys()
    }

    for proc in ml_model_inst.train_nodes.keys():
        @categorizer(
            uses={"events"} | ml_output_columns,
            cls_name=f"sel_dnn_{proc}",
        )
        def sel_dnn(
            self: Selector,
            events: ak.Array,
            this_output_column=f"{ml_model_inst.cls_name}.score_{proc}",
            all_output_columns=ml_output_columns,
            **kwargs,
        ) -> ak.Array:
            f"""
            Dynamically built selector for DNN category '{proc}'.
            """
            mask = ak.ones_like(events.event, dtype=bool)

            # pass if this proc score larger than other processes
            this_score = Route(this_output_column).apply(events)
            other_output_cols = all_output_columns - {this_output_column}
            for other_col in other_output_cols:
                mask = mask & (
                    this_score > Route(other_col).apply(events)
                )

            return mask


@producer(
    # uses in init, produces should not be empty
    produces={"category_ids", "mlscore.max_score"},
    ml_model_name=None,
    # version=law.config.get_expanded("analysis", "add_ml_cats_version", 5),
)
def add_ml_cats(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reproduces category ids after ML Training. Calling this producer also
    automatically adds `MLEvaluation` to the requirements.
    """
    max_score = ak.fill_none(ak.max([events.mlscore[f] for f in events.mlscore.fields], axis=0), 0)
    events = set_ak_column(events, "mlscore.max_score", max_score, value_type=np.float32)
    # category ids
    events = self[category_ids](events, **kwargs)

    return events


@add_ml_cats.requires
def add_ml_cats_reqs(self: Producer, task: law.Task, reqs: dict) -> None:
    if "ml" in reqs:
        return

    from columnflow.tasks.ml import MLTraining, MLEvaluation
    if task.pilot:
        # skip MLEvaluation in pilot, but ensure that MLTraining has already been run
        reqs["mlmodel"] = MLTraining.req(task, ml_model=self.ml_model_name)
    else:
        reqs["ml"] = MLEvaluation.req(
            task,
            ml_model=self.ml_model_name,
        )


@add_ml_cats.setup
def add_ml_cats_setup(
    self: Producer, task: law.Task, reqs: dict, inputs: dict, reader_targets: law.util.InsertableDict,
) -> None:
    # self.uses |= self[category_ids].uses
    reader_targets["mlcolumns"] = inputs["ml"]["mlcolumns"]


@add_ml_cats.init
def add_ml_cats_init(self: Producer) -> None:
    if not self.ml_model_name:
        raise ValueError(f"invalid ml_model_name {self.ml_model_name} for Producer {self.cls_name}")

    # NOTE: if necessary, we could initialize the MLModel ourselves, e.g. via:
    # MLModelMixinBase.get_ml_model_inst(self.ml_model_name, self.analysis_inst, requested_configs=[self.config_inst])

    if not self.config_inst.has_variable("mlscore.max_score"):
        self.config_inst.add_variable(
            name="mlscore.max_score",
            expression="mlscore.max_score",
            binning=(1000, 0., 1.),
            x_title="DNN max output score",
            aux={
                "rebin": 25,
            },
        )

    # add categories to config inst
    from mtt.config.categories import add_categories_ml
    logger.warning("Adding ML categories to config...")
    add_categories_ml(self.config_inst, self.ml_model_name)

    self.uses.add(category_ids)
    self.produces.add(category_ids)


# # get all the derived MLModels and instantiate a corresponding producer for each one
# from mtt.ml.base import MLClassifierBase
# ml_model_names = get_subclasses_deep(MLClassifierBase)
# logger.info(f"deriving {len(ml_model_names)} ML categorizer...")

# for ml_model_name in ml_model_names:
#     add_ml_cats.derive(f"add_ml_cats_{ml_model_name}", cls_dict={"ml_model_name": ml_model_name})
# Create ML categorization producers manually
# from mtt.ml.categories import add_ml_cats

# Create producers for your specific models
add_ml_cats_v1_mergedbkgs = add_ml_cats.derive("add_ml_cats_v1_mergedbkgs", cls_dict={
    "ml_model_name": "v1_mergedbkgs"
})

add_ml_cats_v1_AN_v12 = add_ml_cats.derive("add_ml_cats_v1_AN_v12", cls_dict={
    "ml_model_name": "v1_AN_v12"
})

logger.info("Created ML categorization producers manually")
