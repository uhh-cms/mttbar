# coding: utf-8

"""
Selection methods for ML categorization
"""
from columnflow.ml import MLModel
from columnflow.util import maybe_import
from columnflow.columnar_util import Route
from columnflow.selection import Selector, selector

np = maybe_import("numpy")
ak = maybe_import("awkward")


def register_ml_selectors(ml_model_inst: MLModel) -> None:
    """
    Register selector functions for ML categorization.
    """

    ml_output_columns = {
        f"{ml_model_inst.cls_name}.score_{proc}"
        for proc in ml_model_inst.processes
    }

    for proc in ml_model_inst.processes:
        @selector(
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
