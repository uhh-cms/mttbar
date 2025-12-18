# coding: utf-8

"""
production methods regarding ml stats
taken from hbw analysis.
"""

from __future__ import annotations

import functools

import law

from columnflow.production import Producer, producer
from columnflow.categorization import Categorizer
from columnflow.util import maybe_import
from columnflow.ml import MLModel
from columnflow.columnar_util import set_ak_column
from columnflow.selection.stats import increment_stats
from columnflow.histogramming.default import all_weights


ak = maybe_import("awkward")
np = maybe_import("numpy")

# helper
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

logger = law.logger.get_logger(__name__)


def del_sub_proc_stats(
    stats: dict,
    proc: str,
) -> np.ndarray:
    """
    Function deletes dict keys which are not part of the requested process

    :param stats: Dictionaire containing ML stats for each process.
    :param proc: String of the process.
    :param sub_id: List of ids of sub processes that should be reatined (!).
    """
    item_list = list(stats.weight_map.keys())
    for item in item_list:
        stats[item].pop()


@producer(
    uses={increment_stats, "process_id", "fold_indices"},
    produces={"event_weight"},
    extra_categorizer=None,
)
def prepml(
    self: Producer,
    events: ak.Array,
    task: law.Task,
    stats: dict = {},
    fold_indices: ak.Array | None = None,
    ml_model_inst: MLModel | None = None,
    **kwargs,
) -> ak.Array:
    """
    Producer that is run as part of PrepareMLEvents to collect relevant stats
    """
    if task.task_family == "cf.PrepareMLEvents":
        if self.extra_categorizer:
            for cat_cls in self.categorizers_cls:
                # apply additional categorizer if specified
                events, mask = self[cat_cls](events, **kwargs)
                logger.info(f"Select {ak.sum(mask)} from {len(events)} events using {cat_cls.cls_name}")
                events = events[mask]

    weight_map = {
        "num_events": Ellipsis,  # all events
    }

    if task.dataset_inst.is_mc:
        # full event weight
        events, weight = self[all_weights](events, task=task, **kwargs)
        events = set_ak_column_f32(events, "event_weight", weight)
        stats["sum_weights"] += float(ak.sum(weight, axis=0))
        weight_map["sum_weights"] = weight
        weight_map["sum_pos_weights"] = (weight, weight > 0)
        weight_map["sum_abs_weights"] = np.abs(weight)
        weight_map["num_events_pos_weights"] = weight > 0

        # normalization weight only
        norm_weight = events["normalization_weight"]
        stats["sum_norm_weights"] += float(ak.sum(norm_weight, axis=0))
        weight_map["sum_norm_weights"] = norm_weight
        weight_map["sum_pos_norm_weights"] = (norm_weight, norm_weight > 0)
        weight_map["sum_abs_norm_weights"] = np.abs(norm_weight)

    group_map = {
        "process": {
            "values": events.process_id,
            "mask_fn": (lambda v: events.process_id == v),
        },
        "fold": {
            "values": events.fold_indices,
            "mask_fn": (lambda v: events.fold_indices == v),
            "combinations_only": True,
        },
    }

    group_combinations = [("process", "fold")]

    self[increment_stats](
        events,
        None,  # SelectionResult that is not required
        stats,
        weight_map=weight_map,
        group_map=group_map,
        group_combinations=group_combinations,
        **kwargs,
    )

    key_list = list(weight_map.keys())
    for key in key_list:
        stats.pop(key, None)
        # TODO: pop 'num_fold_events'

    return events


@prepml.init
def prepml_init(self):
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return

    self.uses.add(all_weights)
    self.uses.add("normalization_weight")

    if self.extra_categorizer:
        self.categorizers_cls = []
        for cls_name in law.util.make_list(self.extra_categorizer):
            if not Categorizer.has_cls(cls_name):
                logger.warning(
                    f"Extra categorizer {cls_name} not found, skipping it in {self.cls_name}.",
                )
                continue
            cat_cls = Categorizer.get_cls(cls_name)
            self.categorizers_cls.append(cat_cls)
            self.uses.add(cat_cls)
