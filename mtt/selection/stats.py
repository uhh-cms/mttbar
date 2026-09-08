# coding: utf-8

"""
Custom increment_stats function to keep track of normalized weights.
"""
import law

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.production.cms.btag import btag_weights
from columnflow.columnar_util import optional_column as optional

from columnflow.util import maybe_import

from mtt.production.weights import event_weights_to_normalize
from mtt.util import has_tag

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@selector(
    categorizers=None,  # pass list of categorizers to evaluate number of (selected) events that fall in this category
    uses={increment_stats, optional("mc_weight")},
)
def mtt_selection_step_stats(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    """
    Selector to increment stats
    """
    weight_map = {}
    for step, mask in results.steps.items():
        weight_map[f"num_events_step_{step}"] = mask
    if self.dataset_inst.is_mc:
        for step, mask in results.steps.items():
            weight_map[f"sum_mc_weight_step_{step}"] = (events.mc_weight, mask)

    if self.categorizers:
        # apply list of categorizers and store number of (selected) events in each category
        event_mask = results.event
        for categorizer in self.categorizers:
            if not self.has_dep(categorizer):
                # skip categorizer if it is not used
                continue
            events, mask = self[categorizer](events, results, **kwargs)
            weight_map[f"num_events_cat_{categorizer.cls_name}"] = mask
            weight_map[f"num_events_selected_cat_{categorizer.cls_name}"] = mask & event_mask

    self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map={},
        **kwargs,
    )

    return events


@mtt_selection_step_stats.init
def mtt_selection_step_stats_init(self: Selector) -> None:
    if self.categorizers:
        for categorizer in self.categorizers:
            self.uses |= {categorizer}


@selector(
    uses={increment_stats, event_weights_to_normalize},
)
def mtt_increment_stats(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    """
    Main selector to increment stats needed for weight normalization
    """
    raw_met_column_name = f"Raw{self.config_inst.x.met_selection.column}"
    # collect important information from the results
    no_sel_mask = results.steps.no_sel_mask
    event_mask = results.event
    if not self.config_inst.has_tag("skip_btag_weights"):
        event_mask_no_bjet = results.steps.all_but_bjet

    # weight map definition
    weight_map = {
        # "num" operations
        "num_events_pre_bad_mask": Ellipsis,  # all events
        "num_events": no_sel_mask,  # all events after base mask
        "num_events_selected": event_mask,  # selected events only
    }
    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        weight_map["num_events_selected_no_bjet"] = event_mask_no_bjet

    if self.dataset_inst.is_mc:
        weight_map["num_negative_weights"] = (events.mc_weight < 0)
        weight_map["num_pu_0"] = (events.pu_weight == 0)
        weight_map["num_pu_100"] = (events.pu_weight >= 100)

        raw_met_column = events[raw_met_column_name]
        weight_map["num_raw_met_isinf"] = (~np.isfinite(raw_met_column.pt))
        weight_map["num_raw_met_isinf_selected"] = (~np.isfinite(raw_met_column.pt) & event_mask)
        # "sum" operations
        weight_map["sum_mc_weight_pre_bad_mask"] = events.mc_weight  # weights of all events
        weight_map["sum_mc_weight"] = (events.mc_weight, no_sel_mask)  # weights of all events after base mask
        weight_map["sum_mc_weight_selected"] = (events.mc_weight, event_mask)  # weights of selected events
        if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
            weight_map["sum_mc_weight_no_bjet"] = (events.mc_weight, event_mask_no_bjet)
            weight_map["sum_mc_weight_selected_no_bjet"] = (events.mc_weight, event_mask_no_bjet)

        weight_columns = set(self[event_weights_to_normalize].produced_columns)
        if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
            # btag_weights are not produced and therefore need some manual care
            weight_columns |= set(self[btag_weights].produced_columns)

        weight_columns = sorted([col.string_nano_column for col in weight_columns])

        # mc weight times correction weight (with variations) without any selection
        for name in weight_columns:
            if "weight" not in name:
                # skip non-weight columns here
                continue
            elif name.startswith("btag_weight"):
                # btag weights are handled via histograms
                continue

            weight_map[f"sum_mc_weight_{name}"] = (events.mc_weight * events[name], no_sel_mask)

            # weights for selected events
            weight_map[f"sum_mc_weight_{name}_selected"] = (events.mc_weight * events[name], event_mask)

            if name.startswith("btag_weight"):
                # weights for selected events, excluding the bjet selection
                weight_map[f"sum_mc_weight_{name}_selected_no_bjet"] = (
                    (events.mc_weight * events[name], event_mask_no_bjet)
                )

    group_map = {
        "process": {
            "values": events.process_id,
            "mask_fn": (lambda v: events.process_id == v),
        },
    }

    self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        **kwargs,
    )

    return events


@mtt_increment_stats.init
def mtt_increment_stats_init(self: Selector) -> None:
    raw_met_column = f"Raw{self.config_inst.x.met_selection.column}"
    self.uses |= {f"{raw_met_column}.pt", f"{raw_met_column}.phi"}
    if not getattr(self, "dataset_inst", None):
        return

    if not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {btag_weights}

    if self.dataset_inst.is_mc:
        self.uses |= {"mc_weight"}
