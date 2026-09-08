# coding: utf-8

"""
Selection modules for HH(bbWW) that are used for both SL and DL.
"""

from __future__ import annotations

from collections import defaultdict

import law
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, fill_at
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.met_filters import met_filters
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.cms.jets import jet_veto_map
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.categories import category_ids
from columnflow.production.processes import process_ids
from columnflow.production.cms.seeds import deterministic_seeds

from mtt.production.weights import event_weights_to_normalize, large_weights_killer
from mtt.selection.stats import mtt_selection_step_stats, mtt_increment_stats
from mtt.selection.hists import mtt_selection_hists
from mtt.selection.bad_events import extend_bad_events, get_outlier_scale_weights
from mtt.util import IF_MC, record_calls

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


def masked_sorted_indices(mask: ak.Array, sort_var: ak.Array, ascending: bool = False) -> ak.Array:
    """
    Helper function to obtain the correct indices of an object mask
    """
    indices = ak.argsort(sort_var, axis=-1, ascending=ascending)
    return indices[mask[indices]]


def get_met_filters(self: Selector):
    """ custom function to skip met filter for our Run2 EOY signal samples """
    met_filters = self.config_inst.x.met_filters

    if getattr(self, "dataset_inst", None) and self.dataset_inst.has_tag("is_eoy"):
        # remove filter for EOY sample
        try:
            met_filters.remove("Flag.BadPFMuonDzFilter")
        except (KeyError, AttributeError):
            pass

    return met_filters


mtt_met_filters = met_filters.derive("mtt_met_filters", cls_dict=dict(get_met_filters=get_met_filters))


@selector(
    uses={
        jet_veto_map,
        mtt_met_filters, json_filter, "PV.npvsGood",
        process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    produces={
        mtt_met_filters, json_filter,
        process_ids, attach_coffea_behavior,
        mc_weight, large_weights_killer,
    },
    exposed=False,
)
def cleanup(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    task: law.Task,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """ Methods that are called for both WP and SF before calling the selection modules """
    run_list = []

    with record_calls(self, run_list):
        # temporary fix for optional types from Calibration (e.g. events.Jet.pt --> ?float32)
        # TODO: remove as soon as possible as it might lead to weird bugs when there are none entries in inputs
        events = ak.fill_none(events, EMPTY_FLOAT)

        # prepare the selection results that are updated at every step
        results = SelectionResult()

        # run deterministic seeds when no Calibrator has been requested
        if not task.calibrators:
            events = self[deterministic_seeds](events, **kwargs)

        # mc weight
        if self.dataset_inst.is_mc:
            events = self[mc_weight](events, **kwargs)
            events = self[large_weights_killer](events, stats, **kwargs)

        # create process ids
        events = self[process_ids](events, **kwargs)

        # ensure coffea behavior
        events = self[attach_coffea_behavior](events, **kwargs)

        # apply some general quality criteria on events
        results.steps["good_vertex"] = events.PV.npvsGood >= 1
        events, met_results = self[mtt_met_filters](events, **kwargs)  # produces "met_filter" step

        results += met_results

        if self.dataset_inst.is_data:
            events, json_results = self[json_filter](events, **kwargs)  # produces "json" step
            results += json_results
        else:
            results.steps["json"] = ak.Array(np.ones(len(events), dtype=bool))

        # apply jet veto map
        events, jet_veto_results = self[jet_veto_map](events, **kwargs)
        results += jet_veto_results

        # combine quality criteria into a single step
        results.steps["cleanup"] = (
            results.steps.jet_veto_map &
            results.steps.good_vertex &
            results.steps.met_filter &
            results.steps.json
        )

    logger.info_once(
        "Finished pre-selection steps:\n" +
        "\n".join(run_list),
    )

    return events, results


@cleanup.init
def cleanup_init(self: Selector) -> None:
    if not getattr(self, "dataset_inst", None) or self.dataset_inst.is_data:
        return


@cleanup.post_init
def cleanup_post_init(self: Selector, task: law.Task) -> None:
    if not task.calibrators:
        self.uses.add(deterministic_seeds)
        self.produces.add(deterministic_seeds)


@selector(
    uses={
        get_outlier_scale_weights, extend_bad_events,
        event_weights_to_normalize,
    },
    produces={
        event_weights_to_normalize,
        IF_MC("mc_weight"),
    },
)
def get_weights_and_no_sel_mask(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Helper to get the weights and bad mask for the events.
    """
    events, results = self[get_outlier_scale_weights](events, results=results, **kwargs)

    # produce event weights
    if self.dataset_inst.is_mc:
        events = self[event_weights_to_normalize](events, results=results, **kwargs)
    events, results = self[extend_bad_events](events, results=results, **kwargs)

    # set mc_weight to 0 for events that are considered bad for simplified downstream processing
    if ak.any(bad := ~results.steps.no_sel_mask):
        logger.warning(
            f"Found {ak.sum(bad)} events ({100 * ak.mean(bad):.1f}%) that are considered bad, setting mc_weight to 0",
        )
        if self.dataset_inst.is_mc:
            events = fill_at(events, bad, "mc_weight", 0.0, value_type=np.float32)

    return events, results


@selector(
    uses={
        category_ids, mtt_increment_stats, mtt_selection_step_stats,
        mtt_selection_hists,
    },
    produces={
        category_ids, mtt_increment_stats, mtt_selection_step_stats,
        mtt_selection_hists,
    },
    exposed=False,
)
def post_selection(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: defaultdict,
    hists: dict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    events = self[mtt_selection_step_stats](events, results, stats, **kwargs)
    events = self[mtt_increment_stats](events, results, stats, **kwargs)
    events = self[mtt_selection_hists](events, results, hists, **kwargs)

    def log_fraction(stats_key: str, msg: str | None = None):
        if not stats.get(stats_key):
            return
        if not msg:
            msg = "Fraction of {stats_key}"
        logger.info(f"{msg}: {(100 * stats[stats_key] / stats['num_events']):.2f}%")

    log_fraction("num_negative_weights", "Fraction of negative weights")
    log_fraction("num_pu_0", "Fraction of events with pu_weight == 0")
    log_fraction("num_pu_100", "Fraction of events with pu_weight >= 100")

    # temporary fix for optional types from Calibration (e.g. events.Jet.pt --> ?float32)
    # TODO: remove as soon as possible as it might lead to weird bugs when there are none entries in inputs
    events = ak.fill_none(events, EMPTY_FLOAT)

    logger.info(f"Selected {ak.sum(results.event)} from {len(events)} events")
    return events, results
