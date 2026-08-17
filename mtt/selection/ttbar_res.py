# coding: utf-8

"""
Default selection for m(ttbar).
"""
from __future__ import annotations

import law

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.btag import fill_btag_wp_count_hists
from columnflow.production.processes import process_ids
from columnflow.production.categories import category_ids

from mtt.selection.cleanup import cleanup, get_weights_and_no_sel_mask
from mtt.selection.general import jet_energy_shifts
from mtt.selection.lepton import lepton_selection
from mtt.selection.jets import jet_selection, met_selection, lepton_jet_2d_selection, top_tagged_jets
from mtt.selection.qcd_spikes import qcd_spikes
from mtt.selection.data_trigger_veto import data_trigger_veto
from mtt.selection.cutflow_features import cutflow_features
from mtt.selection.stats import mtt_increment_stats, mtt_selection_step_stats
from mtt.selection.hists import mtt_selection_hists

from mtt.util import has_tag, record_calls

np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


@selector(
    uses={
        cleanup,
        lepton_selection,
        jet_selection,
        met_selection,
        lepton_jet_2d_selection,
        top_tagged_jets,
        get_weights_and_no_sel_mask,
        process_ids,
        category_ids,
        cutflow_features,
        process_ids,
        mtt_selection_step_stats,
        mtt_increment_stats,
        mtt_selection_hists,
    },
    produces={
        cleanup,
        lepton_selection,
        jet_selection,
        met_selection,
        lepton_jet_2d_selection,
        top_tagged_jets,
        get_weights_and_no_sel_mask,
        process_ids,
        category_ids,
        cutflow_features,
        process_ids,
        mtt_selection_step_stats,
        mtt_increment_stats,
        mtt_selection_hists,
    },
    shifts={
        jet_energy_shifts,
    },
    exposed=True,
)
def ttbar_res_sel(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    hists: DotDict[str, hist.Hist],
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    logger.warning("IMPORTANT WARNING INCOMING!!!!!")
    logger.warning("DID YOU REMEMBER TO REMOVE THE n_jets AND n_bjets CATEGORIES FROM THE CONFIG FILE??????!!!")
    run_list = []
    with record_calls(self, run_list):
        # perform cleanup steps:
        # deterministic seeds, large_weights_killer, process_ids
        events, results = self[cleanup](events, stats, hists=hists, **kwargs)

        # lepton selection
        events, lepton_results = self[lepton_selection](events, **kwargs)
        results += lepton_results

        # jet selection
        events, jet_results = self[jet_selection](events, **kwargs)
        results += jet_results

        # met selection
        events, met_results = self[met_selection](events, **kwargs)
        results += met_results

        # jet-lepton 2D cut
        events, lepton_jet_2d_results = self[lepton_jet_2d_selection](events, lepton_results, **kwargs)
        results += lepton_jet_2d_results

        # all-hadronic veto
        events, top_tagged_jets_results = self[top_tagged_jets](events, **kwargs)
        results += top_tagged_jets_results

        if self.dataset_inst.has_tag("is_qcd"):
            events, qcd_sel_results = self[qcd_spikes](events, **kwargs)
            results += qcd_sel_results

        if not self.dataset_inst.is_mc:
            events, trigger_veto_results = self[data_trigger_veto](events, **kwargs)
            results += trigger_veto_results

        # derive event weights and add base mask of all events that are not considered bad to "cleanup" step
        events, results = self[get_weights_and_no_sel_mask](events, results, **kwargs)
        results.steps["cleanup"] = results.steps.cleanup & results.steps["no_sel_mask"]

        results.steps["all_but_trigger_and_2d_iso"] = (
            results.steps.cleanup &
            results.steps.Lepton &
            results.steps.DileptonVeto &
            results.steps.Jet &
            results.steps.BJet &
            results.steps.MET &
            results.steps.AllHadronicVeto
        )

        results.steps["all_but_trigger"] = (
            results.steps.all_but_trigger_and_2d_iso &
            results.steps.JetLepton2DCut
        )

        results.steps["all_but_2d_iso"] = (
            results.steps.cleanup &
            results.steps.Lepton &
            results.steps.DileptonVeto &
            results.steps.LeptonTrigger &
            results.steps.Jet &
            results.steps.BJet &
            results.steps.MET &
            results.steps.JetLepton2DCut &
            results.steps.AllHadronicVeto
        )

        results.steps["all_but_trigger_and_bjet"] = (
            results.steps.cleanup &
            results.steps.Lepton &
            results.steps.DileptonVeto &
            results.steps.Jet &
            results.steps.MET &
            results.steps.JetLepton2DCut &
            results.steps.AllHadronicVeto
        )

        results.steps["all_but_bjet"] = (
            results.steps.all_but_trigger_and_bjet &
            results.steps.LeptonTrigger
        )

        for step in [
            "all_but_trigger_and_2d_iso",
            "all_but_trigger",
            "all_but_2d_iso",
            "all_but_trigger_and_bjet",
            "all_but_bjet",
        ]:
            if self.dataset_inst.has_tag("is_qcd"):
                results.steps[step] = (
                    results.steps[step] &
                    results.steps.QCDSpikes
                )
            if not self.dataset_inst.is_mc:
                results.steps[step] = (
                    results.steps[step] &
                    results.steps.TriggerVeto
                )

        results.steps["all"] = (
            results.steps.all_but_bjet &
            results.steps.BJet
        )

        # combined event selection after all steps
        event_sel = reduce(and_, results.steps.values())
        results.event = event_sel

        for step, sel in results.steps.items():
            n_sel = ak.sum(sel, axis=-1)
            logger.info(f"{step}: {n_sel}")

        n_sel = ak.sum(event_sel, axis=-1)
        if n_sel - ak.sum(results.steps["all"]) != 0:
            logger.info(f"__all__: {n_sel}")
            logger.warning(
                "Number of events passing combined selection does not match number of events passing "
                f"all individual steps: {n_sel} vs {ak.sum(results.steps['all'])}",
            )
            raise ValueError("Inconsistent event selection results")

        # produce cutflow features
        events = self[cutflow_features](events, results=results, **kwargs)

        # build categories
        events = self[category_ids](events, results=results, **kwargs)

        # create process id.  # NOTE: needed again?
        events = self[process_ids](events, results=results, **kwargs)

        # increment stats
        events = self[mtt_selection_step_stats](events, results, stats, **kwargs)
        events = self[mtt_increment_stats](events, results, stats, **kwargs)
        events = self[mtt_selection_hists](events, results, hists, **kwargs)

        if self.dataset_inst.is_mc and has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
            self[fill_btag_wp_count_hists](events, results.event, results.objects.Jet.Jet, hists, **kwargs)

        def log_fraction(stats_key: str, msg: str | None = None):
            if not stats.get(stats_key):
                return
            if not msg:
                msg = "Fraction of {stats_key}"
            logger.info(f"{msg}: {(100 * stats[stats_key] / stats['num_events']):.2f}%")

        log_fraction("num_negative_weights", "Fraction of negative weights")
        log_fraction("num_pu_0", "Fraction of events with pu_weight == 0")
        log_fraction("num_pu_100", "Fraction of events with pu_weight >= 100")

        # avoid none values in events
        events = ak.fill_none(events, EMPTY_FLOAT)

        logger.info(f"Selected {ak.sum(results.event)} from {len(events)} events")

    logger.info_once(
        "Finished ttbar_res_sel selector steps:\n" +
        "\n".join(run_list),
    )

    return events, results


@ttbar_res_sel.init
def ttbar_res_sel_init(self: Selector) -> None:
    if hasattr(self, "dataset_inst") and self.dataset_inst.has_tag("is_qcd"):
        self.uses |= {qcd_spikes}
        self.produces |= {qcd_spikes}

    if hasattr(self, "dataset_inst") and not self.dataset_inst.is_mc:
        self.uses |= {data_trigger_veto}
        self.produces |= {data_trigger_veto}

    if self.dataset_inst.is_mc and has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {fill_btag_wp_count_hists}
        self.produces |= {fill_btag_wp_count_hists}
