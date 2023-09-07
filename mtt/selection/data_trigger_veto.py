# coding: utf-8

"""
Selection methods for data trigger veto to prevent double counting.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.met_filters import met_filters
from columnflow.selection.cms.json_filter import json_filter
from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.processes import process_ids

from mtt.selection.util import masked_sorted_indices
from mtt.selection.general import increment_stats, jet_energy_shifts
from mtt.selection.lepton import lepton_selection
from mtt.selection.cutflow_features import cutflow_features
from mtt.selection.early import check_early

from mtt.production.lepton import choose_lepton
from mtt.production.gen_top import gen_parton_top
from mtt.production.gen_v import gen_v_boson

np = maybe_import("numpy")
ak = maybe_import("awkward")

@selector(
    uses={
        attach_coffea_behavior,
        lepton_selection,
        check_early,
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
    },
    exposed=True,
)
def data_trigger_veto(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    # get trigger requirements
    trigger_config = self.config_inst.x.triggers

    # check if event is in early run period
    is_early = self[check_early](events, trigger_config=trigger_config, **kwargs)

    # ensure lepton selection was run, get lepton pT regime
    events = self[lepton_selection](events, **kwargs)
    pt_regime = events["pt_regime"]

    # pt regime booleans for convenience
    is_lowpt = (pt_regime == 1)
    is_highpt = (pt_regime == 2)

    triggers = {}
    trigger_masks = {}
    pass_trigger = {}
    for object_name in ["muon", "electron", "photon"]:
        triggers[object_name] = {
            "lowpt": trigger_config.get("lowpt", {}).get("all", {}).get("triggers", {}).get(object_name, {}),
            "highpt_early": trigger_config.get("highpt", {}).get("early", {}).get("triggers", {}).get(object_name, {}),
            "highpt_late": trigger_config.get("highpt", {}).get("late", {}).get("triggers", {}).get(object_name, {}),
        }
        trigger_masks[object_name] = object_trigger_masks = {}
        # get trigger decisions if trigger is available
        for key, trigger_names in triggers[object_name].items():
            object_trigger_masks[key] = ak.zeros_like(events.event, dtype=bool)
            for trigger_name in trigger_names:
                if trigger_name in events.HLT.fields:
                    object_trigger_masks[key] = (
                        object_trigger_masks[key] |
                        events.HLT[trigger_name]
                    )
                else:
                    object_trigger_masks[key] = (False)

        object_trigger_masks["highpt"] = ak.where(
            is_early,
            object_trigger_masks["highpt_early"],
            object_trigger_masks["highpt_late"],
        )

        # trigger selection
        pass_object_trigger = ak.zeros_like(events.event, dtype=bool)
        pass_object_trigger = ak.where(
            is_lowpt,
            object_trigger_masks["lowpt"],
            pass_object_trigger,
        )
        pass_object_trigger = ak.where(
            is_highpt,
            object_trigger_masks["highpt"],
            pass_object_trigger,
        )
        pass_trigger[object_name] = pass_object_trigger

    if self.dataset_inst.has_tag("is_e_data"):
        sel_veto = ak.fill_none(pass_trigger["electron"], False)
    if self.dataset_inst.has_tag("is_pho_data"):
        sel_veto = ak.fill_none(pass_trigger["photon"] & ~pass_trigger["electron"], False)
    if self.dataset_inst.has_tag("is_mu_data"):
        sel_veto = ak.fill_none(pass_trigger["muon"] & ~pass_trigger["electron"] & ~pass_trigger["photon"], False)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps={
            "TriggerVeto": sel_veto,
        },
    )
