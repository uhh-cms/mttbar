# coding: utf-8

"""
Selection methods for m(ttbar).
"""

from typing import Tuple

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from columnflow.selection import Selector, SelectionResult, selector

from mtt.selection.util import masked_sorted_indices
from mtt.selection.early import check_early
from mtt.production.lepton import choose_lepton

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "event",
        check_early,
        "Electron.pt", "Electron.eta",
        "Electron.cutBased",
        "Electron.deltaEtaSC",
        "Electron.mvaFall17V2Iso_WP80",
        "Electron.mvaFall17V2noIso_WP80",
    },
)
def electron_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Select electrons and check pt regime and isolation.

    - require exactly 1 electron with abseta_SC < 2.5 and
      either (30 < pt < 120 GeV) and MVA ID WP80,
      or (pt > 120 GeV) and MVA ID WP80 without isolation
    - for low-pt electrons, the isolation is part of the ID and
      is not applied separately
    - for high-pt electrons, no isolation criteria are applied
      here. An additional 2D isolation criterion, which considers
      the separation from the nearest jet, is applied via a
      separate selector.
    """

    lepton = events.Electron

    # general lepton kinematics
    lepton_mask_eta = (
        (abs(lepton.eta + lepton.deltaEtaSC) < 2.5) &
        # filter out electrons in barrel-endcap transition region
        ((abs(lepton.eta) < 1.44) | (abs(lepton.eta) > 1.57))
    )

    # different criteria for low- and high-pt lepton
    lepton_mask_lowpt = (
        lepton_mask_eta &
        (lepton.pt > 35) &
        (lepton.pt <= 120) &
        # MVA electron ID (WP 80, with isolation)
        (lepton.mvaFall17V2Iso_WP80)
    )
    lepton_mask_highpt = (
        lepton_mask_eta &
        (lepton.pt > 120) &
        # MVA electron ID (WP 80, no isolation)
        lepton.mvaFall17V2noIso_WP80
    )
    lepton_mask = (
        lepton_mask_lowpt | lepton_mask_highpt
    )
    lepton_indices = masked_sorted_indices(lepton_mask, lepton.pt)

    # veto events if additional leptons present
    # (note the looser cuts)
    add_leptons = (
        (abs(lepton.eta + lepton.deltaEtaSC) < 2.5) &
        (lepton.pt > 25) &
        # cut-based electron ID (3: tight working point)
        (lepton.cutBased == 3) &
        ~lepton_mask
    )
    dilepton_veto = (ak.sum(add_leptons, axis=-1) < 2)

    # lepton multiplicity
    n_lep = ak.sum(lepton_mask, axis=-1)
    n_lep_lowpt = ak.sum(lepton_mask_lowpt, axis=-1)
    n_lep_highpt = ak.sum(lepton_mask_highpt, axis=-1)

    # mark pt regime of events (0: undefined, 1: low-pt, 2: high-pt)
    pt_regime = ak.zeros_like(events.event, dtype=np.int8)
    pt_regime = ak.where(
        (n_lep == 1) & (n_lep_lowpt == 1), 1, pt_regime)
    pt_regime = ak.where(
        (n_lep == 1) & (n_lep_highpt == 1), 2, pt_regime)

    return SelectionResult(
        steps={
            "Lepton": (ak.num(lepton_indices) == 1),
            "DileptonVeto": dilepton_veto,
        },
        objects={
            "Electron": {
                "Electron": lepton_indices,
            },
        },
        aux={
            "pt_regime": pt_regime,
        },
    )


@selector(
    uses={
        "event",
        check_early,
        "Muon.pt", "Muon.eta", "Muon.tightId", "Muon.highPtId", "Muon.pfIsoId",
    },
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Select muons and check pt regime and isolation.

    - require exactly 1 muon with abseta < 2.4 and
      either (30 < pt < 55 GeV) and CutBasedIdTight,
      or (pt > 55 GeV) and CutBasedIdGlobalHighPt
    - low-pt muons have to satisfy the isolation ID "PfIsoTight"
    - for high-pt muons, no isolation criteria are applied
      here. An additional 2D isolation criterion, which considers
      the separation from the nearest jet, is applied via a
      separate selector.
    """

    lepton = events.Muon

    # general lepton kinematics
    lepton_mask_eta = (abs(lepton.eta) < 2.4)

    # different criteria for low- and high-pt lepton
    lepton_mask_lowpt = (
        lepton_mask_eta &
        (lepton.pt > 30) &
        (lepton.pt <= 55) &
        # 4 == PFIsoTight
        (lepton.pfIsoId == 4) &
        # CutBasedIdTight
        (lepton.tightId)
    )
    lepton_mask_highpt = (
        lepton_mask_eta &
        (lepton.pt > 55) &
        # CutBasedIdGlobalHighPt
        (lepton.highPtId == 2)
    )
    lepton_mask = (
        lepton_mask_lowpt | lepton_mask_highpt
    )
    lepton_indices = masked_sorted_indices(lepton_mask, lepton.pt)

    # veto events if additional leptons present
    # (note the looser cuts)
    add_leptons = (
        (abs(lepton.eta) < 2.4) &
        (lepton.pt > 25) &
        # CutBasedIdTight
        (lepton.tightId) &
        ~lepton_mask
    )
    dilepton_veto = (ak.sum(add_leptons, axis=-1) < 2)

    # lepton multiplicity
    n_lep = ak.sum(lepton_mask, axis=-1)
    n_lep_lowpt = ak.sum(lepton_mask_lowpt, axis=-1)
    n_lep_highpt = ak.sum(lepton_mask_highpt, axis=-1)

    # mark pt regime of events (0: undefined, 1: low-pt, 2: high-pt)
    pt_regime = ak.zeros_like(events.event, dtype=np.int8)
    pt_regime = ak.where(
        (n_lep == 1) & (n_lep_lowpt == 1), 1, pt_regime)
    pt_regime = ak.where(
        (n_lep == 1) & (n_lep_highpt == 1), 2, pt_regime)

    return SelectionResult(
        steps={
            "Lepton": (ak.num(lepton_indices) == 1),
            "DileptonVeto": dilepton_veto,
        },
        objects={
            "Muon": {
                "Muon": lepton_indices,
            },
        },
        aux={
            "pt_regime": pt_regime,
        },
    )


def merge_selection_steps(step_dicts):
    """
    Merge two or more dictionaries of selection steps by concatenating the
    corresponding masks for the different selections steps along the innermost
    dimension (axis=-1).
    """

    step_names = {
        step_name
        for step_dict in step_dicts
        for step_name in step_dict
    }

    # check for step name incompatibilities
    if any(step_names != set(step_dict) for step_dict in step_dicts):
        raise ValueError(
            "Selection steps to merge must have identical "
            "selection step names!",
        )

    merged_steps = {
        step_name: ak.concatenate([
            ak.singletons(step_dict[step_name])
            for step_dict in step_dicts
        ], axis=-1)
        for step_name in step_names
    }

    return merged_steps


@selector(
    uses={
        "event",
        check_early, muon_selection, electron_selection,
        choose_lepton,
        "HLT.*",
    },
    produces={
        "channel_id",
        "pt_regime",
        choose_lepton,
    },
    exposed=True,
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> Tuple[ak.Array, SelectionResult]:
    """Select muons/electrons and determine channel for event."""

    # get channels from the config
    ch_e = self.config_inst.get_channel("e")
    ch_mu = self.config_inst.get_channel("mu")

    # selection results for each channel
    channel_results = {}

    # array of lists to keep track of matching channels
    channel_ids = ak.singletons(events.event)[:, :0]
    channel_indexes = ak.values_astype(
        ak.singletons(events.event)[:, :0],
        np.int8,
    )

    merged_objects = {}
    for ch_index, (channel, ch_selector, lepton_name, lepton_route) in enumerate([
        (ch_mu.id, muon_selection, "muon", "Muon"),
        (ch_e.id, electron_selection, "electron", "Electron"),
    ]):

        # selection results for channel
        channel_results[channel] = results = self[ch_selector](events, **kwargs)

        pt_regime = results.aux["pt_regime"]

        # pt regime booleans for convenience
        is_lowpt = (pt_regime == 1)
        is_highpt = (pt_regime == 2)

        # -- check trigger

        # get trigger requirements
        trigger_config = self.config_inst.x.triggers
        triggers = {
            "lowpt": trigger_config.lowpt.all.triggers[lepton_name],
            "highpt_early": trigger_config.highpt.early.triggers[lepton_name],
            "highpt_late": trigger_config.highpt.late.triggers[lepton_name],
        }

        # check if early run period
        is_early = self[check_early](events, trigger_config=trigger_config)

        # compute trigger masks
        trigger_masks = {}
        trigger_found = {}  # checked at end to see if valid trigger was found
        missing_triggers = set()  # for more information on error
        for key, trigger_names in triggers.items():
            trigger_masks[key] = ak.zeros_like(events.event, dtype=bool)

            # keep track of missing triggers
            missing_triggers |= {
                tn for tn in trigger_names
                if tn not in events.HLT.fields
            }

            # retrieve trigger decisions
            all_triggers_found = True
            for trigger_name in trigger_names:
                # skip if trigger not found
                if trigger_name not in events.HLT.fields:
                    all_triggers_found = False
                    continue
                else:
                    trigger_masks[key] = (
                        trigger_masks[key] |
                        events.HLT[trigger_name]
                    )

            # key is considered 'found' iff all component triggers are found
            trigger_found[key] = ak.full_like(events.event, all_triggers_found, dtype=bool)

        # determine which high-pt trigger combination to use
        # and whether it was found
        for trigger_arr in (trigger_masks, trigger_found):
            trigger_arr["highpt"] = ak.where(
                is_early,
                trigger_arr["highpt_early"],
                trigger_arr["highpt_late"],
            )

        # sanity check: make sure that at least one of the
        # required triggers was found (catch misspellings, etc.)
        trigger_exists = ak.zeros_like(events.event, dtype=bool)
        trigger_exists = ak.where(
            is_lowpt,
            trigger_found["lowpt"],
            trigger_exists,
        )
        trigger_exists = ak.where(
            is_highpt,
            trigger_found["highpt"],
            trigger_exists,
        )
        n_expected = ak.sum(is_lowpt | is_highpt)
        n_trigger_exists = ak.sum(trigger_exists)
        if n_trigger_exists != n_expected:
            missing_triggers_str = "\n".join(
                f"  - {t}" for t in sorted(missing_triggers)
            )
            raise RuntimeError(
                f"One or more required trigger(s) not found in trigger table for "
                f"{n_trigger_exists}/{n_expected} preselected events:\n{missing_triggers_str}",
            )

        # trigger selection
        pass_trigger = ak.zeros_like(events.event, dtype=bool)
        pass_trigger = ak.where(
            is_lowpt,
            trigger_masks["lowpt"],
            pass_trigger,
        )
        pass_trigger = ak.where(
            is_highpt,
            trigger_masks["highpt"],
            pass_trigger,
        )

        # add channel IDs based on selection result
        add_channel_id = ak.singletons(
            ak.where(
                results.steps["Lepton"],
                channel,
                0,
            ),
        )
        channel_ids = ak.concatenate([
            channel_ids,
            add_channel_id[add_channel_id != 0],
        ], axis=-1)

        # keep track of channel index
        add_channel_index = ak.singletons(
            ak.where(
                results.steps["Lepton"],
                ch_index,
                -1,
            ),
        )
        channel_indexes = ak.concatenate([
            channel_indexes,
            add_channel_index[add_channel_index != -1],
        ], axis=-1)

        # add the trigger result as a selection step
        results.steps["LeptonTrigger"] = pass_trigger

        # add the object indices to the selection
        merged_objects.update(results.objects)

    # concatenate selection results
    step_dicts = [r.steps for r in channel_results.values()]
    aux_dicts = [r.aux for r in channel_results.values()]
    merged_steps = merge_selection_steps(step_dicts)
    merged_aux = merge_selection_steps(aux_dicts)

    # decide channel and merge selection results
    merged_steps = {
        step: ak.fill_none(ak.firsts(
            selection[channel_indexes],
        ), False)
        for step, selection in merged_steps.items()
    }
    merged_aux = {
        var: ak.firsts(
            vals[channel_indexes],
        )
        for var, vals in merged_aux.items()
    }

    # veto events with both electrons and muons
    # passing the selection cuts
    merged_steps["DileptonVeto"] = (
        merged_steps["DileptonVeto"] &
        (
            (
                ak.num(merged_objects["Muon"]["Muon"], axis=-1) +
                ak.num(merged_objects["Electron"]["Electron"], axis=-1)
            ) <= 1
        )
    )

    # invalidate channel if both e and mu were found
    channel_id = ak.where(
        ak.num(channel_ids, axis=-1) == 1,
        ak.firsts(channel_ids),
        0,
    )

    # set channel to 0 if undefined
    channel_id = ak.fill_none(channel_id, 0)

    # ensure integer type
    channel_id = ak.values_astype(channel_id, np.int8)

    # put channel in a column
    events = set_ak_column(events, "channel_id", channel_id)

    # put pt regime in a column
    events = set_ak_column(
        events,
        "pt_regime",
        ak.fill_none(merged_aux["pt_regime"], 0),
    )

    # multiplex Muon/Electron to a single Lepton collection
    # based on channel_id
    events = self[choose_lepton](events, **kwargs)

    # build and return selection results plus new columns
    return events, SelectionResult(
        steps=merged_steps,
        objects=merged_objects,
        aux=merged_aux,
    )
