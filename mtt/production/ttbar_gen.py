# coding: utf-8

"""
Column production methods related to ttbar mass reconstruction.
"""
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from mtt.production.util import lv_mass, delta_r_match, delta_r_match_multiple

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        "channel_id",
        "Generator.*",
        "GenJet.*",
        "GenJetAK8.*",
        "GenPart.*",
    },
    produces={
        "GenTTbar.*",
    },
)
def ttbar_gen(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Find the generator particles resulting from the ttbar decay.
    """
    # load coffea behaviors for simplified arithmetic with vectors
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)

    # use `GenParticle` behavior
    gen_part = ak.with_name(events["GenPart"], "GenParticle")

    # derived gen-particle properties
    gen_part["genPartIdxMotherMasked"] = ak.mask(gen_part.genPartIdxMother, gen_part.genPartIdxMother >= 0)
    gen_part["absPdgId"] = abs(gen_part.pdgId)
    gen_part["index"] = ak.local_index(gen_part, axis=1)

    # -- helper functions

    def parent(gen_part):
        """
        Return an array with the same structure as `gen_part` that contains the parent particle
        at the same positions. Entries are masked if not parent particle exists.
        """
        return gen_part[gen_part.genPartIdxMotherMasked]

    def is_descended_from(gen_part, idx):
        """
        Return an array with the same structure as `gen_part` that indicated whether the
        respective generator particle is part of the decay chain of the particle at position `idx`.
        """
        idx = ak.fill_none(idx, -1)
        result = ak.zeros_like(gen_part.index, dtype=bool)

        # apply `parents` repeatedly and check index
        gen_parent = gen_part
        while ak.any(~ak.is_none(gen_parent.index, axis=1)):
            result = result | ak.fill_none(gen_parent.index == idx, False)
            gen_parent = gen_part[gen_parent.genPartIdxMotherMasked]

        return result

    # extract bool flags
    is_hard_proc = gen_part.hasFlags("isHardProcess")
    from_hard_proc = gen_part.hasFlags("fromHardProcess")
    is_last_copy = gen_part.hasFlags("isLastCopy")

    # find top quarks produced during the hard scattering
    gen_part_hp = gen_part[is_hard_proc]
    gen_hp_top = gen_part_hp[gen_part_hp.absPdgId == 6]
    gen_hp_top = ak.pad_none(gen_hp_top, 2)
    gen_hp_top_1 = gen_hp_top[:, 0]
    gen_hp_top_2 = gen_hp_top[:, 1]

    # find last copy of top quarks before decay
    gen_top = gen_part[(gen_part.absPdgId == 6) & is_last_copy]
    gen_top = ak.pad_none(gen_top, 2)
    gen_top_1 = gen_top[:, 0]
    gen_top_2 = gen_top[:, 1]

    # identify top quark decay products
    is_top_1_decay = is_descended_from(gen_part, gen_top_1.index)
    is_top_2_decay = is_descended_from(gen_part, gen_top_2.index)
    gen_part["is_top_decay"] = (is_top_1_decay | is_top_2_decay)

    # b quarks
    is_bquark = is_last_copy & from_hard_proc & (gen_part.absPdgId == 5)
    gen_b_1 = ak.firsts(gen_part[is_bquark & is_top_1_decay])
    gen_b_2 = ak.firsts(gen_part[is_bquark & is_top_2_decay])

    # identify W bosons
    is_w = (gen_part.absPdgId == 24) & from_hard_proc & is_last_copy
    gen_w_1 = ak.firsts(gen_part[is_w & is_top_1_decay])
    gen_w_2 = ak.firsts(gen_part[is_w & is_top_2_decay])

    # identify W boson decay products
    is_w_1_decay = is_descended_from(gen_part, gen_w_1.index)
    is_w_2_decay = is_descended_from(gen_part, gen_w_2.index)
    gen_part["is_w_decay"] = (is_w_1_decay | is_w_2_decay)

    # charged leptons (e, mu only)
    is_lepton = is_last_copy & from_hard_proc & ((gen_part.absPdgId == 11) | (gen_part.absPdgId == 13))
    gen_lep_1 = gen_part[is_lepton & is_w_1_decay]
    gen_lep_2 = gen_part[is_lepton & is_w_2_decay]

    # neutrinos (e, mu only)
    is_neutrino = is_last_copy & from_hard_proc & ((gen_part.absPdgId == 12) | (gen_part.absPdgId == 14))
    gen_nu_1 = gen_part[is_neutrino & is_w_1_decay]
    gen_nu_2 = gen_part[is_neutrino & is_w_2_decay]

    # light quarks (u, d, c, s)
    is_lquark = is_last_copy & from_hard_proc & (gen_part.absPdgId >= 1) & (gen_part.absPdgId <= 4)
    gen_q_1 = gen_part[is_lquark & is_w_1_decay]
    gen_q_2 = gen_part[is_lquark & is_w_2_decay]

    # check if event is semileptonic at gen-level
    n_gen_q_1, n_gen_q_2 = ak.num(gen_q_1), ak.num(gen_q_2)
    is_semileptonic = ak.fill_none(
        (
            ((n_gen_q_1 == 2) & (n_gen_q_2 == 0)) |
            ((n_gen_q_1 == 0) & (n_gen_q_2 == 2))
        ),
        False,
    )

    # -- match gen jets to quarks

    gen_jet = lv_mass(events.GenJet)
    gen_jet["index"] = ak.local_index(gen_jet)

    max_dr = 0.2
    gen_jet_b_1, gen_jet = delta_r_match(gen_jet, gen_b_1, max_dr=max_dr)
    gen_jet_b_2, gen_jet = delta_r_match(gen_jet, gen_b_2, max_dr=max_dr)

    gen_jet_q_1, gen_jet = delta_r_match_multiple(gen_jet, gen_q_1, max_dr=max_dr)
    gen_jet_q_2, gen_jet = delta_r_match_multiple(gen_jet, gen_q_2, max_dr=max_dr)

    # helper function for writing indices
    def set_ak_column_idx(events, route, arr):
        idx = arr.index
        events = set_ak_column(events, route, idx)
        # add nested axis so column can be used
        # to index original `GenPart` array
        if idx.ndim == 1:
            idx = ak.singletons(idx)
            events[tuple(route.split("."))] = idx
        return events

    # -- write gen particle indices to columns

    # top quarks
    events = set_ak_column_idx(events, "GenTTbar.hp_top_1", gen_hp_top_1)
    events = set_ak_column_idx(events, "GenTTbar.hp_top_2", gen_hp_top_2)
    events = set_ak_column_idx(events, "GenTTbar.top_1", gen_top_1)
    events = set_ak_column_idx(events, "GenTTbar.top_2", gen_top_2)
    # w bosons from top quarks
    events = set_ak_column_idx(events, "GenTTbar.w_1", gen_w_1)
    events = set_ak_column_idx(events, "GenTTbar.w_2", gen_w_2)
    # b quarks from top quarks
    events = set_ak_column_idx(events, "GenTTbar.b_1", gen_b_1)
    events = set_ak_column_idx(events, "GenTTbar.b_2", gen_b_2)
    # leptons from w decay
    events = set_ak_column_idx(events, "GenTTbar.lep_1", gen_lep_1)
    events = set_ak_column_idx(events, "GenTTbar.lep_2", gen_lep_2)
    # neutrinos from w decay
    events = set_ak_column_idx(events, "GenTTbar.nu_1", gen_nu_1)
    events = set_ak_column_idx(events, "GenTTbar.nu_2", gen_nu_2)
    # light quarks from w decay
    events = set_ak_column_idx(events, "GenTTbar.q_1", gen_q_1)
    events = set_ak_column_idx(events, "GenTTbar.q_2", gen_q_2)
    # gen jets matched to b and light quarks
    events = set_ak_column_idx(events, "GenTTbar.jet_b_1", gen_jet_b_1)
    events = set_ak_column_idx(events, "GenTTbar.jet_b_2", gen_jet_b_2)
    events = set_ak_column_idx(events, "GenTTbar.jet_q_1", gen_jet_q_1)
    events = set_ak_column_idx(events, "GenTTbar.jet_q_2", gen_jet_q_2)
    # other information
    events = set_ak_column(events, "GenTTbar.is_semileptonic", is_semileptonic)

    return events
