# coding: utf-8

"""
Selection methods for m(ttbar).
"""

from collections import defaultdict
from typing import Tuple

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.calibration.jets import ak_random  # TODO: move function

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.production.categories import category_ids
from columnflow.production.mc_weight import mc_weight
from columnflow.production.processes import process_ids

from mtt.selection.util import masked_sorted_indices
from mtt.selection.general import increment_stats, jet_energy_shifts
from mtt.selection.cutflow_features import cutflow_features

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    #uses={
    #    #deterministic_seeds,  # TODO
    #},
)
def check_early(
    self: Selector,
    events: ak.Array,
    trigger_config: dict,
    **kwargs,
) -> ak.Array:
    """
    For Data, check if event is in the early run period (before some triggers were deployed).

    For MC, decide randomly if an event should be considered "early" based on a configurable
    percentage of the total number of events.

    Returns a boolean array containing `True` if the event is considered "early", otherwise
    `False`.
    """
    # determine if in early run period (or MC equivalent)
    if self.dataset_inst.is_mc:
        # in MC, by predefined event fraction using uniformly distributed random numbers

        # use event numbers in chunk to seed random number generator
        # TODO: use seeds!
        rand_gen = np.random.Generator(np.random.SFC64(events.event.to_list()))

        # uniformly distributed random numbers in [0, 100]
        random_percent = ak_random(
            ak.zeros_like(events.event),
            ak.ones_like(events.event) * 100,
            rand_func=rand_gen.uniform,
        )

        condition_early = (
            random_percent < trigger_config.highpt.early.mc_trigger_percent
        )
    else:
        # in data, by run number
        condition_early = (
            events.run <= trigger_config.highpt.early.run_range_max
        )

    return condition_early
