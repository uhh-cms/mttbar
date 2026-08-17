# coding: utf-8

"""
Column production methods related to sample normalization event weights.
"""
import law

from collections import defaultdict

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from law.util import InsertableDict


np = maybe_import("numpy")
sp = maybe_import("scipy")
maybe_import("scipy.sparse")
ak = maybe_import("awkward")


@producer(
    uses={"process_id", "mc_weight"},
    produces={"normalization_weight"},
    # only run on mc
    mc_only=True,
)
def normalization_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Uses luminosity information of internal py:attr:`config_inst`, the cross section of a process
    obtained through :py:class:`process_ids` and the sum of event weights from the
    py:attr:`selection_stats` attribute to assign each event a normalization weight.

    TODO: normalize based on parent process cross section if leaf process has no associated xs.
    """
    # get the lumi
    lumi = self.config_inst.x.luminosity.nominal

    # read the cross section per process from the lookup table
    process_id = np.asarray(events.process_id)
    xs = np.array(self.xs_table[0, process_id].todense())[0]

    # read the sum of event weights per process from the lookup table
    sum_weights = np.array(self.sum_weights_table[0, process_id].todense())[0]

    # compute the weight and store it
    norm_weight = events.mc_weight * lumi * xs / sum_weights
    events = set_ak_column(events, "normalization_weight", norm_weight, value_type=np.float32)

    return events


@normalization_weights.requires
def normalization_weights_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    """
    Adds the requirements needed by the underlying py:attr:`task` to access selection stats into
    *reqs*.
    """
    from columnflow.tasks.selection import MergeSelectionStats
    reqs["selection_stats"] = MergeSelectionStats.req(
        task,
        branch=-1,
    )


@normalization_weights.setup
def normalization_weights_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: InsertableDict,
) -> None:
    """
    Sets up objects required by the computation of normalization weights and stores them as instance
    attributes:

        - py:attr:`selection_stats`: The stats dict loaded from the output of MergeSelectionsStats.
        - py:attr:`sum_weights_table`: A sparse array serving as a lookup table for the sum of event
          weights per process id.
        - py:attr:`xs_table`: A sparse array serving as a lookup table for cross sections of all
          processes known to the config of the task, with keys being process ids.
    """
    # load the selection stats
    selection_stats = inputs["selection_stats"]["collection"][0]["stats"].load(formatter="json")

    # retrieve per-process MC weights
    sum_mc_weights = {
        int(process_id): sum_weights
        for process_id, sum_weights in selection_stats["sum_mc_weight_per_process"].items()
    }

    # resolve process IDs to process objects
    process_insts = {
        process_id: (
            self.config_inst.get_process(process_id)
            if self.config_inst.has_process(process_id)
            else None
        )
        for process_id in sum_mc_weights
    }

    # ensure that the selection stats do not contain any process that was not previously registered
    unregistered_process_ids = {
        process_id
        for process_id, process_inst in process_insts.items()
        if process_inst is None
    }
    if unregistered_process_ids:
        id_str = ",".join(map(str, sorted(unregistered_process_ids)))
        raise Exception(
            f"selection stats contain ids ({id_str}) of processes that were not previously " +
            f"registered to the config '{self.config_inst.name}'",
        )

    # for the lookup tables below, determine the maximum process id
    max_id = max(process_insts)

    #
    # subprocess handling
    #

    # map process IDs to main processes whose XS should be used for
    # normalization (subprocesses will be mapped to parent processes)
    main_process_insts = {}
    for process_id, process_inst in process_insts.items():
        # skip if not marked as a subprocess
        if not process_inst.has_tag("is_subprocess"):
            continue

        # look up parent processes
        parent_processes = [
            proc
            for proc in process_inst.parent_processes
            if not proc.has_tag("is_subprocess")
        ]

        # ensure there is exactly one parent process
        if len(parent_processes) != 1:
            name_str = ",".join(p.name for p in parent_processes)
            raise Exception(
                f"cannot assign a normalization weight for subprocess '{process_inst.name}' with "
                f"unexpected number of parent processes ({len(parent_processes)}): {name_str}",
            )

        main_process_insts[process_id] = parent_processes[0]

    # build sums of weights for main processes
    main_process_sum_weights = defaultdict(float)
    for process_inst in process_insts.values():
        # resolve subprocesses
        main_process_inst = main_process_insts.get(process_inst.id, process_inst)
        main_process_sum_weights[main_process_inst.id] += sum_mc_weights[process_inst.id]

    #
    # lookup tables
    #

    # event weight sum lookup table for all known processes
    sum_weights_table = sp.sparse.lil_matrix((1, max_id + 1), dtype=np.float32)

    # cross section lookup table for all known processes
    xs_table = sp.sparse.lil_matrix((1, max_id + 1), dtype=np.float32)

    # populate lookup tables
    for process_id, process_inst in process_insts.items():
        # resolve subprocesses
        main_process_inst = main_process_insts.get(process_inst.id, process_inst)

        # fill event sum
        sum_weights_table[0, process_id] = main_process_sum_weights[main_process_inst.id]

        # fill cross section
        xs_table[0, process_id] = main_process_inst.get_xsec(
            self.config_inst.campaign.ecm,
        ).nominal

    # store lookup tables as instance attributes
    self.sum_weights_table = sum_weights_table
    self.xs_table = xs_table
