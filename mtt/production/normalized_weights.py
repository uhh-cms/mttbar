# coding: utf-8

"""
Column production methods related to generic event weights.
"""

from typing import Iterable, Callable

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, safe_div
from columnflow.columnar_util import set_ak_column  # , EMPTY_FLOAT
from columnflow.production.cms.btag import btag_weights

ak = maybe_import("awkward")
np = maybe_import("numpy")


logger = law.logger.get_logger(__name__)


def normalized_weight_factory(
    producer_name: str,
    weight_producers: Iterable[Producer],
    **kwargs,
) -> Callable:

    @producer(
        # TODO: w.produces does not work as intended anymore, so we have to initialize the Producers here
        # uses=set(weight_producers) | set().union(*[w().produced_columns for w in weight_producers]) | {"process_id"},
        cls_name=producer_name,
        mc_only=True,
        # skip the checking existence of used/produced columns because not all columns are there
        check_used_columns=False,
        check_produced_columns=False,
        # remaining produced columns are defined in the init function below
    )
    def normalized_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
        # check existence of requested weights to normalize and run producer if missing
        missing_weights = self.weight_names.difference(events.fields)
        logger.debug(f"weight_names: {self.weight_names}")
        logger.debug(f"events.fields: {sorted(events.fields)}")
        for prod in self.weight_producers:
            logger.debug(
                f"{prod}: produced={self[prod].produced_columns}, "
                f"missing_from_events={self[prod].produced_columns.difference(events.fields)}, "
                f"used_present={self[prod].used_columns.intersection(events.fields)}",
            )

        if missing_weights:
            logger.warning(f"Missing weight columns: {missing_weights}")
            # try to produce missing weights
            for prod in self.weight_producers:
                if (
                        self[prod].produced_columns.difference(events.fields) and
                        self[prod].used_columns.intersection(events.fields)
                ):
                    logger.info(f"Rerun producer {self[prod].cls_name}")
                    events = self[prod](events, **kwargs)

        # Create normalized weight columns if possible
        if not_reproduced := missing_weights.difference(events.fields):
            logger.warning(f"Weight columns {not_reproduced} could not be reproduced")

        for weight_name in self.weight_names.intersection(events.fields):
            logger.debug(f"Creating normalized weight column for {weight_name}")
            # create a weight vector starting with ones
            norm_weight_per_pid = np.ones(len(events), dtype=np.float32)

            # fill weights with a new mask per unique process id (mostly just one)
            for pid in self.unique_process_ids:
                pid_mask = events.process_id == pid
                norm_weight_per_pid[pid_mask] = self.ratio_per_pid[weight_name][pid]

            # multiply with actual weight
            norm_weight_per_pid = norm_weight_per_pid * events[weight_name]

            # store it
            norm_weight_per_pid = ak.values_astype(norm_weight_per_pid, np.float32)
            events = set_ak_column(events, f"normalized_{weight_name}", norm_weight_per_pid)

        return events

    @normalized_weight.post_init
    def normalized_weight_post_init(self: Producer, task: law.Task) -> None:
        self.weight_producers = weight_producers

        self.weight_names = set()
        for prod in self.weight_producers:
            produced = self[prod].produced_columns
            # make sure these columns actually get read from the input file
            self.uses |= produced

            for col in produced:
                col = col.string_nano_column
                if task.shift != "nominal" and (col.endswith("_up") or col.endswith("_down")):
                    continue
                if "weight" in col and "normalized" not in col and "btag" not in col:
                    self.weight_names.add(col)

        self.produces |= set(f"normalized_{weight_name}" for weight_name in self.weight_names)

    @normalized_weight.requires
    def normalized_weight_requires(self: Producer, task: law.Task, reqs: dict) -> None:
        from columnflow.tasks.selection import MergeSelectionStats
        reqs["selection_stats"] = MergeSelectionStats.req(
            task,
            branch=-1,
        )

    @normalized_weight.init
    def normalized_weight_init(self: Producer) -> None:
        self.uses |= {"process_id", *weight_producers}
        for w in weight_producers:
            inst = self.instantiate_dependency(w)
            self.uses |= inst.produced_columns

    @normalized_weight.setup
    def normalized_weight_setup(
        self: Producer, task: law.Task, reqs: dict, inputs: dict, reader_targets: law.util.InsertableDict,
    ) -> None:
        # load the selection stats
        stats = inputs["selection_stats"]["collection"][0]["stats"].load(formatter="json")

        # get the unique process ids in that dataset
        key = "sum_mc_weight_per_process"
        self.unique_process_ids = list(map(int, stats[key].keys()))

        # helper to get numerators and denominators
        def numerator_per_pid(pid):
            key = "sum_mc_weight_per_process"
            return stats[key].get(str(pid), 0.0)

        def denominator_per_pid(weight_name, pid):
            key = f"sum_mc_weight_{weight_name}_per_process"
            return stats[key].get(str(pid), 0.0)

        # extract the ratio per weight and pid
        self.ratio_per_pid = {
            weight_name: {
                pid: safe_div(numerator_per_pid(pid), denominator_per_pid(weight_name, pid))
                for pid in self.unique_process_ids
            }
            for weight_name in self.weight_names
        }

    return normalized_weight


@producer(
    uses={
        btag_weights.PRODUCES, "process_id", "Jet.{pt,eta,phi}", "njet", "ht", "nhf",
    },
    # produced columns are defined in the init function below
    mc_only=True,
    modes=["ht_njet_nhf"],
    # modes=["ht_njet_nhf", "ht_njet", "njet", "ht"],
    from_file=False,
)
def normalized_btag_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    variable_map = {
        # NOTE: might be cleaner to use the ht and njet reconstructed during the selection (and also compare?)
        "ht": ak.sum(events.Jet.pt, axis=1),
        "njet": ak.num(events.Jet.pt, axis=1),
        "nhf": events.nhf,
    }

    # sanity check
    for var in ("ht", "njet"):
        consistency_check = np.isclose(events[var], variable_map[var], rtol=0.0001)
        if not ak.all(consistency_check):
            logger.warning(f"Variable {var} is not consistent between before and after event selection. "
            f"Please check the consistency of {var} and the selection steps.")
            # raise ValueError(f"Variable {var} is not consistent between before and after event selection")

    for mode in self.modes:
        if mode not in ("ht_njet_nhf", "ht_njet", "njet", "ht"):
            raise NotImplementedError(
                f"Normalization mode {mode} not implemented (see topsf.tasks.corrections.GetBtagNormalizationSF)",
            )
        for weight_route in self[btag_weights].produced_columns:
            weight_name = weight_route.string_column
            if not weight_name.startswith("btag_weight"):
                continue

            correction_key = f"{mode}_{weight_name}"
            if correction_key not in set(self.correction_set.keys()):
                raise KeyError(f"Missing scale factor for {correction_key}")

            sf = self.correction_set[correction_key]
            inputs = [variable_map[inp.name] for inp in sf.inputs]

            norm_weight = sf.evaluate(*inputs)
            norm_weight = norm_weight * events[weight_name]
            events = set_ak_column(events, f"normalized_{mode}_{weight_name}", norm_weight, value_type=np.float32)

    return events


@normalized_btag_weights.post_init
def normalized_btag_weights_post_init(self: Producer, task: law.Task) -> None:
    # NOTE: self[btag_weights].produced_columns is empty during the `init`, therefore changed to `post_init`
    # this means that running this Producer directly on command line would not be triggered due to empty produces
    # during task initialization
    for weight_route in self[btag_weights].produced_columns:
        weight_name = weight_route.string_column
        if not weight_name.startswith("btag_weight"):
            continue
        for mode in self.modes:
            self.produces.add(f"normalized_{mode}_{weight_name}")


@normalized_btag_weights.requires
def normalized_btag_weights_requires(self: Producer, task: law.Task, reqs: dict) -> None:
    from topsf.tasks.corrections import GetBtagNormalizationSF
    reqs["btag_renormalization_sf"] = GetBtagNormalizationSF.req(task)


normalized_btag_weights_full = normalized_btag_weights.derive("normalized_btag_weights_full", cls_dict=dict(
    modes=["ht_njet_nhf", "ht_njet", "njet", "ht"],
))


@normalized_btag_weights.setup
def normalized_btag_weights_setup(
    self: Producer,
    task: law.Task,
    reqs: dict,
    inputs: dict,
    reader_targets: law.util.InsertableDict,
) -> None:
    # create the corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    if self.from_file:
        # used when the correction is stored as a JSON dict
        self.correction_set = correctionlib.CorrectionSet.from_file(
            inputs["btag_renormalization_sf"]["btag_renormalization_sf"].fn,
        )
    else:
        # used when correction is stored as a JSON string
        self.correction_set = correctionlib.CorrectionSet.from_string(
            inputs["btag_renormalization_sf"]["btag_renormalization_sf"].load(formatter="json"),
        )
