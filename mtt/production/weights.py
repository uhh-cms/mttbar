# coding: utf-8

"""
Producers related to event weights.
"""
import law

from columnflow.production import Producer, producer
from columnflow.selection import SelectionResult

from columnflow.production.cms.pileup import pu_weight
from columnflow.production.cms.parton_shower import ps_weights
from columnflow.production.cms.btag import btag_weights, btag_wp_weights
from columnflow.production.cms.scale import murmuf_weights, murmuf_envelope_weights
from columnflow.production.cms.pdf import pdf_weights

from columnflow.util import maybe_import
from columnflow.columnar_util import fill_at

from mtt.util import has_tag, record_calls
from mtt.production.normalized_weights import normalized_weight_factory, normalized_btag_weights
from mtt.production.normalization import normalization_weights
from mtt.production.gen_top import top_pt_weight
from mtt.production.gen_v import vjets_weight
from mtt.production.lepton_weights import (
    electron_reco_id_iso_trigger_weights, electron_norm_fix_weights, muon_reco_id_iso_trigger_weights
)

ak = maybe_import("awkward")
np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)


@producer(
    uses={
        pu_weight,
    },
    # produces={
    #     pu_weight,
    # },
    mc_only=True,
)
def event_weights_to_normalize(self: Producer, events: ak.Array, results: SelectionResult, **kwargs) -> ak.Array:
    """
    Wrapper of several event weight producers that are typically called as part of SelectEvents
    since it is required to normalize them before applying certain event selections.
    """

    # compute pu weights
    events = self[pu_weight](events, **kwargs)
    # if self.has_dep(ps_weights):
    if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
        logger.debug("Compute PS weights for normalization")
        events = self[ps_weights](events, **kwargs)

    if (
            has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any) and
            not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any)
    ):
        # compute btag SF weights (for renormalization tasks)
        logger.debug("Compute btag weights for normalization")
        events = self[btag_weights](
            events,
            jet_mask=results.aux["jet_mask"],
            negative_b_score_action="ignore",
            negative_b_score_log_mode="debug",
            **kwargs,
        )

    # skip scale/pdf weights for some datasets (missing columns)
    # if self.has_dep(murmuf_envelope_weights):
    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any) and self.has_dep(murmuf_envelope_weights):  # noqa
        # compute scale weights
        logger.debug("Compute scale weights for normalization")
        events = self[murmuf_envelope_weights](events, **kwargs)

    # if self.has_dep(murmuf_weights):
    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any) and self.has_dep(murmuf_weights):
        # read out mur and weights
        logger.debug("Compute murmuf weights for normalization")
        events = self[murmuf_weights](events, **kwargs)

    # if self.has_dep(pdf_weights):
    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any) and self.has_dep(pdf_weights):
        # compute pdf weights
        logger.debug("Compute pdf weights for normalization")
        events = self[pdf_weights](
            events,
            outlier_threshold=0.99,
            outlier_action="remove",
            outlier_log_mode="debug",
            invalid_weights_action="ignore" if self.dataset_inst.has_tag("partial_lhe_weights") else "raise",
            **kwargs,
        )

    return events


@event_weights_to_normalize.init
def event_weights_to_normalize_init(self) -> None:
    # used Producers need to be set in the init or decorator
    if (
            has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any) and
            not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any)
    ):
        self.uses |= {btag_weights}

    if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {murmuf_envelope_weights, murmuf_weights}
        self.produces |= {murmuf_envelope_weights, murmuf_weights}

    if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {pdf_weights}
        self.produces |= {pdf_weights}

    if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {ps_weights}
        self.produces |= {ps_weights}


@event_weights_to_normalize.post_init
def event_weights_to_normalize_post_init(self, task: law.Task) -> None:
    # produced columns can be set in post_init to choose stored columns based on the shift
    for _cls in self.uses:
        if _cls == btag_weights and task.shift == "nominal":
            self.produces |= {btag_weights}
        elif _cls == btag_weights:
            self.produces |= self.deps[btag_weights].produced_columns
        elif task.shift == "nominal":
            self.produces |= self.deps[_cls].produced_columns
        else:
            self.produces |= {
                route for route in self.deps[_cls].produced_columns
                if not route.nano_column.endswith("_up") and not route.nano_column.endswith("_down")
            }


# renormalized weights
normalized_scale_weights = normalized_weight_factory(
    producer_name="normalized_scale_weights",
    weight_producers={murmuf_envelope_weights, murmuf_weights},
)
normalized_pdf_weights = normalized_weight_factory(
    producer_name="normalized_pdf_weights",
    weight_producers={pdf_weights},
)
normalized_pu_weights = normalized_weight_factory(
    producer_name="normalized_pu_weights",
    weight_producers={pu_weight},
)
normalized_ps_weights = normalized_weight_factory(
    producer_name="normalized_ps_weights",
    weight_producers={ps_weights},
)


@producer(
    uses={"mc_weight", "genWeight"},
    produces={"mc_weight", "genWeight"},
    mc_only=True,
)
def large_weights_killer(self: Producer, events: ak.Array, stats: dict, **kwargs) -> ak.Array:
    """
    Simple producer that sets eventweights to 0 when too large.
    """
    if self.dataset_inst.is_data:
        raise Exception("large_weights_killer is only callable for MC")

    # set mc_weight to zero when genWeight is > 0.5 for powheg HH events
    if self.dataset_inst.has_tag("is_hh") and self.dataset_inst.name.endswith("powheg"):
        # TODO: this feels very unsafe because genWeight can also be just 1 for all events. To be revisited
        weight_too_large = abs(events.genWeight) > 0.5
        logger.warning(f"found {ak.sum(weight_too_large)} HH events with genWeight > 0.5")

        events = fill_at(events, weight_too_large, "mc_weight", 0.0, value_type=np.float32)

    # check for anomalous weights and store in stats
    median_weight = ak.sort(abs(events.mc_weight))[int(len(events) / 2)]
    anomalous_weights_mask = abs(events.mc_weight) > 1000 * median_weight
    if ak.any(anomalous_weights_mask):
        logger.warning(f"found {ak.sum(anomalous_weights_mask)} events with weights > 1000 * median weight")
        stats["num_events_anomalous_weights"] += ak.sum(anomalous_weights_mask)

    return events


@producer
def weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Main event weight producer (e.g. MC generator, scale factors, normalization).
    """
    run_list = []
    logger.debug(f"weights top-level used_columns: {sorted(str(c) for c in self.used_columns)}")
    with record_calls(self, run_list):
        if self.dataset_inst.is_mc:
            # compute normalization weights
            events = self[normalization_weights](events, **kwargs)

            # compute top pT weights
            if self.dataset_inst.has_tag("is_ttbar"):
                events = self[top_pt_weight](events, **kwargs)

            # compute V+jets K factor weights
            if not has_tag("skip_kfactor_weights", self.config_inst, self.dataset_inst, operator=any) and self.dataset_inst.has_tag("is_v_jets"):  # noqa
                events = self[vjets_weight](events, **kwargs)

            if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
                events = self[electron_reco_id_iso_trigger_weights](events, **kwargs)
                events = self[electron_norm_fix_weights](events, **kwargs)

            if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
                events = self[muon_reco_id_iso_trigger_weights](events, **kwargs)

            # normalize event weights using stats
            events = self[normalized_pu_weights](events, **kwargs)

            if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
                logger.debug("Applying PS weights and normalizing them.")
                events = self[normalized_ps_weights](events, **kwargs)

            if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
                logger.debug("Applying scale weights and normalizing them.")
                events = self[normalized_scale_weights](events, **kwargs)

            if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
                logger.debug("Applying pdf weights and normalizing them.")
                events = self[normalized_pdf_weights](events, **kwargs)

            # compute btag weights
            if (
                has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any) and
                not has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any)
            ):
                logger.debug("Skipping shape based btag weights and applying fixed WP SF instead.")
                # skip shape based btag weights and apply fixed WP SF instead (for 2024)
                jet_mask = (events.Jet["pt"] < 10_000) & (abs(events.Jet["eta"]) < 2.5)
                events = self[btag_wp_weights](events, jet_mask=jet_mask, **kwargs)
            elif (
                has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any) and
                not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any)
            ):
                logger.debug("Skipping fixed WP btag weights and applying shape based SF instead.")
                # apply shape based btag weights (for 2022/23)
                # and normalize
                jet_mask = (events.Jet["pt"] >= 100) & (abs(events.Jet["eta"]) < 2.5)
                events = self[btag_weights](events, jet_mask=jet_mask, **kwargs)
                events = self[normalized_btag_weights](events, jet_mask=jet_mask, **kwargs)
            else:
                logger.warning("No btag weights applied.")

            # # compute MC weights
            # # already run in selection, not needed here?
            # events = self[mc_weight](events, **kwargs)

    logger.info_once(
        "Finished computing event weights:\n" +
        "\n".join(run_list),
    )

    return events


@weights.init
def weights_init(self: Producer) -> None:
    if getattr(self, "dataset_inst", None) and self.dataset_inst.is_mc:
        # dynamically add dependencies if running on MC
        if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {
                electron_reco_id_iso_trigger_weights,
                electron_norm_fix_weights,
                "Electron.{pt,eta}",
            }
            self.produces |= {
                electron_reco_id_iso_trigger_weights,
                electron_norm_fix_weights,
            }

        if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {
                muon_reco_id_iso_trigger_weights,
                "Muon.{pt,eta,phi}",
            }
            self.produces |= {
                muon_reco_id_iso_trigger_weights,
            }

        if not self.dataset_inst.has_tag("is_qcd"):
            self.uses |= {ps_weights}
            self.produces |= {ps_weights}

        if self.dataset_inst.has_tag("is_ttbar"):
            self.uses |= {top_pt_weight}
            self.produces |= {top_pt_weight}

        if not has_tag("skip_kfactor_weights", self.config_inst, self.dataset_inst, operator=any) and self.dataset_inst.has_tag("is_v_jets"):  # noqa
            self.uses |= {vjets_weight}
            self.produces |= {vjets_weight}

        self.uses |= {normalization_weights, normalized_pu_weights}
        self.produces |= {normalization_weights, normalized_pu_weights}

        if not has_tag("no_ps_weights", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {normalized_ps_weights}
            self.produces |= {normalized_ps_weights}

        if not has_tag("skip_scale", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {normalized_scale_weights}
            self.produces |= {normalized_scale_weights}

        if not has_tag("skip_pdf", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {normalized_pdf_weights}
            self.produces |= {normalized_pdf_weights}

        if (
            has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any) and
            not has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any)
        ):
            logger.warning_once("Using fixed wp b-tagging weights for 2024.")
            self.uses |= {btag_wp_weights}
            self.produces |= {btag_wp_weights}
        elif (
            has_tag("skip_btag_wp_weights", self.config_inst, self.dataset_inst, operator=any) and
            not has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any)
        ):
            logger.warning_once("Using shape based b-tagging weights for 2022/2023.")
            self.uses |= {btag_weights, normalized_btag_weights}
            self.produces |= {btag_weights, normalized_btag_weights}
        else:
            logger.warning_once("No btag weights producer loaded.")
