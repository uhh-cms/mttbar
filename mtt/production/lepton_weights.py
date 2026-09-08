# coding: utf-8

"""
Producers related to lepton event weights.
"""
import law

from columnflow.production import Producer, producer
from columnflow.production.cms.muon import muon_weights, MuonSFConfig
from columnflow.production.cms.electron import electron_weights, ElectronSFConfig

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from mtt.util import has_tag

ak = maybe_import("awkward")
np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)


def high_pt_muon_reco(producer, corrector, variable_map):
    # for muon reco SFs, use p and eta as variables and apply a mask to only compute weights for high pT muons
    momentum = variable_map["pt"] * np.cosh(variable_map["eta"])
    variable_map["p"] = momentum
    return variable_map


# muon reco, id, iso, and trigger weights for low and high pT muons
muon_reco_weight_low_pt = muon_weights.derive(
    "muon_reco_weight_low_pt",
    cls_dict={
        "weight_name": "muon_reco_weight_low_pt",
        "get_muon_config": (
            lambda self: MuonSFConfig.new(self.config_inst.x.muon_reco_low_pt_sf_config)
        ),
        "update_corrector_variables": (
            lambda self, corrector, variables: high_pt_muon_reco(self, corrector, variables)
        ),
        "get_muon_file": (lambda self, external_files: external_files.muon_low_pt_sf),
    },
)
muon_reco_weight_high_pt = muon_weights.derive(
    "muon_reco_weight_high_pt",
    cls_dict={
        "weight_name": "muon_reco_weight_high_pt",
        "get_muon_config": (
            lambda self: MuonSFConfig.new(self.config_inst.x.muon_reco_high_pt_sf_config)
        ),
        "update_corrector_variables": (
            lambda self, corrector, variables: high_pt_muon_reco(self, corrector, variables)
        ),
        "get_muon_file": (lambda self, external_files: external_files.muon_high_pt_sf),
    },
)

muon_id_weight_low_pt = muon_weights.derive(
    "muon_id_weight_low_pt",
    cls_dict={
        "weight_name": "muon_id_weight_low_pt",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_id_weight_low_pt_sf_config)),
        "get_muon_file": (lambda self, external_files: external_files.muon_low_pt_sf),
    },
)
muon_id_weight_high_pt = muon_weights.derive(
    "muon_id_weight_high_pt",
    cls_dict={
        "weight_name": "muon_id_weight_high_pt",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_id_weight_high_pt_sf_config)),
        "get_muon_file": (lambda self, external_files: external_files.muon_high_pt_sf),
    },
)

muon_iso_weight_low_pt = muon_weights.derive(
    "muon_iso_weight_low_pt",
    cls_dict={
        "weight_name": "muon_iso_weight_low_pt",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_iso_weight_low_pt_sf_config)),
        "get_muon_file": (lambda self, external_files: external_files.muon_low_pt_sf),
    },
)
muon_iso_weight_high_pt = muon_weights.derive(
    "muon_iso_weight_high_pt",
    cls_dict={
        "weight_name": "muon_iso_weight_high_pt",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_iso_weight_high_pt_sf_config)),
        "get_muon_file": (lambda self, external_files: external_files.muon_high_pt_sf),
    },
)

muon_trigger_weight_low_pt = muon_weights.derive(
    "muon_trigger_weight_low_pt",
    cls_dict={
        "weight_name": "muon_trigger_weight_low_pt",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_trigger_weight_low_pt_sf_config)),
        "get_muon_file": (lambda self, external_files: external_files.muon_low_pt_sf),
    },
)
muon_trigger_weight_high_pt = muon_weights.derive(
    "muon_trigger_weight_high_pt",
    cls_dict={
        "weight_name": "muon_trigger_weight_high_pt",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_trigger_weight_high_pt_sf_config)),
        "get_muon_file": (lambda self, external_files: external_files.muon_high_pt_sf),
    },
)

# electron reco, id, iso, and trigger weights for low and high pT electrons
electron_reco_weight_low_pt = electron_weights.derive(
    "electron_reco_weight_low_pt",
    cls_dict={
        "weight_name": "electron_reco_weight_low_pt",
        "get_electron_config": (
            lambda self: ElectronSFConfig.new(self.config_inst.x.electron_reco_weight_low_pt_sf_config)
        ),
    },
)
electron_reco_weight_high_pt = electron_weights.derive(
    "electron_reco_weight_high_pt",
    cls_dict={
        "weight_name": "electron_reco_weight_high_pt",
        "get_electron_config": (
            lambda self: ElectronSFConfig.new(self.config_inst.x.electron_reco_weight_high_pt_sf_config)
        ),
    },
)

electron_id_iso_weight_low_pt = electron_weights.derive(
    "electron_id_iso_weight_low_pt",
    cls_dict={
        "weight_name": "electron_id_iso_weight_low_pt",
        "get_electron_config": (
            lambda self: ElectronSFConfig.new(self.config_inst.x.electron_id_iso_weight_low_pt_sf_config)
        ),
    },
)
electron_id_weight_high_pt = electron_weights.derive(
    "electron_id_weight_high_pt",
    cls_dict={
        "weight_name": "electron_id_weight_high_pt",
        "get_electron_config": (
            lambda self: ElectronSFConfig.new(self.config_inst.x.electron_id_weight_high_pt_sf_config)
        ),
    },
)
electron_iso_weight_high_pt = electron_weights.derive(
    "electron_iso_weight_high_pt",
    cls_dict={
        "weight_name": "electron_iso_weight_high_pt",
        "get_electron_config": (
            lambda self: ElectronSFConfig.new(self.config_inst.x.electron_iso_weight_high_pt_sf_config)
        ),
    },
)

electron_trigger_weight_low_pt = electron_weights.derive(
    "electron_trigger_weight_low_pt",
    cls_dict={
        "weight_name": "electron_trigger_weight_low_pt",
        "get_electron_config": (
            lambda self: ElectronSFConfig.new(self.config_inst.x.electron_trigger_low_pt_sf_config)
        ),
        "get_electron_file": (lambda self, external_files: external_files.electron_trigger_low_pt_sf),
    },
)
electron_trigger_weight_high_pt = electron_weights.derive(
    "electron_trigger_weight_high_pt",
    cls_dict={
        "weight_name": "electron_trigger_weight_high_pt",
        "get_electron_config": (
            lambda self: ElectronSFConfig.new(self.config_inst.x.electron_trigger_high_pt_sf_config)
        ),
        "get_electron_file": (lambda self, external_files: external_files.electron_trigger_high_pt_sf),
    },
)


@producer(
    uses={"Electron.pt"},
    produces={"electron_norm_fix_weight"},
    mc_only=True,
)
def electron_norm_fix_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Add a column with a flat 1.2 SF for testing reasons. Do not use for physics analysis.
    This is just to test the effect of a flat SF on the normalization of the event weights.
    """
    electron_mask = ak.num(events.Electron) == 1
    electron_norm_weight = ak.where(electron_mask, 1.2, 1.0)
    events = set_ak_column(events, "electron_norm_fix_weight", electron_norm_weight)
    logger.warning("electron_norm_fix_weights: flat 1.2 SF. Do not use for physics analysis!")
    return events


# ---------------------------------------------------------------------------
# shared helpers for combining pt-regime and weight-type SF components
# ---------------------------------------------------------------------------

def _combine_pt_regimes(
    self,
    events: ak.Array,
    lepton_key: str,       # "muon" or "electron"
    type_name: str,        # "reco", "id", "iso", "trigger"
    entries: list,          # [(sub_producer, mask), ...] active pt-regime components
    has_shifts: bool,      # whether sub-producers provide _up/_down columns
    **kwargs,
) -> tuple:
    """
    Runs each (sub_producer, mask) pair for one weight type across whichever pt regimes
    are currently wired up, and multiplies them into a single combined column per type.
    Valid because the masks are mutually exclusive per event: multiplying "_up" variants
    together does not double-count any single lepton's uncertainty, it just picks the
    regime-appropriate value per event.
    """
    suffixes = ("", "_up", "_down") if has_shifts else ("",)
    pt_col = events[lepton_key.capitalize()]["pt"]
    totals = {suffix: ak.ones_like(pt_col) * 1.0 for suffix in suffixes}

    for sub_producer, mask in entries:
        mask_kwarg = {f"{lepton_key}_mask": mask}
        events = self[sub_producer](events, **mask_kwarg, **kwargs)
        base_name = sub_producer.cls_name
        for suffix in suffixes:
            totals[suffix] = totals[suffix] * events[f"{base_name}{suffix}"]

    for suffix in suffixes:
        events = set_ak_column(events, f"{lepton_key}_{type_name}_weight{suffix}", totals[suffix])

    return events, totals


def _combine_weight_types(
    self,
    events: ak.Array,
    lepton_key: str,
    per_type_totals: dict,   # {"reco": {"": arr, "_up": arr, "_down": arr}, "id": {...}, ...}
) -> ak.Array:
    """
    Builds the overall <lepton>_weight (product of all active type nominals), plus one
    _up/_down pair PER TYPE that has shifts, each varying only that type's contribution
    while holding the others at nominal. Types without shifts (has_shifts=False) only
    contribute to the nominal product.
    """
    types = list(per_type_totals.keys())

    nominal = None
    for type_name in types:
        nominal = per_type_totals[type_name][""] if nominal is None else nominal * per_type_totals[type_name][""]
    events = set_ak_column(events, f"{lepton_key}_weight", nominal)

    for shifted_type in types:
        if "_up" not in per_type_totals[shifted_type]:
            continue  # this type carries no shift-source columns
        for direction in ("_up", "_down"):
            combined = None
            for type_name in types:
                factor = (
                    per_type_totals[type_name][direction]
                    if type_name == shifted_type
                    else per_type_totals[type_name][""]
                )
                combined = factor if combined is None else combined * factor
            events = set_ak_column(events, f"{lepton_key}_weight_{shifted_type}{direction}", combined)

    return events


# ---------------------------------------------------------------------------
# type configs: single source of truth for both the run function and .init()
# ---------------------------------------------------------------------------

def _muon_type_config(config_inst, mask_low_pt, mask_high_pt) -> dict:
    """
    Maps each weight type to (skip_tag, [(sub_producer, mask), ...]).
    Masks are only used at run time; pass (None, None) when only the structure
    (skip_tags / sub_producers) is needed, e.g. in .init().
    Commented-out entries are SF components not yet available; uncomment once released.
    """
    return {
        "reco": (
            "skip_muon_reco_weights",
            [
                # (muon_reco_weight_low_pt, mask_low_pt),  # not needed per MUON POG recommendation
                (muon_reco_weight_high_pt, mask_high_pt),
            ],
        ),
        "id": (
            "skip_muon_id_weights",
            [
                (muon_id_weight_low_pt, mask_low_pt),
                (muon_id_weight_high_pt, mask_high_pt),
            ],
        ),
        "iso": (
            "skip_muon_iso_weights",
            [
                (muon_iso_weight_low_pt, mask_low_pt),
                # (muon_iso_weight_high_pt, mask_high_pt),  # not yet available
            ],
        ),
        "trigger": (
            "skip_muon_trigger_weights",
            [
                (muon_trigger_weight_low_pt, mask_low_pt),
                # (muon_trigger_weight_high_pt, mask_high_pt),  # not yet available
            ],
        ),
    }


def _electron_type_config(config_inst, mask_low_pt, mask_high_pt) -> dict:
    """
    Same structure as _muon_type_config. NOTE: assumes electron_weights derived producers
    provide _up/_down columns (has_shifts=True below) — verify against ElectronSFConfig /
    your columnflow version before relying on the electron shift columns; if they don't,
    set has_shifts=False at the call sites and drop the shift config block for electrons.
    """
    return {
        "reco": (
            "skip_electron_reco_weights",
            [
                (electron_reco_weight_low_pt, mask_low_pt),
                (electron_reco_weight_high_pt, mask_high_pt),
            ],
        ),
        "id_iso_low": (
            "skip_electron_id_weights",  # or a new dedicated tag if id/iso can be independently skipped
            [(electron_id_iso_weight_low_pt, mask_low_pt)],
        ),
        "id": (
            "skip_electron_id_weights",
            [(electron_id_weight_high_pt, mask_high_pt)],
        ),
        "iso": (
            "skip_electron_iso_weights",
            [
                # (electron_iso_weight_high_pt, mask_high_pt),  # not yet available
            ],
        ),
        "trigger": (
            "skip_electron_trigger_weights",
            [
                (electron_trigger_weight_low_pt, mask_low_pt),
                # (electron_trigger_weight_high_pt, mask_high_pt),  # not yet available
            ],
        ),
    }


# ---------------------------------------------------------------------------
# muon producer
# ---------------------------------------------------------------------------

@producer(mc_only=True)
def muon_reco_id_iso_trigger_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to compute muon reco/id/iso/trigger weights, combined across pt regimes,
    plus an overall muon_weight with one up/down shift pair per active weight type.
    """
    pt_threshold = self.config_inst.x.lepton_selection["mu"]["min_pt"]["high_pt"]
    base_mask = (events.Muon["pt"] >= 30) & (abs(events.Muon["eta"]) < 2.4)
    mask_low_pt = base_mask & (events.Muon["pt"] < pt_threshold)
    mask_high_pt = base_mask & (events.Muon["pt"] >= pt_threshold)

    type_config = _muon_type_config(self.config_inst, mask_low_pt, mask_high_pt)

    per_type_totals = {}
    for type_name, (skip_tag, entries) in type_config.items():
        if has_tag(skip_tag, self.config_inst, self.dataset_inst, operator=any):
            continue
        events, totals = _combine_pt_regimes(
            self, events, "muon", type_name, entries, has_shifts=True, **kwargs,
        )
        per_type_totals[type_name] = totals

    logger.warning(
        "muon_iso and muon_trigger weights for high pT muons are not computed. "
        "Remember to uncomment the lines in _muon_type_config as soon as they are available."
    )

    events = _combine_weight_types(self, events, "muon", per_type_totals)
    return events


@muon_reco_id_iso_trigger_weights.init
def muon_reco_id_iso_trigger_weights_init(self) -> None:
    # masks not needed here, only which (sub_producer, skip_tag) pairs are active
    type_config = _muon_type_config(self.config_inst, None, None)

    active_types = []
    for type_name, (skip_tag, entries) in type_config.items():
        if has_tag(skip_tag, self.config_inst, self.dataset_inst, operator=any):
            continue
        active_types.append(type_name)

        for sub_producer, _ in entries:
            self.uses.add(sub_producer)
            self.produces.add(sub_producer)

        self.produces.add(f"muon_{type_name}_weight")
        self.produces.add(f"muon_{type_name}_weight_up")
        self.produces.add(f"muon_{type_name}_weight_down")

    self.produces.add("muon_weight")
    for type_name in active_types:
        self.produces.add(f"muon_weight_{type_name}_up")
        self.produces.add(f"muon_weight_{type_name}_down")


# ---------------------------------------------------------------------------
# electron producer
# ---------------------------------------------------------------------------

@producer(mc_only=True)
def electron_reco_id_iso_trigger_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to compute electron reco/id/iso/trigger weights, combined across pt regimes,
    plus an overall electron_weight with one up/down shift pair per active weight type.
    """
    pt_threshold = self.config_inst.x.lepton_selection["e"]["min_pt"]["high_pt"]
    base_mask = events.Electron["pt"] >= 25
    mask_low_pt = base_mask & (events.Electron["pt"] < pt_threshold)
    mask_high_pt = base_mask & (events.Electron["pt"] >= pt_threshold)

    type_config = _electron_type_config(self.config_inst, mask_low_pt, mask_high_pt)

    per_type_totals = {}
    for type_name, (skip_tag, entries) in type_config.items():
        if has_tag(skip_tag, self.config_inst, self.dataset_inst, operator=any):
            continue
        events, totals = _combine_pt_regimes(
            self, events, "electron", type_name, entries, has_shifts=True, **kwargs,
        )
        per_type_totals[type_name] = totals

    logger.warning(
        "electron_iso and electron_trigger weights for high pT electrons are not computed. "
        "Remember to uncomment the lines in _electron_type_config as soon as they are available."
    )

    events = _combine_weight_types(self, events, "electron", per_type_totals)
    return events


@electron_reco_id_iso_trigger_weights.init
def electron_reco_id_iso_trigger_weights_init(self) -> None:
    type_config = _electron_type_config(self.config_inst, None, None)

    active_types = []
    for type_name, (skip_tag, entries) in type_config.items():
        if has_tag(skip_tag, self.config_inst, self.dataset_inst, operator=any):
            continue
        active_types.append(type_name)

        for sub_producer, _ in entries:
            self.uses.add(sub_producer)
            self.produces.add(sub_producer)

        self.produces.add(f"electron_{type_name}_weight")
        self.produces.add(f"electron_{type_name}_weight_up")
        self.produces.add(f"electron_{type_name}_weight_down")

    self.produces.add("electron_weight")
    for type_name in active_types:
        self.produces.add(f"electron_weight_{type_name}_up")
        self.produces.add(f"electron_weight_{type_name}_down")
