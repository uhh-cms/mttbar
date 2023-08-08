# coding: utf-8

"""
Producers for L1 prefiring weights.
"""

from __future__ import annotations

import operator as op

from collections import defaultdict
from functools import reduce

from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import, InsertableDict, DotDict
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array


np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={
        attach_coffea_behavior,
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.rawFactor",
        "Photon.pt", "Photon.eta", "Photon.phi", "Photon.mass",
    },
    produces={
        "l1_ecal_prefiring_weight",
        "l1_ecal_prefiring_weight_up",
        "l1_ecal_prefiring_weight_down",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_l1_prefiring_file=(lambda self, external_files: external_files.l1_prefiring),
    # function to determine the correction file
    get_l1_prefiring_config=(lambda self: self.config_inst.x.l1_prefiring),
)
def l1_prefiring_weights(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    L1 prefiring weight producer. Requires an external file in the config as under
    ``l1_prefiring``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "l1_prefiring": "data/json/l1_prefiring.json",
        })

    *get_l1_prefiring_file* can be adapted in a subclass in case it is stored differently in the external
    files.

    The name of the corrections as a function of the *jet* and *photon* transverse momentum, as well as the
    associated uncertainty should be given as an auxiliary entry in the config:

    .. code-block:: python

        cfg.x.l1_prefiring = DotDict.wrap({
            "jet": {
                "value": "l1_prefiring_efficiency_value_jetpt_2017BtoF",
                "error": "l1_prefiring_efficiency_error_jetpt_2017BtoF",
            },
            "photon": {
                "value": "l1_prefiring_efficiency_value_photonpt_2017BtoF",
                "error": "l1_prefiring_efficiency_error_photonpt_2017BtoF",
            },
        })

    Resources:

       - https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe?rev=3
    """

    # attach coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # identify jets that overlap with a photon
    pho_for_jet_cleaning = events.Photon[events.Photon.pt > 20]
    jet_has_photon_overlap = ~ak.all(events.Jet.metric_table(pho_for_jet_cleaning) > 0.4, axis=-1)
    jet_photon = ak.mask(events.Jet, jet_has_photon_overlap, valid_when=True)

    def get_eff(obj_name, key, obj):
        """Obtain , per-object non-prefiring probability."""
        # mark missing inputs
        input_is_none = (
            ak.is_none(obj.eta, axis=1) |
            ak.is_none(obj.pt, axis=1)
        )
        # flat views of object properties
        eta = flat_np_view(ak.fill_none(obj.eta, 0.0), axis=1)
        pt = flat_np_view(ak.fill_none(obj.pt, 0.0), axis=1)
        # get efficiencies from correctionlib evaluator
        eff_flat = self.l1_prefiring_evaluators[obj_name][key](eta, pt)
        # enforce the correct shape
        eff = layout_ak_array(eff_flat, obj.pt)
        # mask values where inputs are missing
        eff = ak.where(~input_is_none, eff, 0.0)
        return eff

    # compute the prefiring probablities and related statistical uncertainties
    eff_jet = DotDict.wrap({})
    eff_jet_photon = DotDict.wrap({})
    eff_photon = DotDict.wrap({})
    eff_jet_no_photon_overlap = DotDict.wrap({})
    for key in ("value", "error"):
        eff_jet[key] = get_eff("jet", key, events.Jet)
        eff_jet_photon[key] = get_eff("jet", key, jet_photon)
        eff_photon[key] = get_eff("photon", key, events.Photon)

        # remove photon-jet overlap
        eff_jet_no_photon_overlap[key] = ak.where(
            (eff_jet_photon.value > eff_jet.value),
            eff_jet_photon[key],
            eff_jet[key],
        )

    weights = defaultdict(lambda: [])
    for eff in (eff_jet_no_photon_overlap, eff_photon):
        val = eff.value
        err = np.sqrt((eff.value * 0.2)**2 + eff.error)

        # get systematics variations (clamped to [0, 1])
        val_up = val + err
        val_up = ak.where(val_up > 1.0, 1.0, val_up)
        val_down = val - err
        val_down = ak.where(val_down < 0.0, 0.0, val_down)

        # compute the weight from the product of non-prefiring probablities
        weights["nominal"].append(ak.prod(1.0 - val, axis=1, mask_identity=False))
        weights["up"].append(ak.prod(1.0 - val_up, axis=1, mask_identity=False))
        weights["down"].append(ak.prod(1.0 - val_down, axis=1, mask_identity=False))

    # merge jet and photon weights
    for var in list(weights):
        weights[var] = reduce(op.mul, weights[var])

    # save the weights
    events = set_ak_column(events, "l1_ecal_prefiring_weight", weights["nominal"], value_type=np.float32)
    events = set_ak_column(events, "l1_ecal_prefiring_weight_up", weights["up"], value_type=np.float32)
    events = set_ak_column(events, "l1_ecal_prefiring_weight_down", weights["down"], value_type=np.float32)

    return events


@l1_prefiring_weights.init
def l1_prefiring_weights_init(self: Producer) -> None:
    shift_inst = getattr(self, "local_shift_inst", None)
    if not shift_inst:
        return


@l1_prefiring_weights.requires
def l1_prefiring_weights_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@l1_prefiring_weights.setup
def l1_prefiring_weights_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    bundle = reqs["external_files"]

    # create the L1 prefiring weight evaluator
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        self.get_l1_prefiring_file(bundle.files).load(formatter="text"),
    )
    corrections = self.get_l1_prefiring_config()
    self.l1_prefiring_evaluators = {
        obj_name: {
            key: correction_set[correction_name]
            for key, correction_name in corrections_obj.items()
        }
        for obj_name, corrections_obj in corrections.items()
    }
