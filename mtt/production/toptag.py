# coding: utf-8

"""
Column producers related to top-tagged jets.
"""

from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.columnar_util import set_ak_column, flat_np_view, layout_ak_array


ak = maybe_import("awkward")
np = maybe_import("numpy")


@producer(
    uses={
        "Muon.pt", "Muon.eta",
    },
    produces={
        "toptag_weight", "toptag_weight_up", "toptag_weight_down",
    },
    # only run on mc
    mc_only=True,
    # function to determine the correction file
    get_toptag_file=(lambda self, external_files: external_files.toptag_sf),
    # function to determine the toptag weight config
    get_toptag_config=(lambda self: self.config_inst.x.toptag_sf_config),
)
def toptag_weights(
    self: Producer,
    events: ak.Array,
    toptag_mask: ak.Array | type(Ellipsis) = Ellipsis,
    **kwargs,
) -> ak.Array:
    """
    Creates weights from top-tagging scale factors using the correctionlib. Requires an external file in the config under
    ``toptag_sf``:

    .. code-block:: python

        cfg.x.external_files = DotDict.wrap({
            "toptag_sf": "/afs/cern.ch/user/d/dsavoiu/public/mirrors/cms-jet-JSON_Format-54860a23/JMAR/DeepAK8/2017_DeepAK8_Top.json",  # noqa
        })

    *get_toptag_file* can be adapted in a subclass in case it is stored differently in the external
    files.

    The name of the correction set and the working point string for the weight evaluation should be
    given as an auxiliary entry in the config:

    .. code-block:: python

        cfg.x.toptag_sf = DotDict.wrap({
            "name": "DeepAK8_Top_MassDecorr",
            "wp": 1p0,
        })

    *get_toptag_config* can be adapted in a subclass in case it is stored differently in the config.

    Optionally, a *toptag_mask* can be supplied to compute the scale factor weight based only on a
    subset of toptags.
    """
    # get top-tagged jet(s)
    topjet = events.FatJetTopTagDeltaRLepton

    # flat absolute eta and pt views
    eta = flat_np_view(topjet.eta, axis=1)
    pt = flat_np_view(topjet.pt, axis=1)

    # loop over systematics
    for syst, postfix in [
        ("nom", ""),
        ("up", "_up"),
        ("down", "_down"),
    ]:
        sf_flat = self.toptag_sf_corrector(eta, pt, syst, self.toptag_sf_wp)

        # add the correct layout to it
        sf = layout_ak_array(sf_flat, topjet.pt)

        # create the product over all toptag SFs in one event
        weight = ak.prod(sf, axis=1, mask_identity=False)

        # store it
        events = set_ak_column(events, f"toptag_weight{postfix}", weight, value_type=np.float32)

    return events


@toptag_weights.requires
def toptag_weights_requires(self: Producer, reqs: dict) -> None:
    if "external_files" in reqs:
        return

    from columnflow.tasks.external import BundleExternalFiles
    reqs["external_files"] = BundleExternalFiles.req(self.task)


@toptag_weights.setup
def toptag_weights_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    bundle = reqs["external_files"]

    # create the corrector
    import correctionlib
    correctionlib.highlevel.Correction.__call__ = correctionlib.highlevel.Correction.evaluate
    correction_set = correctionlib.CorrectionSet.from_string(
        self.get_toptag_file(bundle.files).load(formatter="text"),
    )
    toptag_sf_config = self.get_toptag_config()
    self.toptag_sf_corrector = correction_set[toptag_sf_config["name"]]
    self.toptag_sf_wp = toptag_sf_config["wp"]

    # check versions
    assert self.toptag_sf_corrector.version in [1]
