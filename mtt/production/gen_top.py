# coding: utf-8

"""
Column producers related to gen-level top quark(s).
"""
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={"GenPart.genPartIdxMother", "GenPart.pdgId", "GenPart.statusFlags"},
    produces={"GenTopDecay.*"},
    mc_only=True,
)
def gen_top_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "GenTopDecay" with one element per hard top quark.

    .. note::

        Can only be run when processing NanoAODs, i.e. during *CalibrateEvents* or *SelectEvents*.

    Each element of "GenTopDecay" contains the particles originating from the top quark decay, and is
    organized as a GenParticleArray with five or more objects in a distinct order: top quark, bottom
    quark, W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional
    decay products of the W boson (if any, then most likely radiated photons). Per event, the
    structure will be similar to:

    .. code-block:: python

        [
            # event 1
            [
                # top 1
                [t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)],
                # top 2
                [...],
            ],
            # event 2
            ...
        ]
    """
    # get local index of gen particles
    genpart_index = ak.local_index(events.GenPart, axis=1)

    # find hard top quarks
    abs_id = abs(events.GenPart.pdgId)
    t = events.GenPart[abs_id == 6]
    t_index = genpart_index[abs_id == 6]
    t = t[t.hasFlags("isHardProcess")]
    t_index = t_index[t.hasFlags("isHardProcess")]
    t = t[~ak.is_none(t, axis=1)]
    t_index = t_index[~ak.is_none(t, axis=1)]

    # distinct top quark children (b's and W's)
    t_children = t.distinctChildrenDeep[t.distinctChildrenDeep.hasFlags("isHardProcess")]

    # get b's (or very rarely other down-like quark from prompt top decay)
    down_like_quark = (
        (abs(t_children.pdgId) == 5) |
        (abs(t_children.pdgId) == 3) |
        (abs(t_children.pdgId) == 1)
    )
    b = t_children[down_like_quark][:, :, 0]

    # get W's
    w = t_children[abs(t_children.pdgId) == 24][:, :, 0]

    # distinct W children
    w_children = w.distinctChildrenDeep[w.distinctChildrenDeep.hasFlags("isHardProcess")]

    # reorder the first two W children (leptons or quarks) so that the charged lepton / down-type
    # quark is listed first (they have an odd pdgId)
    w_children_firsttwo = w_children[:, :, :2]
    w_children_firsttwo = w_children_firsttwo[(w_children_firsttwo.pdgId % 2 == 0) * 1]
    w_children_rest = w_children[:, :, 2:]

    # concatenate to create the structure to return
    t_decay = ak.concatenate(
        [
            t[:, :, None],
            b[:, :, None],
            w[:, :, None],
            w_children_firsttwo,
            w_children_rest,
        ],
        axis=2,
    )

    # save the column
    events = set_ak_column(events, "GenTopDecay", t_decay)

    return events


@gen_top_decay_products.skip
def gen_top_decay_products_skip(self: Producer) -> bool:
    """
    Custom skip function that checks whether the dataset is a MC simulation containing top
    quarks in the first place.
    """
    # never skip when there is not dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")


@producer(
    uses={
        "GenPart.pt",
        "GenPart.eta",
        "GenPart.phi",
        "GenPart.mass",
        "GenPart.pdgId",
        "GenPart.statusFlags",
    },
    produces={
        "GenPartonTop.pt",
        "GenPartonTop.eta",
        "GenPartonTop.phi",
        "GenPartonTop.mass",
        "GenPartonTop.pdgId",
    },
)
def gen_parton_top(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Produce parton-level top quarks (before showering and detector simulation).
    """
    # find parton-level top quarks
    abs_id = abs(events.GenPart.pdgId)
    t = events.GenPart[abs_id == 6]
    t = t[t.hasFlags("isLastCopy")]
    t = t[~ak.is_none(t, axis=1)]

    # write columns
    events = set_ak_column(events, "GenPartonTop", t)

    return events


@gen_parton_top.skip
def gen_parton_top_skip(self: Producer) -> bool:
    """
    Custom skip function that checks whether the dataset is a MC simulation containing top
    quarks in the first place.
    """
    # never skip when there is not dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")


@producer(
    uses={
        "GenPartonTop.pt",
    },
    produces={
        "top_pt_weight", "top_pt_weight_up", "top_pt_weight_down",
    },
)
def top_pt_weight(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Compute SF to be used for top pt reweighting.

    The SF should *only be applied in ttbar MC* as an event weight computed
    based on the gen-level top quark transverse momenta.
    """

    # fail if not run in ttbar simulation
    if not self.dataset_inst.has_tag("is_ttbar"):
        raise Exception(f"gen_top_pt_weight should only run for ttbar dataset, got {self.dataset_inst}")

    # get SF function parameters from config
    params = self.config_inst.x.top_pt_reweighting_params

    # clamp top pT < 500 GeV and evaluate SF function
    pt_clamped = ak.where(events.GenPartonTop.pt > 500.0, 500.0, events.GenPartonTop.pt)
    sf = ak.pad_none(np.exp(params["a"] + params["b"] * pt_clamped), 2)

    # compute weight from SF product for top and anti-top
    weight = np.sqrt(sf[:, 0] * sf[:, 1])

    # write out weights
    events = set_ak_column(events, "top_pt_weight", ak.fill_none(weight, 1.0))
    events = set_ak_column(events, "top_pt_weight_up", ak.fill_none(weight * 1.5, 1.0))
    events = set_ak_column(events, "top_pt_weight_down", ak.fill_none(weight * 0.5, 1.0))

    return events


@top_pt_weight.skip
def top_pt_weight_skip(self: Producer) -> bool:
    """
    Skip if running on anything except ttbar MC simulation.
    """
    # never skip when there is no dataset
    if not getattr(self, "dataset_inst", None):
        return False

    return self.dataset_inst.is_data or not self.dataset_inst.has_tag("is_ttbar")
