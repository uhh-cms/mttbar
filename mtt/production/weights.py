# coding: utf-8

"""
Producers related to event weights.
"""

from columnflow.production import Producer, producer
from columnflow.production.cms.btag import btag_weights, btag_wp_weights
from columnflow.production.cms.electron import electron_weights, ElectronSFConfig
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights, MuonSFConfig
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.pileup import pu_weight
from columnflow.util import maybe_import

from mtt.production.gen_top import top_pt_weight
from mtt.production.gen_v import vjets_weight
# from mtt.production.l1_prefiring import l1_prefiring_weights
from mtt.production.toptag import toptag_weights

ak = maybe_import("awkward")


muon_id_weights = muon_weights.derive(
    "muon_id_weights",
    cls_dict={
        "weight_name": "muon_id_weight",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_iso_sf_config)),
    }
)
muon_iso_weights = muon_weights.derive(
    "muon_iso_weights",
    cls_dict={
        "weight_name": "muon_iso_weight",
        "get_muon_config": (lambda self: MuonSFConfig.new(self.config_inst.x.muon_id_sf_config)),
    }
)

electron_reco_weights = electron_weights.derive(
    "electron_reco_weights",
    cls_dict={
        "weight_name": "electron_reco_weight",
        "get_electron_config": (lambda self: ElectronSFConfig.new(self.config_inst.x.electron_reco_sf_config)),
    }
)
electron_id_iso_weights = electron_weights.derive(
    "electron_id_iso_weights",
    cls_dict={
        "weight_name": "electron_id_iso_weight",
        "get_electron_config": (lambda self: ElectronSFConfig.new(self.config_inst.x.electron_id_iso_sf_config)),
    }
)


@producer(
    uses={muon_id_weights, muon_iso_weights},
    produces={muon_id_weights, muon_iso_weights},
    mc_only=True,
)
def muon_id_iso_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to compute muon ID and isolation weights separately.
    """
    lep_sel = self.config_inst.x.lepton_selection["mu"]
    muon_mask = (events.Muon["pt"] >= 30) & (abs(events.Muon["eta"]) < 2.4)
    events = self[muon_id_weights](events, muon_mask=muon_mask, **kwargs)
    # only apply iso weights to muons in low pt regime
    muon_mask = muon_mask & (events.Muon["pt"] < lep_sel["min_pt"]["high_pt"])
    events = self[muon_iso_weights](events, muon_mask=muon_mask, **kwargs)
    return events


@producer(
    uses={electron_reco_weights, electron_id_iso_weights},
    produces={electron_reco_weights, electron_id_iso_weights},
    mc_only=True,
)
def electron_reco_id_iso_weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Producer to compute electron reconstruction, ID and isolation weights separately.
    """
    lep_sel = self.config_inst.x.lepton_selection["e"]
    if self.config_inst.x.year in [2022, 2023]:
        electron_mask = (events.Electron["pt"] >= 35)
    elif self.config_inst.x.year == 2024:
        electron_mask = ((events.Electron["pt"] >= 20.0) & (events.Electron["pt"] < 1000.0))
    events = self[electron_reco_weights](events, electron_mask=electron_mask, **kwargs)
    # only apply iso weights to electrons in low pt regime
    electron_mask = electron_mask & (events.Electron["pt"] < lep_sel["min_pt"]["high_pt"])
    events = self[electron_id_iso_weights](events, electron_mask=electron_mask, **kwargs)
    return events


@producer
def weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Main event weight producer (e.g. MC generator, scale factors, normalization).
    """
    if self.dataset_inst.is_mc:
        # compute electron weights
        events = self[electron_reco_id_iso_weights](events, **kwargs)

        # compute muon weights
        events = self[muon_id_iso_weights](events, **kwargs)

        # compute btag weights
        if self.config_inst.x.year in [2022, 2023]:
            jet_mask = (events.Jet["pt"] >= 100) & (abs(events.Jet["eta"]) < 2.5)
            events = self[btag_weights](events, jet_mask=jet_mask, **kwargs)
        # different for 2024 and not for qcd datasets
        elif self.config_inst.x.year == 2024:
            jet_mask = (events.Jet["pt"] < 10_000) & (abs(events.Jet["eta"]) < 2.5)
            events = self[btag_wp_weights](events, jet_mask=jet_mask, **kwargs)

        # FIXME: not all weights are available for run 3
        if self.config_inst.x.run == 2:
            # # compute L1 prefiring weights
            # FIXME: why is this broken?
            # events = self[l1_prefiring_weights](events, **kwargs)

            # compute V+jets K factor weights
            if self.dataset_inst.has_tag("is_v_jets"):
                events = self[vjets_weight](events, **kwargs)

            # compute top-tagging scale factor weights
            if self.dataset_inst.has_tag("has_top"):
                events = self[toptag_weights](events, **kwargs)

        # # compute top pT weights (disabled for now)
        # if self.dataset_inst.has_tag("is_sm_ttbar"):
        #     events = self[top_pt_weight](events, **kwargs)

        # compute normalization weights
        events = self[normalization_weights](events, **kwargs)

        # compute MC weights
        events = self[mc_weight](events, **kwargs)

        events = self[pu_weight](events, **kwargs)

    return events


@weights.init
def weights_init(self: Producer) -> None:
    if getattr(self, "dataset_inst", None) and self.dataset_inst.is_mc:
        # dynamically add dependencies if running on MC
        self.uses |= {
            electron_reco_id_iso_weights, muon_id_iso_weights,
            # btag_weights,
            normalization_weights,
            pu_weight,
            mc_weight,
            top_pt_weight,
            "Muon.{pt,eta,phi}",
        }
        self.produces |= {
            electron_reco_id_iso_weights, muon_id_iso_weights,
            # btag_weights,
            normalization_weights,
            pu_weight,
            mc_weight,
            top_pt_weight,
        }
        if self.config_inst.x.run == 2:
            self.uses |= {
                # l1_prefiring_weights,
                vjets_weight,
                toptag_weights,
            }
            self.produces |= {
                # l1_prefiring_weights,
                vjets_weight,
                toptag_weights,
            }
        if self.config_inst.x.year in [2022, 2023]:
            self.uses |= {
                btag_weights,
            }
            self.produces |= {
                btag_weights,
            }
        elif self.config_inst.x.year == 2024:
            self.uses |= {
                btag_wp_weights,
            }
            self.produces |= {
                btag_wp_weights,
            }
