# coding: utf-8

"""
Calibration methods.
"""
import functools

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.jets import jets
# TODO should we add these later?
# from columnflow.calibration.cms.egamma import electron_scale_smear
# from columnflow.calibration.cms.muon import muon_sr
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.jet import msoftdrop
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from mtt.calibration.jets import jet_energy, jet_lepton_cleaner, jec_subjets, jer_subjets

ak = maybe_import("awkward")
np = maybe_import("numpy")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@calibrator(
    uses={mc_weight, deterministic_seeds, jets},
    produces={mc_weight, deterministic_seeds, jets},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)
    events = self[jets](events, **kwargs)

    return events


@calibrator(
    uses={
        mc_weight,
        deterministic_seeds,
        jet_lepton_cleaner,
        jet_energy,
        msoftdrop,
        jec_subjets,
        # jer_subjets
    },
    produces={
        mc_weight,
        deterministic_seeds,
        jet_lepton_cleaner,
        jet_energy,
        msoftdrop,
        jec_subjets,
        # jer_subjets
    },
)
def skip_jecunc(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """ only uses jec_nominal for test purposes """
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)
    events = self[jet_lepton_cleaner](events, **kwargs)
    events = self[jet_energy](events, **kwargs)

    # fake subjet area column by setting it to an array with the same structure as the subjet pt column containing 0.5
    # (needed to be able to use same code as for top-level AK4/AK8 jets, as the producer formally requires an `area`
    # column, despite not actually using it)
    events = set_ak_column_f32(events, "SubJet.area", 0.5 * ak.ones_like(events.SubJet.pt))

    events = self[jec_subjets](events, **kwargs)
    # if self.dataset_inst.is_mc:
    #     events = self[jer_subjets](events, **kwargs)
    events = self[msoftdrop](events, **kwargs)

    return events


@calibrator(
    uses={mc_weight, deterministic_seeds, jet_energy},
    produces={mc_weight, deterministic_seeds, jet_energy},
)
def skip_jecunc_wo_cleaner(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """ only uses jec_nominal for test purposes """
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)
    events = self[jet_energy](events, **kwargs)

    return events
