# coding: utf-8

"""
Custom jet energy calibration methods that disable data uncertainties (for searches).
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.jets import jec, jer
from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")


# custom jec calibrator that only runs nominal correction
jec_nominal = jec.derive("jec_nominal", cls_dict={"uncertainty_sources": []})


@calibrator
def jet_energy(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Common calibrator for Jet energy corrections, applying nominal JEC for data, and JEC with
    uncertainties plus JER for MC. Information about used and produced columns and dependent
    calibrators is added in a custom init function below.
    """
    if self.dataset_inst.is_mc:
        # TODO: for testing purposes, only run jec_nominal for now
        events = self[jec_nominal](events, **kwargs)
        events = self[jer](events, **kwargs)
    else:
        events = self[jec_nominal](events, **kwargs)

    return events


@jet_energy.init
def jet_energy_init(self: Calibrator) -> None:
    # add standard jec and jer for mc, and only jec nominal for dta
    if getattr(self, "dataset_inst", None) and self.dataset_inst.is_mc:
        # TODO: for testing purposes, only run jec_nominal for now
        self.uses |= {jec_nominal, jer}
        self.produces |= {jec_nominal, jer}
    else:
        self.uses |= {jec_nominal}
        self.produces |= {jec_nominal}


@calibrator(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass", "nElectron",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "nMuon",
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.rawFactor", "nJet",
        # index of electrons/muons matched to jets
        "Jet.muonIdx1", "Jet.muonIdx2", "Jet.electronIdx1", "Jet.electronIdx2",
        # PF energy fractions
        "Jet.chEmEF", "Jet.muEF",
        attach_coffea_behavior,
    },
    produces={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.rawFactor",
        "Jet.chEmEF", "Jet.muEF",
    },
)
def jet_lepton_cleaner(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Calibrator to clean jet four-vectors from contributions from nearby leptons
    """

    # load coffea behaviors for simplified arithmetic with vectors
    events["Electron"] = ak.with_name(events.Electron, "PtEtaPhiMLorentzVector")
    events["Muon"] = ak.with_name(events.Muon, "PtEtaPhiMLorentzVector")
    events["Jet"] = ak.with_name(events.Jet, "PtEtaPhiMLorentzVector")

    # revert JEC for jet pt and jet mass,
    # set correction factor to 0
    events = set_ak_column(events, "Jet.pt", events.Jet.pt * (1 - events.Jet.rawFactor))
    events = set_ak_column(events, "Jet.mass", events.Jet.mass * (1 - events.Jet.rawFactor))
    events = set_ak_column(events, "Jet.rawFactor", 0)

    # create arrays with indices of leptons that are matched to jet,
    # non if no matched lepton
    idx_e1 = ak.mask(events.Jet.electronIdx1, events.Jet.electronIdx1 >= 0)
    idx_e2 = ak.mask(events.Jet.electronIdx2, events.Jet.electronIdx2 >= 0)
    idx_m1 = ak.mask(events.Jet.muonIdx1, events.Jet.muonIdx1 >= 0)
    idx_m2 = ak.mask(events.Jet.muonIdx2, events.Jet.muonIdx2 >= 0)

    # energy sum of PF leptons clustered into jet
    jet_muon_energy = events.Jet.energy * events.Jet.muEF
    jet_charged_em_energy = events.Jet.energy * events.Jet.chEmEF

    # list with matched leptons
    jet_leptons = [
        events.Electron[idx_e1],
        events.Electron[idx_e2],
        events.Muon[idx_m1],
        events.Muon[idx_m2],
    ]

    # only do cleaning if lepton energy is compatible with PF energy
    tolerance = 0.1
    jet_lepton_do_cleaning = [
        jet_leptons[0].energy > (1 - tolerance) * jet_charged_em_energy,
        jet_leptons[1].energy > (1 - tolerance) * jet_charged_em_energy,
        jet_leptons[2].energy > (1 - tolerance) * jet_muon_energy,
        jet_leptons[3].energy > (1 - tolerance) * jet_muon_energy,
    ]
    jet_leptons = [
        ak.mask(jet_lepton, do_cleaning)
        for jet_lepton, do_cleaning
        in zip(jet_leptons, jet_lepton_do_cleaning)
    ]

    # convert to four-vectors for leptons in jets
    jet_lepton_p4s = [
        ak.with_name(
            ak.zip(
                {var: getattr(jet_lepton, var) for var in ["pt", "eta", "phi", "mass"]}
            ),
            "PtEtaPhiMLorentzVector"
        )
        for jet_lepton in jet_leptons
    ]

    # sum lepton contributions
    jet_lepton_sum = ak.concatenate(
        [ak.singletons(jet_lepton_p4, axis=1) for jet_lepton_p4 in jet_lepton_p4s],
        axis=2,
    ).sum(axis=2)

    # subtract lepton contributions from jets
    jet_cleaned = ak.with_name(events.Jet - jet_lepton_sum, "LorentzVector")

    # keep only cases where cleaning results in positive mass
    mass_is_positive = (jet_cleaned.energy > jet_cleaned.pvec.rho)
    jet_cleaned = ak.mask(jet_cleaned, mass_is_positive)

    # also filter out cases where cleaning results in a large change of direction
    delta_r = events.Jet.delta_r(jet_cleaned)
    jet_cleaned = ak.mask(jet_cleaned, delta_r < np.pi/2)

    # save updated jet variables
    for var in ["pt", "eta", "phi", "mass"]:
        value = ak.fill_none(getattr(jet_cleaned, var), 0.0)
        events = set_ak_column(events, f"Jet.{var}", value)

    return events
