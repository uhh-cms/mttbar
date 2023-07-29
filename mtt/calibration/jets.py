# coding: utf-8

"""
Custom jet energy calibration methods that disable data uncertainties (for searches).
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.jets import jec, jer
from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import set_ak_column

from mtt.production.util import lv_xyzt, lv_mass


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
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.rawFactor",
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

    # build jet lorentz vectors
    jet_lv = lv_xyzt(events.Jet)

    # create arrays with indices of leptons that are matched to jet,
    # non if no matched lepton
    idx_e1 = ak.mask(events.Jet.electronIdx1, events.Jet.electronIdx1 >= 0)
    idx_e2 = ak.mask(events.Jet.electronIdx2, events.Jet.electronIdx2 >= 0)
    idx_m1 = ak.mask(events.Jet.muonIdx1, events.Jet.muonIdx1 >= 0)
    idx_m2 = ak.mask(events.Jet.muonIdx2, events.Jet.muonIdx2 >= 0)

    # list with matched leptons
    jet_leptons_types = [
        (events.Electron[idx_e1], "e"),
        (events.Electron[idx_e2], "e"),
        (events.Muon[idx_m1], "mu"),
        (events.Muon[idx_m2], "mu"),
    ]

    # total energy from clustered leptonic PF candidates
    jet_pf_energies = {
        "mu": jet_lv.energy * events.Jet.muEF,
        "e": jet_lv.energy * events.Jet.chEmEF,
    }

    # subtract lepton contributions from jets
    tolerance = 0.1
    for jet_lepton, jet_lepton_type in jet_leptons_types:
        jet_lepton_lv = lv_xyzt(jet_lepton)
        jet_lv_cleaned = lv_xyzt(jet_lv - jet_lepton_lv)

        jet_pf_energy = jet_pf_energies[jet_lepton_type]
        jet_pf_energy_cleaned = jet_pf_energy - jet_lepton_lv.energy

        # only perform the cleaning of the current lepton
        # if the following conditions are met

        # lepton energy compatible with PF energy fraction (within tolerance)
        lep_energy_pf_compatible = (jet_lepton_lv.energy < (1 + tolerance) * jet_pf_energy)

        # calculate square of cleaned jet mass
        jet_lv_cleaned_mass_sq = jet_lv_cleaned.energy**2 - jet_lv_cleaned.rho**2
        # mask values that would lead to imaginary masses,
        # but substitute absolute value if the mass square is only negative within
        # tolerance (high probablility that this was a lepton fake)
        jet_lv_cleaned_mass = ak.mask(
            np.sqrt(abs(jet_lv_cleaned_mass_sq)),
            jet_lv_cleaned_mass_sq >= -tolerance
        )

        # cleaning does not result in a negative/imaginary/undefined mass
        mass_stays_positive = ~ak.is_none(jet_lv_cleaned_mass, axis=1)

        # angle before/after cleaning is similar (delta_r < max_angle_diff)
        #
        # note: we use a pt-dependent heuristic for `max_angle_diff` to increase the
        # allowed angle difference for low-pt jets. The reason is to catch the cases
        # where the jet is a pure lepton fake (i.e. consists of only one lepton). Since
        # the resulting momentum after the cleaning is close to zero in this case, the
        # angle may change sign due to resolution effects. We want to clean these jets as
        # well, so if the jet pt after cleaning is below a chosen threshold `ref_pt`, we
        # calculate the maximum angle difference as a function ~1/pt, reaching the maximum
        # value `np.pi` when extrapolating towards pt = 0.

        #
        # method 1: fixed threshold
        #

        # clean only if angle difference below fixed threshold OR the cleaned pt is very low
        # (high probablility that this was a lepton fake)
        angle_change_small = (
            (jet_lv.delta_r(jet_lv_cleaned) <= np.pi / 2) |
            (jet_lv_cleaned.pt < 10)
        )

        #
        # method 2: heuristic (needs tweaking)
        #

        # # tweakable parameters
        # ref_pt = 30.  # pt below which heuristic is active
        # ref_angle = np.pi / 2  # constant max angle difference at (pt > = ref_pt)

        # # calculate maximum allowed angle difference using heuristic below `ref_pt`
        # pt_scale = (ref_angle * ref_pt) / (np.pi - ref_angle)
        # max_angle_diff = np.pi * pt_scale / (jet_lv_cleaned.pt + pt_scale)
        # max_angle_diff = ak.where(jet_lv_cleaned.pt > ref_pt, ref_angle, max_angle_diff)

        # # clean only if angle difference passes the check
        # angle_change_small = jet_lv.delta_phi(jet_lv_cleaned) <= max_angle_diff

        # AND of cleaning conditions
        do_clean = mass_stays_positive & angle_change_small & lep_energy_pf_compatible
        # `None` if no matched lepton -> no cleaning
        do_clean = ak.fill_none(do_clean, False)

        # update jet LV
        jet_lv = ak.where(
            do_clean,
            jet_lv_cleaned,
            jet_lv,
        )

        # update jet PF energies
        jet_pf_energies[jet_lepton_type] = ak.where(
            do_clean,
            jet_pf_energy_cleaned,
            jet_pf_energy,
        )

    # save updated jet variables
    jet_lv = lv_mass(jet_lv)
    for var in ["pt", "eta", "phi", "mass"]:
        # ensure no missing values
        value = ak.fill_none(ak.nan_to_none(getattr(jet_lv, var)), 0.0)
        events = set_ak_column(events, f"Jet.{var}", value)

    return events
