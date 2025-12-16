# coding: utf-8

"""
Column producers related to neutrino reconstruction.
"""
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from mtt.production.lepton import choose_lepton

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        choose_lepton,
        "PuppiMET.pt", "PuppiMET.phi",
    },
    produces={
        choose_lepton,
        "NeutrinoCandidates.*",
    },
)
def neutrino_candidates(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reconstruct possible candidates for the neutrino, assuming the azimuthal
    and radial components are equal to those of the missing transverse momentum.
    """

    # load coffea behaviors for simplified arithmetic with vectors
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Electron"] = ak.with_name(events.Electron, "PtEtaPhiMLorentzVector")
    events["Muon"] = ak.with_name(events.Muon, "PtEtaPhiMLorentzVector")
    events["MET"] = ak.with_name(events.PuppiMET, "MissingET")

    # choose lepton
    events = self[choose_lepton](events, **kwargs)
    lepton = events["Lepton"]

    # attach lorentz vector behavior to lepton
    lepton = ak.with_name(lepton, "PtEtaPhiMLorentzVector")

    lnu_delta_phi = lepton.delta_phi(events.MET)

    # TODO: move to config
    m_w = 80.0  # GeV

    # -- calculate longitudinal component of neutrino

    # helper mass
    lnu_mu = (
        0.5 * m_w ** 2 +
        events.MET.pt * lepton.pt * np.cos(lnu_delta_phi)
    )

    # real part of solution
    lnu_pz_0 = lnu_mu * lepton.z / (lepton.pt ** 2)

    # calculate discriminant
    lnu_delta_e_sq = ((lepton.energy * events.MET.pt) ** 2 - lnu_mu ** 2) / (lepton.pt ** 2)
    lnu_disc = lnu_pz_0 ** 2 - lnu_delta_e_sq

    # quadratic solutions
    # (truncate discriminant at 0 to discard imaginary part)
    lnu_disc_nonneg = ak.where(lnu_disc > 0, lnu_disc, 0)
    lnu_pz_p = lnu_pz_0 + np.sqrt(lnu_disc_nonneg)
    lnu_pz_m = lnu_pz_0 - np.sqrt(lnu_disc_nonneg)

    # pack solutions into a nested list
    lnu_pz_pm = ak.concatenate([
        ak.singletons(lnu_pz_p),
        ak.singletons(lnu_pz_m),
    ], axis=1)

    # choose either 2 real solutions as candidates,
    # or real part of complex solutions
    lnu_pz_cands = ak.where(
        lnu_disc > 0,
        lnu_pz_pm,
        ak.singletons(lnu_pz_0),
    )

    # replace null values (no lepton available) with empty lists
    lnu_pz_cands = ak.fill_none(lnu_pz_cands, [], axis=0)

    nu_cands = ak.zip({
        "x": events.MET.x,
        "y": events.MET.y,
        "z": lnu_pz_cands,
    })

    # attach three-vector behavior to neutrino candidates
    nu_cands = ak.with_name(nu_cands, "ThreeVector")

    # sanity checks: pt and phi of all neutrino candidates
    # should be equal to those of MET
    # tolerance of 2e-3 due to one tt_sl event with a 0.00195 GeV difference in pt
    tol = 2e-3
    assert ak.all((abs(nu_cands.delta_phi(events.MET)) < tol)), \
        "Sanity check failed: neutrino candidates and MET 'phi' differ"
    if ak.any((abs(nu_cands.pt - events.MET.pt) >= tol)):
        print("Debug info for nu cand pt != MET pt:")
        print("nu_cands.pt:", nu_cands.pt)
        print("events.MET.pt:", events.MET.pt)
    assert ak.all((abs(nu_cands.pt - events.MET.pt) < tol)), \
        "Sanity check failed: neutrino candidates and MET 'pt' differ"

    # build neutrino candidate four-vectors
    nu_cands_lv = ak.zip({
        "pt": nu_cands.pt,
        # no shortcut for pseudorapitiy
        "eta": -np.log(np.tan(nu_cands.theta / 2)),
        "phi": nu_cands.phi,
        "mass": 0,
    }, with_name="PtEtaPhiMLorentzVector")

    # commit neutrino candidates to events array
    events = set_ak_column(events, "NeutrinoCandidates", nu_cands_lv)

    return events
