# coding: utf-8

"""
Column production methods related to ttbar mass reconstruction.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, EMPTY_FLOAT
from columnflow.production.util import attach_coffea_behavior

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

@producer(
    uses={
        "channel_id", "category_ids",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
    },
    produces={
        "Lepton.*"
    }
)
def choose_lepton(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """Chooses either muon or electron as the main choose_lepton per event
    based on `channel_id` information."""

    # extract only LV columns
    muon = events.Muon[["pt", "eta", "phi", "mass"]]
    electron = events.Electron[["pt", "eta", "phi", "mass"]]

    # choose either muons or electrons based on channel ID
    lepton = ak.concatenate([
        ak.mask(muon, events.channel_id == 2),
        ak.mask(electron, events.channel_id == 1),
    ], axis=1)

    # if more than one lepton, choose the first
    lepton = ak.firsts(lepton, axis=1)

    # attach lorentz vector behavior to lepton
    lepton = ak.with_name(lepton, "PtEtaPhiMLorentzVector")

    # commit lepton to events array
    events = set_ak_column(events, "Lepton", lepton)

    return events


@producer(
    uses={
        choose_lepton,
        "MET.pt", "MET.phi",
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
    events["MET"] = ak.with_name(events.MET, "MissingET")

    # choose lepton
    events = self[choose_lepton](events, **kwargs)
    lepton = events["Lepton"]

    # attach lorentz vector behavior to lepton
    lepton = ak.with_name(lepton, "PtEtaPhiMLorentzVector")

    lnu_delta_phi = lepton.delta_phi(events.MET)
    lnu_delta_phi = lepton.phi - events.MET.phi

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
        ak.singletons(lnu_pz_0)
    )

    # replace null values (no lepton available) with empty lists
    lnu_pz_cands = ak.fill_none(lnu_pz_pm, [], axis=0)

    nu_cands = ak.zip({
        "x": events.MET.x,
        "y": events.MET.y,
        "z": lnu_pz_cands,
    })

    # attach three-vector behavior to neutrino candidates
    nu_cands = ak.with_name(nu_cands, "ThreeVector")

    # sanity checks: pt and phi of all neutrino candidates
    # should be equal to those of MET
    assert (
        ak.all((abs(nu_cands.delta_phi(events.MET)) < 1e-3)),
        "Sanity check failed: neutrino candidates and MET 'phi' differ"
    )
    assert (
        ak.all((abs(nu_cands.pt - events.MET.pt) < 1e-3)),
        "Sanity check failed: neutrino candidates and MET 'pt' differ"
    )

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


@producer(
    uses={
        choose_lepton, neutrino_candidates,
        "channel_id", "category_ids",
        "pt_regime",
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        "FatJetTopTag.pt", "FatJetTopTag.eta", "FatJetTopTag.phi", "FatJetTopTag.mass",
    },
    produces={
        choose_lepton, neutrino_candidates,
        "TTbar.*"
    },
)
def ttbar(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reconstruct the ttbar pair in the semileptonic decay mode.
    This is done by evaluating all possibilities of assigning the lepton,
    jets, and neutrino to the hadronic and leptonic legs of the decay,
    in terms of a chi2 metric. The configuration with the lowest
    chi2 out of all possibilities is selected.
    """
    # load coffea behaviors for simplified arithmetic with vectors
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    events["Jet"] = ak.with_name(events.Jet, "PtEtaPhiMLorentzVector")
    events["FatJetTopTag"] = ak.with_name(events.FatJetTopTag, "PtEtaPhiMLorentzVector")

    # reconstruct neutrino candidates
    events = self[neutrino_candidates](events, **kwargs)
    nu_cands_lv = events["NeutrinoCandidates"]

    # get leptons
    events = self[choose_lepton](events, **kwargs)
    lepton = events["Lepton"]

    # -- only keep <= 10 jets per event
    jet = events.Jet[ak.local_index(events.Jet) < 10]

    # -- filter fat jets
    fatjet = events.FatJetTopTag

    # well separated from all AK4 jets (deltaR >= 1.2)
    delta_r_fatjet_jet = fatjet.metric_table(jet)
    fatjet = fatjet[ak.all(delta_r_fatjet_jet > 1.2, axis=-1)]

    # well separated from lepton
    delta_r_fatjet_lepton = ak.firsts(fatjet.metric_table(lepton))
    fatjet = fatjet[delta_r_fatjet_lepton > 0.8]

    # -- combinatorics

    # all possible combinations of four jets
    # (the first is assigned to the leptonic decaying top quark,
    # and the remaining ones to the hadronic decaying one)
    jet_comb = ak.combinations(
        jet, 4,
        fields=('lep', 'had_1', 'had_2', 'had_3')
    )

    # possible assignments to leptonic and hadronic decay
    hyp_jet_lep, hyp_jet_had_1, hyp_jet_had_2, hyp_jet_had_3 = ak.unzip(jet_comb)

    # hadronic top hypotheses from LV sum over three jets
    hyp_top_had = ak.concatenate(
        list(map(ak.singletons, [hyp_jet_had_1, hyp_jet_had_2, hyp_jet_had_3])),
        axis=1,
    ).sum(axis=1)

    # expand hypothesis space to cover leptonic decay
    hyp_lep, hyp_nu, hyp_jet_lep = ak.unzip(
        ak.cartesian([ak.singletons(lepton), nu_cands_lv, hyp_jet_lep])
    )

    # sum over leptonic decay products
    hyp_top_lep = ak.concatenate(
        list(map(ak.singletons, [hyp_lep, hyp_nu, hyp_jet_lep])),
        axis=1,
    ).sum(axis=1)

    # calculate hypothesis chi2 scores
    chi2_pars = self.config_inst.x.chi2_parameters.resolved  # TODO: boosted
    hyp_top_had_chi2 = ((hyp_top_had.mass - chi2_pars.m_had) / chi2_pars.s_had) ** 2
    hyp_top_lep_chi2 = ((hyp_top_lep.mass - chi2_pars.m_lep) / chi2_pars.s_lep) ** 2

    # reduce hypytheses based on minimal chi2 score
    hyp_top_had_chi2_argmin = ak.argmin(hyp_top_had_chi2, axis=1, keepdims=True)
    hyp_top_lep_chi2_argmin = ak.argmin(hyp_top_lep_chi2, axis=1, keepdims=True)
    top_had_chi2 = ak.firsts(hyp_top_had_chi2[hyp_top_had_chi2_argmin])
    top_lep_chi2 = ak.firsts(hyp_top_lep_chi2[hyp_top_lep_chi2_argmin])
    top_had = ak.firsts(hyp_top_had[hyp_top_had_chi2_argmin])
    top_lep = ak.firsts(hyp_top_lep[hyp_top_lep_chi2_argmin])

    # sum over top quarks to form ttbar system
    ttbar = ak.concatenate(
        list(map(ak.singletons, [top_had, top_lep])),
        axis=1,
    ).sum(axis=1)

    # final chi2 value
    chi2 = top_had_chi2 + top_lep_chi2

    # -- calculate cos(theta*)

    # boost lepton + leptonic top quark to ttbar rest frame
    top_lep_ttrest = top_lep.boost(-ttbar.boostvec)

    # get cosine from three-vector dot product and magnitudes
    cos_theta_star = ttbar.dot(top_lep_ttrest) / (ttbar.pvec.p * top_lep_ttrest.pvec.p)

    # write out columns
    for var in ('pt', 'eta', 'phi', 'mass'):
        events = set_ak_column(events, f"TTbar.top_had_{var}", ak.fill_none(getattr(top_had, var), EMPTY_FLOAT))
        events = set_ak_column(events, f"TTbar.top_lep_{var}", ak.fill_none(getattr(top_lep, var), EMPTY_FLOAT))
        events = set_ak_column(events, f"TTbar.{var}", ak.fill_none(getattr(ttbar, var), EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.chi2_had", ak.fill_none(top_had_chi2, EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.chi2_lep", ak.fill_none(top_lep_chi2, EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.chi2", ak.fill_none(top_lep_chi2, EMPTY_FLOAT))
    events = set_ak_column(events, "TTbar.cos_theta_star", ak.fill_none(cos_theta_star, EMPTY_FLOAT))

    return events
