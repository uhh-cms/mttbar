# coding: utf-8

"""
Producers for ML inputs
"""
import law
import functools
import itertools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from mtt.config.variables import add_variables_ml
from mtt.config.categories import add_categories_production
from mtt.production.weights import weights
from mtt.production.lepton import choose_lepton
# from mtt.production.ttbar_reco import ttbar

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

# use float32 type for ML input columns
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
logger = law.logger.get_logger(__name__)


@producer(
    uses={
        weights,
        choose_lepton,
        # AK4 jets
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        # AK8 jets
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass",
        "FatJet.msoftdrop",
        "FatJet.tau1", "FatJet.tau2", "FatJet.tau3",
        "BJet.pt",
        "channel_id", "category_ids",
    },
    produces={
        weights,
        # columns for ML inputs are set by the init function
    },
)
def ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # set several columns different for eras
    met_col = self.config_inst.x.met_selection.column
    btag_col = self.config_inst.x.jet_selection.ak4.btagger.column

    # attach coffea behavior
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)

    # name of table to place ML variables in
    ns = self.ml_namespace

    # run dependencies
    events = self[choose_lepton](events, **kwargs)
    # events = self[ttbar](events, **kwargs)

    # object arrays
    jet = ak.with_name(events.Jet, "Jet")
    fatjet = ak.with_name(events.FatJet, "FatJet")
    lepton = ak.with_name(events.Lepton, "PtEtaPhiMLorentzVector")
    met = events[met_col]

    # extract the lepton channel (0 = electron, 1 = muon) and boosted flag (0 = resolved, 1 = boosted)
    # from the last (most specific) entry of each event's category_ids list; within that id, the
    # last digit encodes the channel (1 = electron, 2 = muon) and the second-to-last digit encodes
    # boosted-ness (1 = resolved, 2 = boosted)
    leaf_category_id = ak.firsts(events.category_ids[:, ::-1])

    channel_digit = leaf_category_id % 10          # 1 = electron, 2 = muon
    boosted_digit = (leaf_category_id // 10) % 10  # 1 = resolved, 2 = boosted

    lepton_channel = ak.where(channel_digit == 2, 1, 0)
    is_boosted = ak.where(boosted_digit == 2, 1, 0)

    lepton_channel = ak.fill_none(lepton_channel, -1)
    is_boosted = ak.fill_none(is_boosted, -1)

    events = set_ak_column(events, f"{ns}.lepton_channel", lepton_channel)
    events = set_ak_column(events, f"{ns}.is_boosted", is_boosted)

    # btag score for AK4 jets
    # get btagging working points for the given column from config
    wp_dict = self.config_inst.x.btag_wp[btag_col].fixed_wp
    edges = sorted(wp_dict.values())

    scores = jet[btag_col]

    # count how many WPs are passed
    buckets = sum(scores >= edge for edge in edges)

    jet = ak.with_field(jet, buckets, f"{btag_col}_buckets")

    # store btag pass/fail decision as boolean for each WP as well
    # 1: pass, 0: fail, -1: undefined (e.g. no jet or no score)
    for wp, edge in wp_dict.items():
        pass_fail = ak.where(scores >= edge, 1, 0)
        pass_fail = ak.where(ak.is_none(scores), -1, pass_fail)
        jet = ak.with_field(jet, pass_fail, f"{btag_col}_pass_{wp}")

    # n-subjettiness discriminants for AK8 jets with safe division
    # avoid fatjet.tau2 to prevent conflict with vector behaviour
    fatjet["tau32"] = ak.where(
        (fatjet["tau2"] != 0) & (fatjet.tau3 != 0),
        fatjet.tau3 / fatjet["tau2"],
        -1,
    )

    fatjet["tau21"] = ak.where(
        (fatjet.tau1 != 0) & (fatjet.tau2 != 0),
        fatjet["tau2"] / fatjet.tau1,
        -1,
    )

    # jet/fatjet multiplicities
    events = set_ak_column(events, f"{ns}.n_jet", ak.num(events.Jet, axis=1))
    events = set_ak_column(events, f"{ns}.n_fatjet", ak.num(events.FatJet, axis=1))
    events = set_ak_column(events, f"{ns}.n_bjet", ak.num(events.BJet, axis=1))

    # event-level variables
    ht_jets = ak.sum(jet.pt, axis=-1)
    ht_fatjets = ak.sum(fatjet.pt, axis=-1)
    ht_bjets = ak.sum(events.BJet.pt, axis=-1)

    had_sum = ak.sum(jet.pt, axis=-1) + ak.sum(fatjet.pt, axis=-1)  # don't add bjets to avoid double counting

    st = ak.sum(jet.pt, axis=-1) + lepton.pt + met.pt

    # build deltaR between lepton and jets/fatjets for later use in ML input selection
    lepton_jet_deltar = ak.firsts(jet.metric_table(lepton), axis=-1)
    jet = ak.with_field(jet, lepton_jet_deltar, "lepton_deltar")
    lepton_fatjet_deltar = ak.firsts(fatjet.metric_table(lepton), axis=-1)
    fatjet = ak.with_field(fatjet, lepton_fatjet_deltar, "lepton_deltar")

    # build deltaR between MET and jets/fatjets for later use in ML input selection
    # for MET, we can treat it as a particle at phi = met.phi and eta = 0
    met_vector = ak.zip(
        {"pt": met.pt, "phi": met.phi, "eta": ak.zeros_like(met.pt)}, with_name="PtEtaPhiMLorentzVector",
    )
    met_jet_deltar = ak.firsts(jet.metric_table(met_vector), axis=-1)
    jet = ak.with_field(jet, met_jet_deltar, "met_deltar")
    met_fatjet_deltar = ak.firsts(fatjet.metric_table(met_vector), axis=-1)
    fatjet = ak.with_field(fatjet, met_fatjet_deltar, "met_deltar")

    # build deltaR between two leading jets/fatjets for later use in ML input selection
    # Safely pad arrays to ensure we have at least 2 elements
    jet_safe = ak.pad_none(jet, 2)        # Pad to at least 2 jets
    fatjet_safe = ak.pad_none(fatjet, 2)  # Pad to at least 2 fatjets

    # Now we can safely calculate deltaR (will be None for padded elements)
    leading_jets_deltar = ak.where(
        ak.num(jet) >= 2,  # Only use real result if we have 2+ jets
        jet_safe[:, 0].deltaR(jet_safe[:, 1]),
        -1.0,
    )
    leading_fatjets_deltar = ak.where(
        ak.num(fatjet) >= 2,
        fatjet_safe[:, 0].deltaR(fatjet_safe[:, 1]),
        -1.0,
    )
    leading_jetfatjet_deltar = ak.where(
        (ak.num(jet) >= 1) & (ak.num(fatjet) >= 1),
        ak.pad_none(jet, 1)[:, 0].deltaR(ak.pad_none(fatjet, 1)[:, 0]),
        -1.0,
    )

    # # add reco vars from ttbar reconstruction
    # TODO find solution without needing to rerun the ttbar producer here

    # -- helper functions

    def set_vars(events, name, arr, n_max, attrs, default=-10.0):
        # pad to miminal length
        arr = ak.pad_none(arr, n_max)
        # extract fields
        for i, attr in itertools.product(range(1, n_max + 1), attrs):
            value = ak.nan_to_none(getattr(arr[:, i - 1], attr))
            value = ak.fill_none(value, default)
            events = set_ak_column_f32(events, f"{self.ml_namespace}.{name}_{i}_{attr}", value)
        logger.debug(f"Set {n_max} {name} variables: {', '.join(attrs)}")
        return events

    def set_vars_single(events, name, arr, attrs, default=-10.0):
        for attr in attrs:
            value = ak.nan_to_none(getattr(arr, attr))
            value = ak.fill_none(value, default)
            events = set_ak_column_f32(events, f"{self.ml_namespace}.{name}_{attr}", value)
        return events

    def set_event_vars(events, name, value, default=-10.0):
        value = ak.nan_to_none(value)
        value = ak.fill_none(value, default)
        events = set_ak_column_f32(events, f"{self.ml_namespace}.{name}", value)
        return events

    # AK4 jets
    events = set_vars(
        events, "jet", jet, n_max=5,
        attrs=("energy", "pt", "eta", "phi", "mass", btag_col, f"{btag_col}_buckets", "lepton_deltar", "met_deltar"),
    )
    events = set_vars(
        events, "jet", jet, n_max=5,
        attrs=(
            f"{btag_col}_pass_loose",
            f"{btag_col}_pass_medium",
            f"{btag_col}_pass_tight",
            f"{btag_col}_pass_xtight",
            f"{btag_col}_pass_xxtight"
        ),
    )

    # AK8 jets
    events = set_vars(
        events, "fatjet", fatjet, n_max=3,
        attrs=(
            "energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32",
            "tau1", "tau2", "tau3", "lepton_deltar", "met_deltar",
        ),
    )

    # Lepton
    events = set_vars_single(
        events, "lepton", lepton,
        attrs=("energy", "pt", "eta", "phi"),
    )

    # MET
    events = set_vars_single(
        events, "met", met,
        attrs=("pt", "phi"),
    )

    # event-level variables
    events = set_event_vars(events, "ht_jets", ht_jets)
    events = set_event_vars(events, "ht_fatjets", ht_fatjets)
    events = set_event_vars(events, "ht_bjets", ht_bjets)
    events = set_event_vars(events, "ht_sum", had_sum)
    events = set_event_vars(events, "st", st)
    events = set_event_vars(events, "leading_jets_deltar", leading_jets_deltar)
    events = set_event_vars(events, "leading_fatjets_deltar", leading_fatjets_deltar)
    events = set_event_vars(events, "leading_jetfatjet_deltar", leading_jetfatjet_deltar)

    # weights
    events = self[weights](events, **kwargs)

    return events


@ml_inputs.init
def ml_inputs_init(self: Producer) -> None:
    # put ML input columns in separate namespace/table
    self.ml_namespace = "MLInput"
    btag_col = self.config_inst.x.jet_selection.ak4.btagger.column
    met_col = self.config_inst.x.met_selection.column

    # store column names
    self.ml_columns = {
        "n_jet",
        "n_fatjet",
        "n_bjet",
        "lepton_channel",
        "is_boosted",
    } | {
        f"jet_{i + 1}_{var}"
        for var in (
            "energy", "pt", "eta", "phi", "mass", btag_col, f"{btag_col}_buckets", "lepton_deltar", "met_deltar",
        )
        for i in range(5)
    } | {
        f"jet_{i + 1}_{btag_col}_pass_{wp}"
        for wp in self.config_inst.x.btag_wp[btag_col].fixed_wp.keys()
        for i in range(5)
    } | {
        f"fatjet_{i + 1}_{var}"
        for var in (
            "energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32",
            "tau1", "tau2", "tau3", "lepton_deltar", "met_deltar",
        )
        for i in range(3)
    } | {
        f"lepton_{var}"
        for var in ("energy", "pt", "eta", "phi")
    } | {
        f"met_{var}"
        for var in ("pt", "phi")
    } | {
        "ht_jets", "ht_fatjets", "ht_bjets", "ht_sum", "st",
    } | {
        "leading_jets_deltar", "leading_fatjets_deltar", "leading_jetfatjet_deltar",
    }

    # declare produced columns
    self.produces |= {
        f"{self.ml_namespace}.{col}"
        for col in self.ml_columns
    }

    # add production categories to config
    if not self.config_inst.get_aux("has_categories_production", False):
        add_categories_production(self.config_inst)
        self.config_inst.x.has_categories_production = True

    # add ml variables to config
    if not self.config_inst.get_aux("has_variables_ml", False):
        add_variables_ml(self.config_inst)
        self.config_inst.x.has_variables_ml = True

    self.uses |= {
        f"Jet.{btag_col}",
        f"{met_col}.pt", f"{met_col}.phi",
    }
