# coding: utf-8

"""
Producers for ML inputs
"""
import functools
import itertools

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from mtt.config.variables import add_variables_ml
from mtt.config.categories import add_categories_production
from mtt.production.weights import weights
from mtt.production.lepton import choose_lepton

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

# use float32 type for ML input columns
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@producer(
    uses={
        weights,
        choose_lepton,
        # AK4 jets
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Jet.btagDeepFlavB",
        # AK8 jets
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass",
        "FatJet.msoftdrop",
        "FatJet.tau1", "FatJet.tau2", "FatJet.tau3",
        # MET
        "MET.pt", "MET.phi",
    },
    produces={
        weights,
        # columns for ML inputs are set by the init function
    },
)
def ml_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # attach coffea behavior
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)

    # name of table to place ML variables in
    ns = self.ml_namespace

    # run dependencies
    events = self[choose_lepton](events, **kwargs)

    # object arrays
    jet = ak.with_name(events.Jet, "Jet")
    fatjet = ak.with_name(events.FatJet, "FatJet")
    lepton = ak.with_name(events.Lepton, "PtEtaPhiMLorentzVector")
    met = events.MET

    # btag score for AK4 jets
    jet["btag"] = jet.btagDeepFlavB

    # n-subjettiness discriminants for AK8 jets
    fatjet["tau32"] = fatjet.tau3 / fatjet.tau2
    fatjet["tau21"] = fatjet.tau2 / fatjet.tau1

    # jet/fatjet multiplicities
    events = set_ak_column(events, f"{ns}.n_jet", ak.num(events.Jet, axis=1))
    events = set_ak_column(events, f"{ns}.n_fatjet", ak.num(events.FatJet, axis=1))

    # -- helper functions

    def set_vars(events, name, arr, n_max, attrs, default=-10.0):
        # pad to miminal length
        arr = ak.pad_none(arr, n_max)
        # extract fields
        for i, attr in itertools.product(range(1, n_max + 1), attrs):
            value = ak.nan_to_none(getattr(arr[:, i - 1], attr))
            value = ak.fill_none(value, default)
            events = set_ak_column_f32(events, f"{self.ml_namespace}.{name}_{attr}_{i}", value)
        return events

    def set_vars_single(events, name, arr, attrs, default=-10.0):
        for attr in attrs:
            value = ak.nan_to_none(getattr(arr, attr))
            value = ak.fill_none(value, default)
            events = set_ak_column_f32(events, f"{self.ml_namespace}.{name}_{attr}", value)
        return events

    # AK4 jets
    events = set_vars(
        events, "jet", jet, n_max=5,
        attrs=("energy", "pt", "eta", "phi", "mass", "btag"),
    )

    # AK8 jets
    events = set_vars(
        events, "fatjet", fatjet, n_max=3,
        attrs=("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32"),
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

    # weights
    events = self[weights](events, **kwargs)

    return events


@ml_inputs.init
def ml_inputs_init(self: Producer) -> None:
    # put ML input columns in separate namespace/table
    self.ml_namespace = "MLInput"

    # store column names
    self.ml_columns = {
        "n_jet",
        "n_fatjet",
    } | {
        f"jet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "mass", "btag")
        for i in range(5)
    } | {
        f"fatjet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    } | {
        f"lepton_{var}"
        for var in ("energy", "pt", "eta", "phi")
    } | {
        f"met_{var}"
        for var in ("pt", "phi")
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
