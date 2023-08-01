# coding: utf-8

"""
Column producers related to leptons.
"""
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer(
    uses={
        "channel_id",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
    },
    produces={
        "Lepton.*",
    },
)
def choose_lepton(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Choose either muon or electron as the main lepton per event
    based on `channel_id` information and write it to a new column
    `Lepton`.
    """

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

    # if no lepton, ensure optional type is on fields
    # and not on record itself, and fill with zeroes
    lepton = ak.merge_option_of_records(lepton)
    lepton = ak.fill_none(lepton, 0)

    # attach lorentz vector behavior to lepton
    lepton = ak.with_name(lepton, "PtEtaPhiMLorentzVector")

    # commit lepton to events array
    events = set_ak_column(events, "Lepton", lepton)

    return events
