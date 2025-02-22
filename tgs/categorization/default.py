# coding: utf-8

"""
Categorizers for mttbar trigger study (tgs)
"""

from columnflow.categorization import Categorizer, categorizer
from columnflow.util import maybe_import


ak = maybe_import("awkward")


#
# categorizer functions used by category definitions
#

# -- inclusive category (every event passes)

@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, ak.ones_like(events.event) == 1


# -- electron pt categorizers

@categorizer(uses={"Electron.pt"})
def cat_ele_pt_lt_120(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    electron_pt = ak.firsts(events.Electron.pt, axis=1)
    return events, (electron_pt < 120.0)


@categorizer(uses={"Electron.pt"})
def cat_ele_pt_120_200(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    electron_pt = ak.firsts(events.Electron.pt, axis=1)
    return events, (electron_pt > 120.0) & (electron_pt < 200)


@categorizer(uses={"Electron.pt"})
def cat_ele_pt_gt_200(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    electron_pt = ak.firsts(events.Electron.pt, axis=1)
    return events, (electron_pt > 200.0)


# -- electron trigger pass/fail categorizers

@categorizer(uses={"HLT.Ele35_WPTight_Gsf"})
def cat_ele_trigger_pass(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.HLT.Ele35_WPTight_Gsf


@categorizer(uses={"HLT.Ele35_WPTight_Gsf"})
def cat_ele_trigger_fail(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, ~events.HLT.Ele35_WPTight_Gsf
