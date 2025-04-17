# coding: utf-8

from columnflow.reduction import Reducer, reducer
from columnflow.reduction.default import cf_default
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")

@reducer(
    uses={cf_default},
    produces={cf_default},
)
def default(self: Reducer, events: ak.Array, selection: ak.Array, **kwargs) -> ak.Array:
    # run cf's default reduction which handles event selection and collection creation
    events = self[cf_default](events, selection, **kwargs)

    # store gen level information of the remaining events
    if self.dataset_inst.has_tag("is_sm_ttbar"):
        events = self[gen_parton_top](events, **kwargs)

    if self.dataset_inst.has_tag("is_v_jets"):
        events = self[gen_v_boson](events, **kwargs)

    return events
