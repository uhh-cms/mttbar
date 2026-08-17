# coding: utf-8

"""
Calibration methods.
"""
import functools
import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.met import met_phi
# TODO should we add these later?
# from columnflow.calibration.cms.egamma import electron_scale_smear
# from columnflow.calibration.cms.muon import muon_sr
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.jet import msoftdrop, jet_id, fatjet_id
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from mtt.calibration.jets import jet_energy_nominal, jet_energy, jet_lepton_cleaner
from mtt.util import record_calls, has_tag

ak = maybe_import("awkward")
np = maybe_import("numpy")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

logger = law.logger.get_logger(__name__)


@calibrator(
    uses={
        mc_weight,
        deterministic_seeds,
        jet_lepton_cleaner,
        msoftdrop,
        "Muon.pt",
        "Muon.tunepRelPt",
    },
    produces={
        mc_weight,
        deterministic_seeds,
        jet_lepton_cleaner,
        msoftdrop,
        "Muon.pt",
        "Muon.rawPt",
    },
    jerc_mode=None,  # can be set to "nominal_jercunc" or "all_jercunc" in derived calibrators
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    run_list = []
    with record_calls(self, run_list):
        # use tuneP pT as per recommendation of MUO POG for highPt muons, should be the same as pT for lowPt muons
        events = set_ak_column_f32(events, "Muon.rawPt", events.Muon.pt)
        events = set_ak_column_f32(events, "Muon.pt", events.Muon.tunepRelPt * events.Muon.pt)
        logger.info_once(
            "Finished recalculating muon pt with tuneP for highPt muons. Stored original pt in Muon.rawPt.",
        )
        if self.dataset_inst.is_mc:
            events = self[mc_weight](events, **kwargs)
        events = self[deterministic_seeds](events, **kwargs)
        # run JEC calibrators for AK4, AK8 jets, and AK8 subjets
        # fake subjet area column by setting it to an array with the same structure as the subjet pt column containing
        # 0.5 (needed to be able to use same code as for top-level AK4/AK8 jets, as the producer formally requires an
        # `area` column, despite not actually using it)
        events = set_ak_column_f32(events, "SubJet.area", 0.5 * ak.ones_like(events.SubJet.pt))

        events = self[jet_lepton_cleaner](events, **kwargs)
        if self.jerc_mode == "nominal_jercunc":
            logger.info_once("Running jet energy calibrators with nominal JEC and JER uncertainties.")
            events = self[jet_energy_nominal](events, **kwargs)
        elif self.jerc_mode == "all_jercunc":
            logger.info_once("Running jet energy calibrators with all JEC and JER uncertainties.")
            events = self[jet_energy](events, **kwargs)
        else:
            raise ValueError(f"Unknown jerc_mode: {self.jerc_mode}")
        events = self[msoftdrop](events, **kwargs)

        if self.config_inst.x.year not in [2024, 2025, 2026]:
            events = self[met_phi](events, **kwargs)

        if not has_tag("skip_jet_ids", self.config_inst, self.dataset_inst, operator=any):
            logger.info("Recalulating (fat)jet IDs.")
            events = self[jet_id](events, **kwargs)
            events = self[fatjet_id](events, **kwargs)

    logger.info_once(
        "Finished default calibration steps:\n" +
        "\n".join(run_list),
    )

    return events


@default.init
def default_init(self: Calibrator) -> None:
    if self.config_inst.x.year not in [2024, 2025]:
        self.uses |= {met_phi}
        self.produces |= {met_phi}
    if not has_tag("skip_jet_ids", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {jet_id, fatjet_id}
        self.produces |= {jet_id, fatjet_id}


default_nominal = default.derive("default_nominal", cls_dict={"jerc_mode": "nominal_jercunc"})
default_all = default.derive("default_all", cls_dict={"jerc_mode": "all_jercunc"})


@default_nominal.init
def default_nominal_init(self: Calibrator) -> None:
    self.uses |= {jet_energy_nominal}
    self.produces |= {jet_energy_nominal}
    if self.config_inst.x.year not in [2024, 2025]:
        self.uses |= {met_phi}
        self.produces |= {met_phi}
    if not has_tag("skip_jet_ids", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {jet_id, fatjet_id}
        self.produces |= {jet_id, fatjet_id}


@default_all.init
def default_all_init(self: Calibrator) -> None:
    self.uses |= {jet_energy}
    self.produces |= {jet_energy}
    if self.config_inst.x.year not in [2024, 2025]:
        self.uses |= {met_phi}
        self.produces |= {met_phi}
    if not has_tag("skip_jet_ids", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {jet_id, fatjet_id}
        self.produces |= {jet_id, fatjet_id}
