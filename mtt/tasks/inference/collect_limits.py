
import law
import luigi
from columnflow.tasks.framework.base import CommandTask, Requirements
from columnflow.tasks.framework.mixins import (
    CalibratorClassesMixin,
    HistHookMixin,
    HistProducerClassMixin,
    MLModelsMixin,
    ProducerClassesMixin,
    ReducerClassMixin,
    SelectorClassMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import dev_sandbox, maybe_import

from mtt.tasks.base import MTTTask
from mtt.tasks.inference.limits import AsymptoticLimits

pd = maybe_import("pandas")

logger = law.logger.get_logger(__name__)


class LimitsPlottingBaseTask(
    MTTTask,
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    HistProducerClassMixin,
    HistHookMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
    CommandTask,
):
    single_config = False
    masses = law.CSVParameter(cls=luigi.IntParameter, description="mass points to include")
    strict = luigi.BoolParameter(
        default=False,
        description="require every requested combination to exist instead of skipping missing ones",
    )

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs.update(self.requires_from_branch())
        return reqs


class CollectLimits(LimitsPlottingBaseTask):
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar_dev.sh")
    workflow = "local"   # aggregation is cheap and local-only; never submit this one remotely

    variants = law.CSVParameter(default=("v2",), description="inference-model variants to scan")
    width_percentages = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(10,),
        description="relative signal widths (as % of mass) to scan",
    )
    source_csv = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description=(
            "path to an existing limits.csv from a prior, broader CollectLimits run to reuse "
            "instead of re-deriving from AsymptoticLimits; errors if any requested "
            "(mass, width_pct, variant) combination is missing from it"
        ),
    )

    reqs = Requirements(
        RemoteWorkflow.reqs,
        AsymptoticLimits=AsymptoticLimits,
    )

    def create_branch_map(self):
        return [{}]

    def _model_name(self, mass, width_pct):
        width = (mass // 100) * width_pct
        return width, f"zp_m{mass}_w{width}"  # base name; variant suffix appended by caller

    def _candidates(self):
        for mass in self.masses:
            for width_pct in self.width_percentages:
                width, base_name = self._model_name(mass, width_pct)
                for variant in self.variants:
                    model = f"{base_name}__{variant}"
                    task = self.reqs.AsymptoticLimits.req(self, inference_model=model, branch=0)
                    yield mass, width_pct, width, variant, task

    def requires(self):
        if self.source_csv != law.NO_STR:
            return {}  # bypassing AsymptoticLimits entirely by explicit request
        if self.strict:
            return {(m, wp, v): t for m, wp, _, v, t in self._candidates()}
        return {}

    def output(self):
        return {"limits_csv": self.target("limits.csv"), "manifest": self.target("manifest.json")}

    def store_parts(self):
        parts = super().store_parts()
        masses_str = f"m{min(self.masses)}-{max(self.masses)}_n{len(self.masses)}"
        variants_str = "-".join(sorted(self.variants))
        widths_str = "-".join(str(w) for w in sorted(self.width_percentages))
        scan_id = f"{masses_str}__w{widths_str}__{variants_str}"
        parts.insert_after("hist_producer", "limits_scan", scan_id)
        return parts

    def _run_from_source_csv(self):
        src = pd.read_csv(self.source_csv)
        requested = {
            (mass, width_pct, variant)
            for mass in self.masses
            for width_pct in self.width_percentages
            for variant in self.variants
        }
        available = set(zip(src["mass"], src["width_pct"], src["variant"]))
        missing = requested - available
        if missing:
            raise Exception(
                f"source CSV '{self.source_csv}' is missing {len(missing)} requested "
                f"combinations: {sorted(missing)}"
            )
        idx = src.set_index(["mass", "width_pct", "variant"]).index
        subset = src[idx.isin(requested)]
        self.output()["limits_csv"].dump(subset, formatter="pandas", index=False)
        self.output()["manifest"].dump(
            {"found": sorted(requested), "missing": [], "source_csv": self.source_csv},
            formatter="json",
        )

    def run(self):
        if self.source_csv != law.NO_STR:
            self._run_from_source_csv()
            return
        rows, found, missing = [], [], []
        for mass, width_pct, width, variant, task in self._candidates():
            key = (mass, width_pct, variant)
            if not task.complete():
                missing.append(key)
                continue
            row = pd.read_csv(task.output()["asymptotic_limits_csv"].path) \
                .set_index("limit_type")["limit_value"].to_dict()
            row.update(mass=mass, width_pct=width_pct, width=width, variant=variant)
            rows.append(row)
            found.append(key)

        if missing:
            logger.warning(f"skipped {len(missing)} missing (mass, width_pct, variant) combinations: {missing}")

        self.output()["limits_csv"].dump(pd.DataFrame(rows), formatter="pandas", index=False)
        self.output()["manifest"].dump({"found": found, "missing": missing}, formatter="json")
