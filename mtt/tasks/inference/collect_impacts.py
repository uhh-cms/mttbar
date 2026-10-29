
import json

import law
import luigi
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import dev_sandbox, maybe_import

from mtt.tasks.inference.collect_limits import LimitsPlottingBaseTask
from mtt.tasks.inference.impacts import Impacts

pd = maybe_import("pandas")

logger = law.logger.get_logger(__name__)


class CollectImpacts(LimitsPlottingBaseTask):
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar_dev.sh")
    workflow = "local"

    variants = law.CSVParameter(default=("v2",), description="inference-model variants to scan")
    width_percentages = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(10,),
        description="relative signal widths (as % of mass) to scan",
    )
    impacts_modes = law.CSVParameter(
        default=("default",),
        description="Impacts task 'impacts_mode' values to scan",
    )
    source_csv = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description=(
            "path to an existing impacts.csv from a prior, broader CollectImpacts run to reuse "
            "instead of re-deriving from Impacts; errors if any requested "
            "(mass, width_pct, variant, impacts_mode) combination is missing from it"
        ),
    )

    reqs = Requirements(RemoteWorkflow.reqs, Impacts=Impacts)

    def create_branch_map(self):
        return [{}]

    def _model_name(self, mass, width_pct):
        width = (mass // 100) * width_pct
        return width, f"zp_m{mass}_w{width}"

    def _candidates(self):
        for mass in self.masses:
            for width_pct in self.width_percentages:
                width, base_name = self._model_name(mass, width_pct)
                for variant in self.variants:
                    model = f"{base_name}__{variant}"
                    for impacts_mode in self.impacts_modes:
                        task = self.reqs.Impacts.req(
                            self, inference_model=model, impacts_mode=impacts_mode, branch=0,
                        )
                        yield mass, width_pct, width, variant, impacts_mode, task

    def requires(self):
        if self.source_csv != law.NO_STR:
            return {}
        if self.strict:
            return {(m, wp, v, im): t for m, wp, _, v, im, t in self._candidates()}
        return {}

    def output(self):
        return {"impacts_csv": self.target("impacts.csv"), "manifest": self.target("manifest.json")}

    def store_parts(self):
        parts = super().store_parts()
        masses_str = f"m{min(self.masses)}-{max(self.masses)}_n{len(self.masses)}"
        variants_str = "-".join(sorted(self.variants))
        widths_str = "-".join(str(w) for w in sorted(self.width_percentages))
        modes_str = "-".join(sorted(self.impacts_modes))
        scan_id = f"{masses_str}__w{widths_str}__{variants_str}__{modes_str}"
        parts.insert_after("hist_producer", "impacts_scan", scan_id)
        return parts

    def _run_from_source_csv(self):
        src = pd.read_csv(self.source_csv)
        requested = {
            (mass, width_pct, variant, impacts_mode)
            for mass in self.masses
            for width_pct in self.width_percentages
            for variant in self.variants
            for impacts_mode in self.impacts_modes
        }
        available = set(
            src[["mass", "width_pct", "variant", "impacts_mode"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        missing = requested - available
        if missing:
            raise Exception(
                f"source CSV '{self.source_csv}' is missing {len(missing)} requested "
                f"combinations: {sorted(missing)}"
            )
        idx = src.set_index(["mass", "width_pct", "variant", "impacts_mode"]).index
        subset = src[idx.isin(requested)]
        self.output()["impacts_csv"].dump(subset, formatter="pandas", index=False)
        self.output()["manifest"].dump(
            {"found": sorted(requested), "missing": [], "source_csv": self.source_csv},
            formatter="json",
        )

    def run(self):
        if self.source_csv != law.NO_STR:
            self._run_from_source_csv()
            return
        rows, found, missing = [], [], []
        for mass, width_pct, width, variant, impacts_mode, task in self._candidates():
            key = (mass, width_pct, variant, impacts_mode)
            if not task.complete():
                missing.append(key)
                continue
            with open(task.output()["impacts_json"].path) as f:
                data = json.load(f)
            for entry in data.get("params", []):
                fit_lo, fit_mid, fit_hi = entry["fit"]
                prefit = entry.get("prefit") or [None, None, None]
                r_down, r_nom, r_up = entry["r"]
                rows.append({
                    "mass": mass, "width_pct": width_pct, "width": width,
                    "variant": variant, "impacts_mode": impacts_mode,
                    "param_name": entry["name"], "param_type": entry.get("type"),
                    "fit_lo": fit_lo, "fit_mid": fit_mid, "fit_hi": fit_hi,
                    "prefit_lo": prefit[0], "prefit_mid": prefit[1], "prefit_hi": prefit[2],
                    "r_down": r_down, "r_nom": r_nom, "r_up": r_up,
                    "impact_r": entry.get("impact_r"),
                })
            found.append(key)

        if missing:
            logger.warning(
                f"skipped {len(missing)} missing (mass, width_pct, variant, impacts_mode) "
                f"combinations: {missing}",
            )

        self.output()["impacts_csv"].dump(pd.DataFrame(rows), formatter="pandas", index=False)
        self.output()["manifest"].dump({"found": found, "missing": missing}, formatter="json")
