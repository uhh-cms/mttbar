
import law
import luigi
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import dev_sandbox, maybe_import

from mtt.tasks.inference.collect_impacts import CollectImpacts
from mtt.tasks.inference.collect_limits import LimitsPlottingBaseTask

pd = maybe_import("pandas")

logger = law.logger.get_logger(__name__)

IMPACT_UP_COLOR = "#E24A33"
IMPACT_DOWN_COLOR = "#348ABD"


class PlotImpactsComparison(LimitsPlottingBaseTask):
    """
    Compare nuisance-parameter pulls and signed impacts on r across mass points and, optionally,
    inference-model variants. One figure per requested nuisance parameter; variants become
    colored/offset series within each figure.

    A parameter/mass/variant combination that's simply absent from the collected data (e.g. a
    signal-region-specific nuisance requested alongside a control-region-only variant) is left
    out of that series rather than failing the task. A parameter with no data at all across every
    requested variant still gets its declared output written, as a placeholder explaining why,
    so the task's completeness isn't blocked by a param that was never going to have data.
    """
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    variants = law.CSVParameter(default=("v2",), description="inference-model variants to compare")
    width_percentage = luigi.IntParameter(default=10, description="relative signal width (% of mass)")
    impacts_mode = luigi.Parameter(default="default", description="Impacts task mode to plot")
    params = law.CSVParameter(description="nuisance parameter name(s) to plot, one figure each")
    impact_yscale = luigi.ChoiceParameter(choices=("linear", "symlog"), default="linear")
    impact_linthresh = luigi.FloatParameter(default=-1.0, significant=False)  # <0 => auto

    reqs = Requirements(RemoteWorkflow.reqs, CollectImpacts=CollectImpacts)

    def create_branch_map(self):
        return [{}]

    def requires(self):
        return {
            "impacts": self.reqs.CollectImpacts.req(
                self,
                masses=self.masses,
                variants=self.variants,
                width_percentages=(self.width_percentage,),
                impacts_modes=(self.impacts_mode,),
                strict=self.strict,
            ),
        }

    def output(self):
        return {
            param: self.target(f"impacts_compare__{param}.pdf")
            for param in self.params
        }

    def store_parts(self):
        parts = super().store_parts()
        masses_str = f"m{min(self.masses)}-{max(self.masses)}_n{len(self.masses)}"
        variants_str = "-".join(sorted(self.variants))
        plot_id = f"{masses_str}__w{self.width_percentage}__{self.impacts_mode}__{variants_str}"
        parts.insert_after("hist_producer", "impacts_comparison", plot_id)
        return parts

    def run(self):
        import matplotlib.pyplot as plt

        outputs = self.output()
        pending = [param for param in self.params if not outputs[param].exists()]

        if not pending:
            logger.info("all requested parameter plots already exist, nothing to do")
            return

        skipped = set(self.params) - set(pending)
        if skipped:
            logger.info(f"skipping {len(skipped)} already-existing plots: {sorted(skipped)}")

        df = pd.read_csv(self.input()["impacts"]["impacts_csv"].path)

        for param in pending:
            sub = df[(df["param_name"] == param) & (df["variant"].isin(self.variants))]
            if sub.empty:
                logger.warning(
                    f"no impacts data found for parameter '{param}' in any requested variant "
                    f"{list(self.variants)}; writing placeholder instead of skipping the whole task",
                )
                self._write_placeholder(param, plt)
                continue
            self._plot_param(sub, param, plt)

    def _plot_param(self, sub, param, plt):
        found_variants = [v for v in self.variants if v in set(sub["variant"])]
        missing_variants = set(self.variants) - set(found_variants)
        if missing_variants:
            logger.warning(f"no data for variants {sorted(missing_variants)} on parameter '{param}'")

        masses_sorted = sorted(sub["mass"].unique())
        colors = plt.get_cmap("tab10").colors
        n = len(found_variants)

        fig, (ax_pull, ax_impact) = plt.subplots(
            2, 1, figsize=(max(6, 0.6 * len(masses_sorted) + 2), 7), sharex=True,
        )

        all_impacts = []
        for i, variant in enumerate(found_variants):
            vsub = sub[sub["variant"] == variant].set_index("mass")
            skipped_masses = sorted(set(masses_sorted) - set(vsub.index))
            if skipped_masses:
                logger.warning(
                    f"variant '{variant}' has no '{param}' entry at masses {skipped_masses}; "
                    f"leaving those points out",
                )
            vsub = vsub.sort_index()

            offset = (i - (n - 1) / 2.0) * 0.15
            color = colors[i % len(colors)]
            x = [masses_sorted.index(m) + offset for m in vsub.index]

            fit_lo = vsub["fit_mid"] - vsub["fit_lo"]
            fit_hi = vsub["fit_hi"] - vsub["fit_mid"]
            ax_pull.errorbar(
                x, vsub["fit_mid"], yerr=[fit_lo, fit_hi],
                fmt="o", color=color, capsize=3, linestyle="none", label=variant,
            )

            impact_down = vsub["r_down"] - vsub["r_nom"]
            impact_up = vsub["r_up"] - vsub["r_nom"]
            ax_impact.vlines(x, impact_down, impact_up, color="gray", alpha=0.25, linewidth=1.2)
            ax_impact.plot(
                x, impact_up, "^", color=IMPACT_UP_COLOR, linestyle="none",
                markeredgecolor=color, markeredgewidth=1.4,
                label=f"{variant} +1$\\sigma$" if n > 1 else "+1$\\sigma$ Impact",
            )
            ax_impact.plot(
                x, impact_down, "v", color=IMPACT_DOWN_COLOR, linestyle="none",
                markeredgecolor=color, markeredgewidth=1.4,
                label=f"{variant} -1$\\sigma$" if n > 1 else "-1$\\sigma$ Impact",
            )
            all_impacts.extend(impact_up.tolist() + impact_down.tolist())

        if self.impact_yscale == "symlog":
            nonzero = sorted(abs(v) for v in all_impacts if v == v and v != 0)  # v==v filters NaN
            if nonzero:
                linthresh = (
                    self.impact_linthresh if self.impact_linthresh > 0
                    else max(nonzero[len(nonzero) // 2] * 3, 1e-12)
                )
                ax_impact.set_yscale("symlog", linthresh=linthresh)

        ax_pull.axhline(0, color="gray", lw=0.8, ls="--")
        ax_pull.axhline(-1, color="gray", lw=0.8, ls=":")
        ax_pull.axhline(1, color="gray", lw=0.8, ls=":")
        ax_pull.set_ylim(-1.1, 1.1)
        ax_pull.set_ylabel(r"$(\hat\theta-\theta)/\sigma_{\rm prefit}$")
        ax_pull.set_title(f"{param}: postfit value across mass points")
        ax_pull.legend(fontsize=8)

        ax_impact.axhline(0, color="gray", lw=0.8, ls="--")
        ax_impact.set_ylabel(r"impact on $\hat{r}$")
        ax_impact.set_xlabel("mass point")
        handles, labels_ = ax_impact.get_legend_handles_labels()
        seen = dict(zip(labels_, handles))
        ax_impact.legend(seen.values(), seen.keys(), fontsize=7, ncol=2)
        ax_impact.set_xticks(range(len(masses_sorted)))
        ax_impact.set_xticklabels([str(m) for m in masses_sorted], rotation=45, ha="right")

        fig.tight_layout()
        target = self.output()[param]
        target.parent.touch()
        fig.savefig(target.path, dpi=150)
        plt.close(fig)
