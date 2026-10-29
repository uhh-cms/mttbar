
import law
import luigi
from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.util import dev_sandbox, maybe_import

from mtt.tasks.inference.collect_limits import CollectLimits, LimitsPlottingBaseTask

pd = maybe_import("pandas")
np = maybe_import("numpy")

logger = law.logger.get_logger(__name__)

PATH_TO_RUN2_LIMITS = "/data/dust/user/matthiej/run2_datacards/limits/limits_{model}.csv"  # model in RSGluon, ZprimeDM, ZPrime_w10, ZPrime_w1, ZPrime_w30

RUN2_THEORY_LINES = {
    "ZPrime_w1": {
        "th": np.array([
            5.83131e+01, 1.36051e+01, 4.50540e+00, 1.80866e+00,
            8.13716e-01, 3.97420e-01, 2.05510e-01, 1.10890e-01,
            6.17038e-02, 3.52336e-02, 2.05665e-02, 1.21935e-02,
            7.34662e-03, 4.46826e-03, 2.75870e-03, 1.72335e-03,
            1.09115e-03, 6.99838e-04, 4.58135e-04, 3.04742e-04,
            2.07506e-04, 1.44911e-04, 1.03407e-04, 7.60116e-05,
            5.71530e-05, 4.42244e-05, 3.49246e-05,
        ]),
        "X_th": np.array([
            0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
            3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5,
            5.75, 6, 6.25, 6.5, 6.75, 7,
        ]) * 1000,
    },
    "ZPrime_w10": {
        "th": np.array([
            5.36522e+02, 1.31954e+02, 4.48526e+01, 1.83735e+01,
            8.47610e+00, 4.24656e+00, 2.26215e+00, 1.26395e+00,
            7.34314e-01, 4.41032e-01, 2.72788e-01, 1.73249e-01,
            1.12874e-01, 7.53710e-02, 5.15542e-02, 3.61230e-02,
            2.59114e-02, 1.90265e-02, 1.42839e-02, 1.09496e-02,
            8.55804e-03, 6.80722e-03, 5.50276e-03, 4.50943e-03,
            3.74190e-03, 3.13889e-03, 2.65864e-03
        ]),
        "X_th": np.array([
            0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
            3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5,
            5.75, 6, 6.25, 6.5, 6.75, 7,
        ]) * 1000,
    },
    "ZPrime_w30": {
        "th": np.array([
            1.32811e+03, 3.61926e+02, 1.29361e+02, 5.52047e+01,
            2.65207e+01, 1.38696e+01, 7.74166e+00, 4.55142e+00,
            2.79430e+00, 1.78062e+00, 1.17252e+00, 7.95149e-01,
            5.53916e-01, 3.95485e-01, 2.88839e-01, 2.15406e-01,
            1.63747e-01, 1.26704e-01, 9.95997e-02, 7.94496e-02,
            6.42139e-02, 5.25241e-02, 4.34250e-02, 3.62486e-02,
            3.05223e-02, 2.59064e-02, 2.21476e-02
        ]),
        "X_th": np.array([
            0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
            3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5,
            5.75, 6, 6.25, 6.5, 6.75, 7,
        ]) * 1000,
    },
}


class PlotLimits(LimitsPlottingBaseTask):
    """
    Plot expected (and observed, if present) limits vs. mass for a single inference-model variant.
    """
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    variant = luigi.Parameter(default="v2", description="inference-model variant to plot")
    width_percentage = luigi.IntParameter(default=10, description="relative signal width (% of mass)")
    include_run2 = luigi.ChoiceParameter(default="None", choices=["none", "raw", "lumi-scaled"], description="Include Run 2 limits in plot.")
    include_theory = luigi.BoolParameter(default=False, description="Include Run 2 limits in plot.")

    reqs = Requirements(RemoteWorkflow.reqs, CollectLimits=CollectLimits)

    def create_branch_map(self):
        return [{}]

    def requires(self):
        return {
            "limits": self.reqs.CollectLimits.req(
                self,
                masses=self.masses,
                variants=(self.variant,),
                width_percentages=(self.width_percentage,),
                strict=self.strict,
            ),
        }

    def output(self):
        return {
            "plot_pdf": self.target(f"limits_vs_mass__{self.variant}.pdf"),
            "plot_png": self.target(f"limits_vs_mass__{self.variant}.png"),
        }

    def store_parts(self):
        parts = super().store_parts()
        masses_str = f"m{min(self.masses)}-{max(self.masses)}_n{len(self.masses)}"
        plot_id = f"{masses_str}__w{self.width_percentage}__{self.variant}"
        parts.insert_after("hist_producer", "limits_plot", plot_id)
        return parts

    def run(self):
        import matplotlib.pyplot as plt
        import mplhep as hep
        lumi = 0
        for cfg in self.config_insts:
            lumi += cfg.x.luminosity.nominal / 1000
        
        # reading prefit signal norm from config (should be the same (0.1) for all configs)
        prefit_norm = self.config_insts[0].x.signal_norm.nominal

        df = pd.read_csv(self.input()["limits"]["limits_csv"].path)
        df = df[df["variant"] == self.variant].sort_values("mass")
        if df.empty:
            raise Exception(f"no limits found for variant '{self.variant}' in collected CSV")
        if self.include_run2:
            df_run2 = pd.read_csv(PATH_TO_RUN2_LIMITS.format(model=f"ZPrime_w{self.width_percentage}"))

        plt.style.use(hep.style.CMS)
        fig, ax = plt.subplots(figsize=(12, 8))
        if "observed" in df.columns and df["observed"].notna().any():
            ax.plot(
                df["mass"] / 1000,
                df["observed"] * prefit_norm,
                color="black", label="Observed"
            )

        ax.plot(
            df["mass"] / 1000,
            df["expected_50"] * prefit_norm,
            color="black", linestyle="--", label="Expected"
        )

        ax.fill_between(
            df["mass"] / 1000,
            df["expected_2.5"] * prefit_norm,
            df["expected_97.5"] * prefit_norm,
            color="#fdd835", label="95% expected"
        )
        ax.fill_between(
            df["mass"] / 1000,
            df["expected_16"] * prefit_norm,
            df["expected_84"] * prefit_norm,
            color="#4caf50", label="68% expected"
        )

        # include expected Run 2 limits in blue
        if self.include_run2 == "raw":
            ax.plot(
                df_run2["m"] / 1000,
                df_run2["exp"],
                label="Run 2 Expected\n(CMS-PAS-B2G-22-006)",
                color="blue",
                linestyle=":",
                zorder=7,
            )
        elif self.include_run2 == "lumi-scaled":
            run2_label = "Run 2 Expected (lumi scaled)\n(CMS-PAS-B2G-22-006)"
            ax.plot(
                df_run2["m"] / 1000,
                df_run2["exp"] * 0.79,
                label=run2_label,
                color="purple",
                linestyle=":",
                zorder=7,
            )
        if self.include_theory:
            model = f"ZPrime_w{self.width_percentage}"
            ax.plot(
                RUN2_THEORY_LINES[model]["X_th"] / 1000,  # convert to TeV
                RUN2_THEORY_LINES[model]["th"],
                color="red",
                linestyle="-",
                zorder=5,
                label=rf"Z' {self.width_percentage}% width (13 TeV)"
            )
        if self.include_run2 == "None":
            lumi_str = f"{lumi:,.1f}"
            com_str = "13.6 TeV"
        else:
            lumi_str = f"138/{lumi:,.1f}"
            com_str = "13/13.6 TeV"
        hep.cms.label("Private Work", data=False, ax=ax, lumi=lumi_str, com=com_str)

        ax.set_xlabel("Mass [TeV]")
        ax.set_ylabel(r"$\sigma \times BR (Z\prime \rightarrow t \bar{t})$ [pb]")
        ax.set_yscale("log")
        ax.legend(title="95% CL upper limit", loc="upper right", frameon=False)
        plt.tight_layout()

        self.output()["plot_pdf"].parent.touch()
        plt.savefig(self.output()["plot_pdf"].path)
        plt.savefig(self.output()["plot_png"].path)
        plt.close()


class PlotLimitsComparison(LimitsPlottingBaseTask):
    """
    Compare limits across inference-model variants at one or more mass points: a ratio plot
    (primary output), overlaid mass-scan curves, and — only when few enough masses are requested
    to stay readable — a grouped bar chart per mass.
    """
    sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/venv_columnar.sh")

    variants = law.CSVParameter(default=("default", "v2"), description="inference-model variants to compare")
    width_percentage = luigi.IntParameter(default=10, description="relative signal width (% of mass)")
    reference_variant = luigi.Parameter(
        default=law.NO_STR,
        description="variant used as the ratio denominator; defaults to the first entry in --variants",
    )

    reqs = Requirements(RemoteWorkflow.reqs, CollectLimits=CollectLimits)

    def create_branch_map(self):
        return [{}]

    @property
    def ref_variant(self):
        return self.reference_variant if self.reference_variant != law.NO_STR else self.variants[0]

    def requires(self):
        return {
            "limits": self.reqs.CollectLimits.req(
                self,
                masses=self.masses,
                variants=self.variants,
                width_percentages=(self.width_percentage,),
                strict=self.strict,
            ),
        }

    def output(self):
        out = {
            "ratio_plot_pdf": self.target("ratio_vs_mass.pdf"),
            "ratio_plot_png": self.target("ratio_vs_mass.png"),
            "overlay_plot_pdf": self.target("overlay_vs_mass.pdf"),
            "overlay_plot_png": self.target("overlay_vs_mass.png"),
        }
        if len(self.masses) <= 5:
            out["bar_plot_pdf"] = self.target("bar_per_mass.pdf")
            out["bar_plot_png"] = self.target("bar_per_mass.png")
        return out

    def store_parts(self):
        parts = super().store_parts()
        masses_str = f"m{min(self.masses)}-{max(self.masses)}_n{len(self.masses)}"
        variants_str = "-".join(sorted(self.variants))
        plot_id = f"{masses_str}__w{self.width_percentage}__{variants_str}"
        parts.insert_after("hist_producer", "limits_comparison", plot_id)
        return parts

    def run(self):
        import matplotlib.pyplot as plt
        import mplhep as hep

        df = pd.read_csv(self.input()["limits"]["limits_csv"].path)
        df = df[df["variant"].isin(self.variants)]

        found_variants = set(df["variant"])
        missing_variants = set(self.variants) - found_variants
        if missing_variants:
            logger.warning(f"no limits found for variants: {sorted(missing_variants)}")
        if self.ref_variant not in found_variants:
            raise Exception(f"reference variant '{self.ref_variant}' has no data to compare against")

        plt.style.use(hep.style.CMS)
        colors = plt.get_cmap("tab10").colors

        # --- overlay: expected_50 vs mass, one line per variant ---
        fig, ax = plt.subplots(figsize=(12, 8))
        hep.cms.label("Private Work", data=False, ax=ax)
        for i, variant in enumerate(self.variants):
            sub = df[df["variant"] == variant].sort_values("mass")
            if sub.empty:
                continue
            ax.plot(sub["mass"], sub["expected_50"], marker="o", color=colors[i % 10], label=variant)
        ax.set_xlabel("Mass [GeV]")
        ax.set_ylabel("Expected limit")
        ax.set_yscale("log")
        ax.legend(title="Variant", frameon=False)
        plt.tight_layout()
        self.output()["overlay_plot_pdf"].parent.touch()
        plt.savefig(self.output()["overlay_plot_pdf"].path)
        plt.savefig(self.output()["overlay_plot_png"].path)
        plt.close()

        # --- ratio: expected_50[variant] / expected_50[ref_variant] vs mass ---
        ref = df[df["variant"] == self.ref_variant].set_index("mass")["expected_50"]
        fig, ax = plt.subplots(figsize=(12, 8))
        hep.cms.label("Private Work", data=False, ax=ax)
        for i, variant in enumerate(self.variants):
            if variant == self.ref_variant:
                continue
            sub = df[df["variant"] == variant].set_index("mass")["expected_50"]
            common_mass = sub.index.intersection(ref.index)
            if common_mass.empty:
                continue
            ratio = (sub.loc[common_mass] / ref.loc[common_mass]).sort_index()
            ax.plot(ratio.index, ratio.values, marker="o", color=colors[i % 10], label=f"{variant} / {self.ref_variant}")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Mass [GeV]")
        ax.set_ylabel(f"Expected limit ratio (ref: {self.ref_variant})")
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(self.output()["ratio_plot_pdf"].path)
        plt.savefig(self.output()["ratio_plot_png"].path)
        plt.close()

        # --- bar chart: grouped by mass, only when few masses requested ---
        if "bar_plot_pdf" in self.output():
            fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(self.masses)), 8))
            hep.cms.label("Private Work", data=False, ax=ax)
            masses_sorted = sorted(df["mass"].unique())
            x = range(len(masses_sorted))
            width = 0.8 / max(len(self.variants), 1)
            for i, variant in enumerate(self.variants):
                sub = df[df["variant"] == variant].set_index("mass")["expected_50"]
                values = [sub.get(m, float("nan")) for m in masses_sorted]
                offsets = [xi + i * width for xi in x]
                ax.bar(offsets, values, width=width, color=colors[i % 10], label=variant)
            ax.set_xticks([xi + width * (len(self.variants) - 1) / 2 for xi in x])
            ax.set_xticklabels([str(m) for m in masses_sorted])
            ax.set_xlabel("Mass [GeV]")
            ax.set_ylabel("Expected limit")
            ax.set_yscale("log")
            ax.legend(title="Variant", frameon=False)
            plt.tight_layout()
            plt.savefig(self.output()["bar_plot_pdf"].path)
            plt.savefig(self.output()["bar_plot_png"].path)
            plt.close()
