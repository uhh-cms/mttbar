# m(ttbar) analysis

Analysis based on [columnflow](https://github.com/uhh-cms/columnflow), [law](https://github.com/riga/law) and [order](https://github.com/riga/order).


### Quickstart

A couple test tasks are listed below.
They might require a **valid voms proxy** for accessing input data.

```shell
# clone the project
git clone --recursive git@github.com:uhh-cms/mttbar.git
cd mttbar

# source the setup and store decisions in .setups/dev.sh (arbitrary name)
source setup.sh dev

# index existing tasks once to enable auto-completion for "law run"
law index --verbose

# run your first task
# (they are all shipped with columnflow and thus have the "cf." prefix)
law run cf.ReduceEvents \
    --version v1 \
    --dataset st_tchannel_t \
    --branch 0

# create a plot
law run cf.PlotVariables \
    --version v1 \
    --datasets st_tchannel_t \
    --producers features \
    --variables jet1_pt \
    --categories 1e \
    --hist-producer no_trigger_weights \
    --branch 0

# create a (test) datacard (CMS-style)
law run cf.CreateDatacards \
    --version v1 \
    --producers features \
    --inference-model test \
    --hist-producer no_trigger_weights \
    --workers 3
```

### Statistical inference with Combine

Full documentation of the tasks, the sandbox and the output layout: [docs/combine.md](docs/combine.md).

The tasks in `mtt/tasks/inference/` wrap the CMS Combine fit workflow. They run in the `combine_cmssw` sandbox (a CMSSW area with Combine, set up on first use by `sandboxes/combine_cmssw.sh`). Remote jobs (HTCondor) use the same sandbox, so it must be listed in `ana.x.cmssw_sandboxes` of the analysis (already done for `analysis_mtt`).

The chain is `CombineDatacards` -> `CreateWorkspace` -> `AsymptoticLimits` / `Impacts` -> plotting. Every task requires its predecessors, so it is enough to run the last one.

The examples below share these arguments, abbreviated as `$COMMON`:

```shell
COMMON="--version v1 --configs run3_mtt_2024_nano_v15_new --producers features --hist-producer no_trigger_weights"
```

```shell
# datacards and workspace for one inference model
law run mtt.CombineDatacards --inference-model zp_m1000_w100__v2 $COMMON
law run mtt.CreateWorkspace --inference-model zp_m1000_w100__v2 $COMMON

# expected asymptotic limits (--run-mode: blind, expected, observed or both)
law run mtt.AsymptoticLimits --inference-model zp_m1000_w100__v2 --run-mode expected $COMMON
# restrict the scanned signal-strength range / inject a signal (non-default values are part of the output path)
law run mtt.AsymptoticLimits --inference-model zp_m1000_w100__v2 --run-mode expected --rMin -1 --rMax 5 $COMMON

# impacts, as an HTCondor workflow; --impacts-mode is "bkgonly" (default) or "sens"
# "sens" first requires AsymptoticLimits and derives rMax / rMin from its expected limit
# (r_max = 4 * expected_97.5, r_min = -0.3 * r_max); --rMin/--rMax/--expectSignal are only valid in "sens"
law run mtt.Impacts --inference-model zp_m1000_w100__v2 --impacts-mode sens --workflow htcondor $COMMON

# impacts plot (requires the Impacts above)
law run mtt.PlotImpacts --inference-model zp_m1000_w100__v2 --impacts-mode sens --workflow htcondor $COMMON
```

#### Several inference models at once

The Combine tasks listed below have a `...Wrapper` variant that takes `--inference-models` (comma-separated, brace expansion works) and runs the task once per model. All other arguments are passed on unchanged.

```shell
law run mtt.PlotImpactsWrapper \
    --inference-models zp_m1000_w10__v2,zp_m1000_w100__v2,zp_m1000_w300__v2 \
    --impacts-mode sens --workflow htcondor --workers 3 $COMMON

law run mtt.AsymptoticLimitsWrapper --inference-models "zp_m1000_w{10,100,300}__v2" $COMMON
```

Wrappers: `CombineDatacardsWrapper`, `CreateWorkspaceWrapper`, `AsymptoticLimitsWrapper`, `ImpactsWrapper`, `PlotImpactsWrapper`. The tasks log a harmless `set_status_message not set` warning when started through a wrapper.

#### Limits and impacts over a mass scan

```shell
# collect the limits of all mass points into one csv, then plot them
# Collect* only aggregate limits/impacts that already exist; run the per-mass tasks first (e.g. mtt.AsymptoticLimitsWrapper) or add --strict. After computing more points, rerun with --remove-output, otherwise the old aggregate is reused.
law run mtt.CollectLimits --masses 1000,2000,3000 --variants v2 --width-percentages 10 $COMMON
law run mtt.PlotLimits --masses 1000,2000,3000 --variant v2 --width-percentage 10 --include-run2 lumi-scaled $COMMON

# same for impacts
law run mtt.CollectImpacts --masses 1000,2000,3000 --impacts-modes bkgonly,sens $COMMON

# compare impacts of several variants for several signal widths (one plot per parameter and width)
law run mtt.PlotImpactsComparison --masses 1200 --variants v2 --impacts-mode sens \
    --params pu,btag_light,xsec_tt --width-percentages 1,10,30 $COMMON

# limits comparison takes a single width per call
law run mtt.PlotLimitsComparison --masses 1200 --variants v2 --width-percentage 10 $COMMON
```

#### Notes

- Output paths encode the mode and settings (for example `impacts__sens__s0.0_r0.0_20.0`), so runs with different settings do not overwrite each other.
- The tasks expose many Combine options as parameters (see `COMBINE_OPTION_MIXINS` in `mixins.py`). Each task lists the options it actually translates into the Combine command in its `implemented_options`; setting any other option to a non-default value raises a `NotImplementedError`, so it is never silently ignored. To add one, extend `build_command` of the task and add the name to its `implemented_options`.
- Remote jobs reuse the software installed next to the columnflow venvs (`htcondor_share_software: True` in `law.cfg`), so run any Combine task locally once before submitting jobs.


### Development

- Source hosted at [GitHub](https://github.com/uhh-cms/mttbar)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/uhh-cms/mttbar/issues)