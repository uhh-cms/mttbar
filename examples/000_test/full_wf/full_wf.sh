#!/usr/bin/env bash

#
# commands for testing different parts of the workflow
#

# dedicated version for test
version="test"

# common args for all tasks
args=(
    --version $version
    --dataset
        zprime_tt_m1000_w100_madgraph
    --remove-output 0,a,y
)

# test CalibrateEvents
echo law run cf.CalibrateEventsWrapper \
    "${args[@]}"

# test SelectEvents
echo law run cf.SelectEventsWrapper \
    "${args[@]}"

# test PlotCutflow
plot_cf_args=(
    --version $version
    --categories incl,1e,1e__0t,1e__1t
    --processes
        zprime_tt_m1000_w100
    --dataset
        zprime_tt_m1000_w100_madgraph
    --process-settings
        "zprime_tt_m1000_w100,color1=#000000"
    --shape-norm True
    --view-cmd echo
    --remove-output 0,a,y
)
echo law run cf.PlotCutflow \
    "${plot_cf_args[@]}"

# test PlotCutflowVariables
plot_cf_vars_args=(
    --version $version
    --variables cf_n_toptag
    --categories incl,1e,1e__0t,1e__1t
    --processes
        zprime_tt_m1000_w100
    --dataset
        zprime_tt_m1000_w100_madgraph
    --process-settings
        "zprime_tt_m1000_w100,color1=#000000"
    --hide-errors
    --shape-norm True
    --view-cmd echo
    --remove-output 0,a,y
)
echo run cf.PlotCutflowVariables1D \
    "${plot_cf_vars_args[@]}"

# test ReduceEvents
echo law run cf.ReduceEventsWrapper \
    "${args[@]}"

# test MergeReduceEvents
echo law run cf.MergeReducedEventsWrapper \
    "${args[@]}"

# test ProduceColumns
echo law run cf.ProduceColumnsWrapper \
    "${args[@]}"

# test PlotVariables1D
plot_args=(
    --version $version
    --variables ttbar_mass_wide
    --categories 1e__0t__chi2pass__acts_0_5
    --processes
        zprime_tt_m1000_w100
    --dataset
        zprime_tt_m1000_w100_madgraph
    --process-settings
        "zprime_tt_m1000_w100,color1=#000000"
    --hide-errors
    --shape-norm True
    --view-cmd echo
    --remove-output 0,a,y
)
echo law run cf.PlotVariables1D \
    "${plot_args[@]}"
