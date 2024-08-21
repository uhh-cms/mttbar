#!/usr/bin/env bash

#
# example commands for plotting ML scores
#

# law args
args=(
    --version test_dnn
    --ml-models simple
    --producers default,ml_inputs
    --categories 1e
    --variables "simple.score_*"
    --hide-errors
    --skip-ratio
    --shape-norm
    --processes
        tt,w_lnu,dy,st,qcd,vv,zprime_tt_m500_w50,zprime_tt_m1000_w100,zprime_tt_m3000_w300
    # unstack all processes
    --process-settings
        "tt,unstack:w_lnu,unstack:dy,unstack:st,unstack:qcd,unstack:vv,unstack:zprime_tt_m500_w50:color1=#000000:zprime_tt_m1000_w100:color1=#cccccc:zprime_tt_m3000_w300:color1=#666666"
    --workers 8
    --workflow htcondor
    --local-scheduler False
    $@
)

law run cf.PlotVariables1D \
    "${args[@]}"
