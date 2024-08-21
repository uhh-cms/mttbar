#!/usr/bin/env bash

#
# plot the distribution of the chi2 variable
# for various processes in different categories
#

# common arguments for all tasks
args=(
    --version v3
    --categories 1e,1e__0t,1e__1t,1e__0t__chi2pass,1e__1t__chi2pass
    --hide-errors
    --skip-ratio
    --workers 8
    --workflow htcondor
    --local-scheduler False
    --shape-norm True
    --remove-output 0,a,y
)

# plot the chi2 distributions
law run cf.PlotVariables1D \
    --variables chi2,chi2_lt30 \
    --processes \
        tt,w_lnu,dy,st,qcd,vv,zprime_tt_m500_w50,zprime_tt_m1000_w100,zprime_tt_m3000_w300 \
    --process-settings \
        "zprime_tt_m500_w50,color1=#000000:zprime_tt_m1000_w100,color1=#cccccc:zprime_tt_m3000_w300,color1=#666666" \
    --yscale log \
    "${args[@]}"
