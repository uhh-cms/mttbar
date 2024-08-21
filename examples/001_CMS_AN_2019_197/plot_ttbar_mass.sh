#!/usr/bin/env bash

#
# plot the distribution of the ttbar mass
# for various processes in different categories
#

# note: our naming convention from AN-19-197 vs our category names
#   SRbin1_0T: 1e__0t__chi2pass__acts_0_5
#   SRbin1_1T: 1e__1t__chi2pass__acts_0_5
#   SRbin2_0T: 1e__0t__chi2pass__acts_5_7
#   SRbin2_1T: 1e__1t__chi2pass__acts_5_7
#   SRbin3:    1e__chi2pass__acts_7_9
#   SRbin4:    1e__chi2pass__acts_9_1

# common arguments for all tasks
args=(
    --version v3
    --categories 1e,1e__0t,1e__1t,1e__0t__chi2pass,1e__1t__chi2pass,1e__0t__chi2pass__acts_0_5,1e__1t__chi2pass__acts_0_5,1e__0t__chi2pass__acts_5_7,1e__1t__chi2pass__acts_5_7,1e__chi2pass__acts_7_9,1e__chi2pass__acts_9_1
    --hide-errors
    --skip-ratio
    --workers 8
    --workflow htcondor
    --local-scheduler False
    --shape-norm True
    --remove-output 0,a,y
)

# two versions of the ttbar mass
law run cf.PlotVariables1D \
    --variables ttbar_mass,ttbar_mass_wide \
    --processes \
        tt,w_lnu,dy,st,qcd,vv,zprime_tt_m500_w50,zprime_tt_m1000_w100,zprime_tt_m3000_w300 \
    --yscale log \
    --process-settings \
        "zprime_tt_m500_w50,color1=#000000:zprime_tt_m1000_w100,color1=#cccccc:zprime_tt_m3000_w300,color1=#666666" \
    "${args[@]}"
