#!/usr/bin/env bash

#
# plot the distribution of cos(theta*)
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

# compare ttbar with Z prime signals
law run cf.PlotVariables1D \
    --variables cos_theta_star,abs_cos_theta_star \
    --processes \
        tt,zprime_tt_m500_w50,zprime_tt_m1000_w100 \
    --process-settings \
        "tt,color1=#ff0000,unstack,hide_errors:zprime_tt_m500_w50,color1=#000000:zprime_tt_m1000_w100,color1=#0000ff" \
    "${args[@]}"

# compare ttbar with heavy Higgs signals
law run cf.PlotVariables1D \
    --variables cos_theta_star,abs_cos_theta_star \
    --processes \
        tt,hscalar_tt_sl_m365_w36p5_res,hpseudo_tt_sl_m365_w36p5_res \
    --process-settings \
        "tt,color1=#ff0000,unstack,hide_errors:hscalar_tt_sl_m365_w36p5_res,color1=#0000ff:hpseudo_tt_sl_m365_w36p5_res,color1=#ff00ff" \
    "${args[@]}"
