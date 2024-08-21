#!/usr/bin/env bash

#
# plot the distribution of gen-level variables
# and the 2D distributions of matched reco and gen
# variable pairs
#

# law args
args=(
    --version test_gen_studies
    --categories 1e,1e__0t,1e__1t,1m,1m__0t,1m__1t
    --workers 8
    --workflow htcondor
    --local-scheduler False
    $@
)

# 2D gen-reco plots
for process in tt zprime_tt_m1000_w100; do
    law run cf.PlotVariables2D \
        --variables gen_ttbar_mass-ttbar_mass,gen_top_had_eta-top_had_eta,gen_top_lep_pt-top_lep_pt,gen_top_had_pt-top_had_pt,deltar_gen_top_had-deltar_gen_top_lep,gen_cos_theta_star-cos_theta_star,gen_abs_cos_theta_star-abs_cos_theta_star,deltar_gen_top_had_wide-deltar_gen_top_lep_wide \
        --processes $process \
        "${args[@]}"
done

# 1D gen plots
law run cf.PlotVariables1D \
    --skip-ratio \
    --hide-errors \
    --processes \
        tt,w_lnu,dy,st,qcd,vv,zprime_tt_m500_w50,zprime_tt_m1000_w100,zprime_tt_m3000_w300 \
    --variables \
        gen_ttbar_mass,gen_top_had_eta,gen_top_lep_pt,gen_top_had_pt,deltar_gen_top_had,deltar_gen_top_lep,gen_cos_theta_star,gen_abs_cos_theta_star,deltar_gen_top_had_wide,deltar_gen_top_lep_wide \
    "${args[@]}"
