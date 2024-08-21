#!/usr/bin/env bash

#
# example commands for plotting jet/lepton distances
#

# law args
args=(
    --version test_jet_lepton_distances
    --categories 1e,1m
    --processes
        tt,w_lnu,dy,st,qcd,vv,zprime_tt_m500_w50,zprime_tt_m1000_w100,zprime_tt_m3000_w300

    #--calibrator skip_jecunc_wo_cleaner
    --calibrator skip_jecunc
    --producers weights,jet_lepton_features 
    --workers 8
    #--workflow htcondor
    #--local-scheduler False
    $@
)

# 2D distributions
law run cf.PlotVariables2D \
    --variables jet_lep_delta_r-jet_lep_pt_rel,jet_lep_delta_r_zoom-jet_lep_pt_rel_zoom \
    --shape-norm True \
    "${args[@]}"

# 1D distributions
law run cf.PlotVariables1D \
    --variables jet_lep_delta_r,jet_lep_pt_rel,jet_lep_delta_r_zoom,jet_lep_pt_rel_zoom \
    --skip-ratio \
    --hide-errors \
    --process-settings \
        "tt,unstack:w_lnu,unstack:dy,unstack:st,unstack:qcd,unstack:vv,unstack:zprime_tt_m500_w50:color1=#000000:zprime_tt_m1000_w100:color1=#cccccc:zprime_tt_m3000_w300:color1=#666666" \
    "${args[@]}"
