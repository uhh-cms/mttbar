#!/usr/bin/env bash

# set parameters
# datasets=(tt_sl_powheg st_tchannel_t_4f_powheg dy_m4to50_ht800to1500_madgraph ww_pythia w_lnu_mlnu0to120_ht800to1500_madgraph qcd_ht1000to1200_madgraph data_mu_c)
datasets=(tt_sl_powheg)
datasets_mc=(tt_sl_powheg st_tchannel_t_4f_powheg dy_m4to50_ht800to1500_madgraph ww_pythia w_lnu_mlnu0to120_ht800to1500_madgraph qcd_ht1000to1200_madgraph)

version=test_updates_251013_v2
analysis=mtt.config.run3.analysis_mtt.analysis_mtt_new
config=run3_mtt_2024_nano_v15_limited_new
calibrator=skip_jecunc
selector=default
reducer=default

datasets_str=$(IFS=,; echo "${datasets[*]}")
datasets_mc_str=$(IFS=,; echo "${datasets_mc[*]}")

# process datasets
for dataset in "${datasets[@]}"; do
# for dataset in $datasets_str; do
    echo "Processing dataset: $dataset"
    echo law run cf.CalibrateEvents \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --workflow local \
    --calibrator $calibrator \
    --remove-output 0,a,y
    law run cf.SelectEvents \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrators $calibrator \
    --selector $selector
    echo law run cf.ReduceEvents \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrators $calibrator \
    --selector $selector \
    --reducer $reducer
    echo law run cf.ProduceColumns \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrators $calibrator \
    --producer ttbar \
    --selector $selector \
    --reducer $reducer
    echo law run cf.ProduceColumns \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrators $calibrator \
    --producer features \
    --selector $selector \
    --reducer $reducer
    # don't run weights for data
    if [[ $dataset == data* ]]; then
        continue
    fi
    echo law run cf.ProduceColumns \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrators $calibrator \
    --producer weights \
    --selector $selector \
    --reducer $reducer
done

# plot cutflow of mc samples  # FIXME not working
echo law run cf.PlotCutflow \
    --workers 1 \
    --shape-norm \
    --yscale log \
    --processes tt \
    --process-settings tt,unstack:st,unstack:dy,unstack:w_lnu,unstack:qcd,unstack \
    --categories 1m,1e \
    --version $version \
    --variable mc_weight \
    --config $config \
    --analysis $analysis \
    --file-type png,pdf \
    --datasets tt_sl_powheg \
    --remove-output 0,a,y


    # --processes tt,st,dy,w_lnu,qcd \
    # --datasets $datasets_mc_str \

# create yield table of samples
echo law run cf.CreateYieldTable \
    --version $version \
    --analysis $analysis \
    --config $config \
    --categories 1m,1e \
    --producers ttbar,features,weights \
    --calibrator $calibrator \
    --table-format simple \
    --workers 1 \
    --workflow local \
    --datasets tt_sl_powheg
    # --print-status 4,1
    # --remove-output 0,a,y


#     --datasets $datasets_str \

# plot pt of muon and electron in 1e and 1m - normalized
echo law run cf.PlotVariables1D \
    --version $version \
    --analysis $analysis \
    --config $config \
    --producers add_prod_cats,features,weights \
    --datasets $datasets_str \
    --categories 1m__0t,1e \
    --variables electron_pt,muon_pt \
    --file-types pdf,png \
    --yscale log \
    --remove-output 0,a,y \
    --shape-norm \
    --plot-suffix norm \
    --workers 1 \
    --workflow local \
    --local-scheduler false \
    --selector $selector \
    --reducer $reducer

# plot pt of muon and electron in 1e and 1m - not normalized
# law run cf.PlotVariables1D \
#     --version $version \
#     --analysis $analysis \
#     --config $config \
#     --producers ttbar,features,weights \
#     --datasets $datasets_str \
#     --categories 1m,1e \
#     --variables electron_pt,muon_pt \
#     --file-types pdf,png \
#     --yscale log \
#     --remove-output 0,a,y

# plot ttbar_mass in 1e and 1m - normalized
# law run cf.PlotVariables1D \
#     --version $version \
#     --analysis $analysis \
#     --config $config \
#     --producers ttbar,features,weights \
#     --datasets $datasets_mc_str \
#     --categories 1m,1e \
#     --variables ttbar_mass \
#     --file-types pdf,png \
#     --yscale log \
#     --remove-output 0,a,y \
#     --shape-norm \
#     --plot-suffix norm

# plot ttbar_mass in 1e and 1m - not normalized
# law run cf.PlotVariables1D \
#     --version $version \
#     --analysis $analysis \
#     --config $config \
#     --producers ttbar,features,weights \
#     --datasets $datasets_mc_str \
#     --categories 1m,1e \
#     --variables ttbar_mass \
#     --file-types pdf,png \
#     --yscale log \
#     --remove-output 0,a,y

# echo law run cf.PlotVariables1D \
#     --version $version \
#     --analysis $analysis \
#     --config $config \
#     --producers ttbar,features,weights \
#     --datasets $datasets_str \
#     --categories 1m,1e \
#     --variables n_jet,n_muon,n_electron \
#     --file-types pdf,png \
#     --shape-norm \
#     --remove-output 0,a,y \
#     --workflow local \
#     --workers 8
