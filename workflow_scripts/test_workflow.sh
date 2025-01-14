#!/usr/bin/env bash

# set parameters
datasets=(tt_sl_powheg st_tchannel_t_4f_powheg dy_m4to50_ht800to1500_madgraph ww_pythia w_lnu_mlnu0to120_ht800to1500_madgraph qcd_ht1000to1200_madgraph data_mu_c)
datasets_mc=(tt_sl_powheg st_tchannel_t_4f_powheg dy_m4to50_ht800to1500_madgraph ww_pythia w_lnu_mlnu0to120_ht800to1500_madgraph qcd_ht1000to1200_madgraph)

version=test_updates_250113
analysis=mtt.config.run3.analysis_mtt.analysis_mtt
config=run3_mtt_2022_preEE_nano_v12_limited
calibrator=skip_jecunc

datasets_str=$(IFS=,; echo "${datasets[*]}")
datasets_mc_str=$(IFS=,; echo "${datasets_mc[*]}")

# process datasets
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    law run cf.CalibrateEvents \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrator $calibrator
    law run cf.SelectEvents \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrator $calibrator
    law run cf.ProduceColumns \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrator $calibrator \
    --producer ttbar
    law run cf.ProduceColumns \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrator $calibrator \
    --producer features
    # don't run weights for data
    if [[ $dataset == data* ]]; then
        continue
    fi
    law run cf.ProduceColumns \
    --version $version \
    --analysis $analysis \
    --config $config \
    --dataset $dataset \
    --calibrator $calibrator \
    --producer weights
done

# plot cutflow of mc samples
# law run cf.PlotCutflow \
#     --workers 5 \
#     --shape-norm \
#     --yscale log \
#     --processes tt,st,dy,w_lnu,qcd \
#     --process-settings tt,unstack:st,unstack:dy,unstack:w_lnu,unstack:qcd,unstack \
#     --categories 1m,1e \
#     --version $version \
#     --variable mc_weight \
#     --config $config \
#     --analysis $analysis \
#     --file-type png,pdf \
#     --datasets $datasets_mc_str \
#     --remove-output 0,a,y

# create yield table of samples
# law run cf.CreateYieldTable \
#     --version $version \
#     --analysis $analysis \
#     --config $config \
#     --categories 1m,1e \
#     --datasets $datasets_str \
#     --producers ttbar,features,weights \
#     --calibrator $calibrator \
#     --table-format simple \
#     --workers 8 \
#     --workflow htcondor \
#     --remove-output 0,a,y

# plot pt of muon and electron in 1e and 1m - normalized
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
#     --remove-output 0,a,y \
#     --shape-norm \
#     --plot-suffix norm \
#     --workers 8 \
#     --workflow local

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
