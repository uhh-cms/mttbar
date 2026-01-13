#!/usr/bin/env bash

ML_MODEL=v1_AN_v12

# common args
common_args=(

    --version 251209_v5
    --config run3_mtt_2024_nano_v15_new
    --analysis mtt.config.run3.analysis_mtt.analysis_mtt_new
    --local-scheduler False
    --cf.BundleRepo-custom-checksum v_0113_1125
    --workers 10
    --cf.CalibrateEvents-htcondor-memory 7GB
    --cf.SelectEvents-htcondor-memory 7GB
    $@
)

# reduce events
# loop over datasets to avoid having too many htcondor jobs at once (capped at 5000 jobs)
for ds in tt st dy w_lnu qcd vv; do
    echo "Reducing $ds"
    echo claw run cf.ReduceEventsWrapper \
    --cf.ReduceEvents-workflow htcondor \
    --cf.ReduceEvents-pilot \
    --cf.ReduceEvents-htcondor-memory 4GB \
    --datasets $ds \
    "${common_args[@]}"
done

# produce columns
for prod in ttbar features weights ml_inputs; do
    echo claw run cf.ProduceColumnsWrapper \
        --datasets all \
        --cf.ProduceColumns-producer $prod \
        --cf.ProduceColumns-workflow htcondor \
        --cf.ProduceColumns-selector default \
        --cf.ProduceColumns-pilot \
        --cf.ProduceColumns-htcondor-memory 5GB \
        --cf.MergeReducedEvents-workflow local \
        --cf.MergeSelectionStats-workflow local \
        --remove-output 0,a,y \
        "${common_args[@]}"
done

# ML training and plotting
echo law run cf.MLTraining \
    --ml-model $ML_MODEL \
    --workflow htcondor \
    --cf.MLTraining-htcondor-memory 30GB \
    --cf.MLTraining-htcondor-runtime 4h \
    --mtt.MLPreTraining-htcondor-memory 10GB \
    "${common_args[@]}"

echo law run mtt.PlotMLResultsSingleFold \
    --ml-model $ML_MODEL \
    --fold 0 \
    --workflow htcondor \
    --mtt.PlotMLResultsSingleFold-htcondor-memory 20GB \
    "${common_args[@]}"

# final plots of output node distributions and ttbar mass
law run cf.PlotVariables1D \
    --selector default \
    --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
    --ml-models $ML_MODEL \
    --hist-producer all_weights \
    --variables ttbar_mass_ext,mlscore.tt,mlscore.st,mlscore.other \
    --categories incl,1m__dnn_st,1e__dnn_st,1m__dnn_other,1e__dnn_other \
    --processes qcd,st,dy,w_lnu,tt \
    --yscale log \
    --skip-ratio \
    --plot-suffix log \
    --workflow htcondor \
    --file-types pdf,png \
    --cms-label simpw \
    "${common_args[@]}"
