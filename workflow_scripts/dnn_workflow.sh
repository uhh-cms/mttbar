#!/usr/bin/env bash

ML_MODEL=v7

# common args
common_args=(

    --version 260219_v7
    --config run3_mtt_2024_nano_v15_new
    --analysis mtt.config.run3.analysis_mtt.analysis_mtt_new
    --local-scheduler False
    --cf.BundleRepo-custom-checksum v_0326_1400
    --workers 50
    --cf.CalibrateEvents-htcondor-memory 7GB
    --cf.SelectEvents-htcondor-memory 7GB
    $@
)

# reduce events
# loop over datasets to avoid having too many htcondor jobs at once (capped at 5000 jobs)
for ds in tt st dy w_lnu qcd vv data zprime_tt_m500_w5_madgraph zprime_tt_m4000_w40_madgraph zprime_tt_m4500_w45_madgraph zprime_tt_m7000_w70_madgraph; do
    echo "Reducing $ds"
    echo claw run cf.ReduceEventsWrapper \
    --cf.ReduceEvents-workflow htcondor \
    --cf.ReduceEvents-htcondor-memory 4GB \
    --cf.ReduceEvents-pilot \
    --datasets $ds \
    --cf.CalibrateEvents-version 260216_v6 \
    "${common_args[@]}"
done

# produce columns
for prod in ml_inputs; do
    # for ds in tt_fh_powheg; do
    for ds in bkg; do
        echo "Producing columns with $prod for dataset $ds"
        echo law run cf.ProduceColumnsWrapper \
            --datasets $ds \
            --cf.ProduceColumns-producer $prod \
            --cf.ProduceColumns-selector default \
            --cf.ProduceColumns-workflow htcondor \
            --cf.ProduceColumns-pilot \
            --cf.ProduceColumns-htcondor-memory 2GB \
            --cf.CalibrateEvents-version 260216_v6 \
            --cf.MergeSelectionStats-workflow local \
            --remove-output 0,a,y \
            "${common_args[@]}"
    done
done

echo law run cf.PlotVariables1D \
    --selector default \
    --variables AN_v12_mli_jet_btagUParTAK4B_1,AN_v12_mli_jet_btagUParTAK4B_2,AN_v12_mli_jet_btagUParTAK4B_3 \
    --producers add_prod_cats,ttbar,features,weights,ml_inputs \
    --hist-producer all_weights \
    --processes tt,w_lnu,st,dy,vv,qcd \
    --process-settings tt,unstack:w_lnu,unstack:st,unstack:dy,unstack:vv,unstack:qcd,unstack \
    --yscale log \
    --categories 1m,1e \
    --cms-label simpw \
    --file-types pdf,png \
    --workflow local \
    "${common_args[@]}"

# plot input variable distributions
echo law run cf.PlotVariables1D \
    --variables "gen_ttbar_mass_narrow_ext" \
    --selector default \
    --producers add_prod_cats,ttbar,features,weights \
    --hist-producer all_weights \
    --processes zprime_tt_m500_w5,zprime_tt_m4000_w40,zprime_tt_m7000_w70 \
    --process-settings tt,unstack:zprime_tt_m7000_w70,unstack:zprime_tt_m500_w5,unstack:zprime_tt_m4000_w40,unstack:zprime_tt_m7000_w70,unstack \
    --categories incl \
    --cms-label simpw \
    --file-types pdf,png \
    --workflow local \
    --shape-norm \
    --yscale log \
    --skip-ratio \
    --remove-output 0,a,y \
    "${common_args[@]}"

# ML training and plotting
echo law run cf.MLTraining \
    --ml-model $ML_MODEL \
    --cf.MLTraining-htcondor-memory 30GB \
    --cf.MLTraining-htcondor-runtime 4h \
    --mtt.MLPreTraining-htcondor-memory 35GB \
    --mtt.MLPreTraining-workflow htcondor \
    --workflow htcondor \
    "${common_args[@]}"

law run mtt.PlotMLResultsSingleFold \
    --ml-model $ML_MODEL \
    --workflow htcondor \
    --mtt.PlotMLResultsSingleFold-htcondor-memory 20GB \
    --mtt.MLPreTraining-htcondor-memory 30GB \
    --mtt.MLPreTraining-workflow htcondor \
    "${common_args[@]}"

# final plots of output node distributions and ttbar mass
echo law run cf.PlotVariables1D \
    --cf.MLTraining-htcondor-memory 30GB \
    --cf.MLTraining-htcondor-runtime 4h \
    --mtt.MLPreTraining-htcondor-memory 15GB \
    --mtt.MLPreTraining-workflow local \
    --cf.MLEvaluation-version 251209_v5 \
    --selector default \
    --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
    --ml-models $ML_MODEL \
    --hist-producer all_weights \
    --variables ttbar_mass_ext,mlscore.tt,mlscore.st,mlscore.other \
    --categories v12_simplified \
    --processes qcd,st,dy,w_lnu,tt,zprime_tt_m7000_w70,zprime_tt_m500_w5,zprime_tt_m4000_w40 \
    --yscale log \
    --skip-ratio \
    --plot-suffix log \
    --workflow local \
    --file-types pdf,png \
    --cms-label simpw \
    --print-status 3 \
    "${common_args[@]}"

# create datacards
for inf_model in an_v12_simplified__m500_w5 an_v12_simplified__m4000_w40 an_v12_simplified__m4500_w45 an_v12_simplified__m7000_w70; do
# for inf_model in an_v12_simplified__m7000_w70; do
    echo "Creating datacard for model $inf_model"
    echo claw run cf.CreateDatacards \
        --inference-model $inf_model \
        --hist-producer all_weights \
        --selector default \
        --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
        --workflow htcondor \
        "${common_args[@]}"
done