#!/usr/bin/env bash

# ML_MODEL=v1_2425
ML_MODEL=v4_2425

# common args
common_args=(

    --version 260804_v8
    # --configs run3_mtt_2025_nano_v15_new,run3_mtt_2024_nano_v15_new
    --config run3_mtt_2024_nano_v15_new
    --analysis mtt.config.run3.analysis_mtt.analysis_mtt_new
    --local-scheduler False
    --cf.BundleRepo-custom-checksum v_0930_1400
    --workers 6
    --htcondor-memory 6GB
    # --no-poll
    --poll-interval 5
    --tasks-per-job 10
    # --print-status 0
    $@
)

# # reduce events
# # loop over datasets to avoid having too many htcondor jobs at once (capped at 5000 jobs)
# # for ds in data_e data_mu zprime_tt dy w_lnu 'qcd,vv' 'tt,st'; do
# for ds in zprime_tt; do
#     echo "Reducing $ds"
#     # echo claw run cf.CalibrateEventsWrapper \
#     #     --datasets $ds \
#     #     --cf.CalibrateEvents-workflow htcondor \
#     #     --print-status 0 \
#     #     "${common_args[@]}"
#     # claw run cf.SelectEventsWrapper \
#     #     --datasets $ds \
#     #     --cf.SelectEvents-workflow local \
#     #     "${common_args[@]}"
#     echo claw run cf.ReduceEventsWrapper \
#         --cf.ReduceEvents-workflow local \
#         --cf.ReduceEvents-pilot \
#         --datasets $ds \
#         --pilot \
#         "${common_args[@]}"
# done

# produce columns
# call external bash script to produce columns for all datasets in the dataset.yaml file
# ./data/dust/user/matthiej/mttbar/workflow_scripts/produce_columns.sh

# create histograms
# ./data/dust/user/matthiej/mttbar/workflow_scripts/shifted_histograms.sh
echo claw run cf.PlotVariables1D \
    --variables ttbar_mass \
    --categories 1m \
    --producers add_prod_cats,ttbar,features,weights \
    --processes bkg,zprime_tt_m500_w50,zprime_tt_m3000_w300,zprime_tt_m5000_w500 \
    --process-settings zprime_tt_m3000_w300,scale=2000:zprime_tt_m500_w50,scale=2000:zprime_tt_m5000_w500,scale=2000 \
    --hist-producer no_trigger_weights \
    --cms-label simpw \
    --file-types pdf,png \
    --workflow local \
    --yscale log \
    --plot-suffix fsp \
    --no-poll \
    --skip-ratio \
    --custom-style-config my_default \
    --remove-output 0,a,y \
    "${common_args[@]}"

echo claw run cf.PlotVariables1D \
    --variables chi2_lt100 \
    --categories 1e,1e__1t,1e__0t \
    --producers add_prod_cats,ttbar,features,weights \
    --processes bkg,zprime_tt_m3000_w300 \
    --process-settings zprime_tt_m3000_w300,scale=2000 \
    --hist-producer no_trigger_weights \
    --cms-label pw \
    --file-types pdf,png \
    --workflow local \
    --yscale log \
    --plot-suffix fsp \
    --no-poll \
    --skip-ratio \
    --custom-style-config my_default \
    --remove-output 0,a,y \
    "${common_args[@]}"

echo claw run cf.PlotVariables1D \
    --variables top_had_mass,top_lep_mass \
    --categories 1e,1e__1t__chi2pass,1e__1t,1e__0t,1e__0t__chi2pass \
    --producers add_prod_cats,ttbar,features,weights \
    --processes bkg,zprime_tt_m3000_w300 \
    --process-settings zprime_tt_m3000_w300,scale=stack \
    --hist-producer no_trigger_weights \
    --cms-label pw \
    --file-types pdf,png \
    --workflow local \
    --yscale log \
    --plot-suffix fsp \
    --no-poll \
    --skip-ratio \
    --custom-style-config my_default \
    --remove-output 0,a,y \
    "${common_args[@]}"

echo claw run cf.PlotVariables1D \
    --variables jet_lep_pt_rel_zoom,jet_lep_delta_r_zoom,n_fatjet \
    --selector ttbar_res_sel \
    --producers add_prod_cats,ttbar,features,weights \
    --hist-producer no_trigger_weights \
    --processes data,bkg \
    --categories 1m,1e,incl \
    --cms-label simpw \
    --file-types pdf,png \
    --workflow local \
    --yscale log \
    --plot-suffix for_an \
    --no-poll \
    --custom-style-config for_an \
    "${common_args[@]}"

for proc in tt st qcd dy vv data; do
    echo "Plotting 2D distributions for process $proc"
    claw run cf.PlotVariables2D \
        --variables jet_lep_pt_rel_zoom-jet_lep_delta_r_zoom \
        --selector ttbar_res_sel \
        --producers add_prod_cats,ttbar,features,weights \
        --hist-producer no_trigger_weights \
        --processes $proc \
        --categories 1m,1e,incl \
        --cms-label simpw \
        --file-types pdf,png \
        --workflow local \
        --plot-suffix for_an \
        --no-poll \
        --custom-style-config for_an \
        "${common_args[@]}"
done
# # plot variable distributions
#     # --cf.MergeHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} \
#     # --processes data,bkg,sig,sig_narrow,sig_wide,sig_low_mass,sig_medium_mass,sig_high_mass,sig_very_high_mass \
# for sig_group in sig sig_narrow sig_wide sig_low_mass sig_medium_mass sig_high_mass sig_very_high_mass; do
#     echo "Plotting basic variable distributions for signal group $sig_group"
#     echo claw run cf.PlotVariables1D \
#         --variables base_kinematics \
#         --selector ttbar_res_sel \
#         --producers add_prod_cats,ttbar,features,weights \
#         --hist-producer no_trigger_weights \
#         --processes data,bkg,$sig_group \
#         --process-settings zprime_tt_m500_w5,scale=stack:zprime_tt_m1000_w10,scale=stack:zprime_tt_m3000_w30,scale=stack:zprime_tt_m7000_w70,scale=stack:zprime_tt_m500_w50,scale=stack:zprime_tt_m1000_w100,scale=stack:zprime_tt_m3000_w300,scale=stack:zprime_tt_m7000_w700,scale=stack:zprime_tt_m500_w150,scale=stack:zprime_tt_m1000_w300,scale=stack:zprime_tt_m3000_w900,scale=stack:zprime_tt_m7000_w2100,scale=stack \
#         --categories 1m,1e,1m__0t,1m__1t,1e__0t,1e__1t,chi2p,chi2f \
#         --cms-label simpw \
#         --file-types pdf,png \
#         --workflow htcondor \
#         --yscale log \
#         --plot-suffix for_an \
#         --no-poll \
#         --custom-style-config for_an \
#         "${common_args[@]}"
#     echo "Plotting sensitive variable distributions for signal group $sig_group"
#     echo claw run cf.PlotVariables1D \
#         --variables sensitive_vars \
#         --selector ttbar_res_sel \
#         --producers add_prod_cats,ttbar,features,weights \
#         --hist-producer no_trigger_weights \
#         --processes bkg,$sig_group \
#         --process-settings zprime_tt_m500_w5,scale=stack:zprime_tt_m1000_w10,scale=stack:zprime_tt_m3000_w30,scale=stack:zprime_tt_m7000_w70,scale=stack:zprime_tt_m500_w50,scale=stack:zprime_tt_m1000_w100,scale=stack:zprime_tt_m3000_w300,scale=stack:zprime_tt_m7000_w700,scale=stack:zprime_tt_m500_w150,scale=stack:zprime_tt_m1000_w300,scale=stack:zprime_tt_m3000_w900,scale=stack:zprime_tt_m7000_w2100,scale=stack \
#         --categories 1m,1e,1m__0t,1m__1t,1e__0t,1e__1t,chi2p,chi2f \
#         --cms-label simpw \
#         --file-types pdf,png \
#         --workflow htcondor \
#         --yscale log \
#         --plot-suffix for_an \
#         --skip-ratio \
#         --no-poll \
#         --custom-style-config for_an \
#         "${common_args[@]}"
#     echo "Plotting gen variable distributions for signal group $sig_group"
#     # gen vars
#     echo claw run cf.PlotVariables1D \
#         --variables reco_vars_gen \
#         --selector ttbar_res_sel \
#         --producers add_prod_cats,ttbar,features,weights \
#         --hist-producer no_trigger_weights \
#         --processes tt,$sig_group \
#         --process-settings zprime_tt_m500_w5,scale=stack:zprime_tt_m1000_w10,scale=stack:zprime_tt_m3000_w30,scale=stack:zprime_tt_m7000_w70,scale=stack:zprime_tt_m500_w50,scale=stack:zprime_tt_m1000_w100,scale=stack:zprime_tt_m3000_w300,scale=stack:zprime_tt_m7000_w700,scale=stack:zprime_tt_m500_w150,scale=stack:zprime_tt_m1000_w300,scale=stack:zprime_tt_m3000_w900,scale=stack:zprime_tt_m7000_w2100,scale=stack:tt,unstack \
#         --categories 1m,1e,1m__0t,1m__1t,1e__0t,1e__1t,chi2p,chi2f,1e__chi2pass \
#         --cms-label simpw \
#         --file-types pdf,png \
#         --workflow htcondor \
#         --yscale log \
#         --plot-suffix for_an \
#         --no-poll \
#         --skip-ratio \
#         --custom-style-config for_an \
#         --shape-norm \
#         "${common_args[@]}"
# done

# echo claw run cf.PlotVariables1D \
#     --variables gen_ttbar_mass_narrow_ext_ext \
#     --selector ttbar_res_sel \
#     --producers add_prod_cats,ttbar,features,weights \
#     --hist-producer no_btag_and_trigger_weights \
#     --processes sig_narrow_all \
#     --process-settings zprime_tt_m400_w4,color=#1313eb,scale=1:zprime_tt_m500_w5,color=#133ceb,scale=1:zprime_tt_m600_w6,color=#1365eb,scale=1:zprime_tt_m700_w7,color=#138eeb,scale=1:zprime_tt_m800_w8,color=#13b8eb,scale=1:zprime_tt_m900_w9,color=#13e1eb,scale=1:zprime_tt_m1000_w10,color=#13ebcc,scale=1:zprime_tt_m1200_w12,color=#13eba3,scale=1:zprime_tt_m1400_w14,color=#13eb7a,scale=1:zprime_tt_m1600_w16,color=#13eb51,scale=1:zprime_tt_m1800_w18,color=#13eb27,scale=1:zprime_tt_m2000_w20,color=#27eb13,scale=1:zprime_tt_m2500_w25,color=#51eb13,scale=1:zprime_tt_m3000_w30,color=#7aeb13,scale=1:zprime_tt_m3500_w35,color=#a3eb13,scale=1:zprime_tt_m4000_w40,color=#cceb13,scale=1:zprime_tt_m4500_w45,color=#ebe113,scale=1:zprime_tt_m5000_w50,color=#ebb813,scale=1:zprime_tt_m6000_w60,color=#eb8e13,scale=1:zprime_tt_m7000_w70,color=#eb6513,scale=1:zprime_tt_m8000_w80,color=#eb3c13,scale=1:zprime_tt_m9000_w90,color=#eb1313,scale=1:tt,unstack \
#     --categories incl,1e,1m \
#     --cms-label simpw \
#     --file-types pdf,png \
#     --workflow local \
#     --yscale log \
#     --plot-suffix for_an \
#     --skip-ratio \
#     --shape-norm \
#     --custom-style-config for_an \
#     --hide-stat-errors \
#     "${common_args[@]}"

# echo claw run cf.PlotVariables1D \
#     --variables gen_ttbar_mass_narrow_ext_ext \
#     --selector ttbar_res_sel \
#     --producers add_prod_cats,ttbar,features,weights \
#     --hist-producer no_btag_and_trigger_weights \
#     --processes sig_medium_all \
#     --process-settings zprime_tt_m400_w40,color=#1313eb,scale=1:zprime_tt_m500_w50,color=#133ceb,scale=1:zprime_tt_m600_w60,color=#1365eb,scale=1:zprime_tt_m700_w70,color=#138eeb,scale=1:zprime_tt_m800_w80,color=#13b8eb,scale=1:zprime_tt_m900_w90,color=#13e1eb,scale=1:zprime_tt_m1000_w100,color=#13ebcc,scale=1:zprime_tt_m1200_w120,color=#13eba3,scale=1:zprime_tt_m1400_w140,color=#13eb7a,scale=1:zprime_tt_m1600_w160,color=#13eb51,scale=1:zprime_tt_m1800_w180,color=#13eb27,scale=1:zprime_tt_m2000_w200,color=#27eb13,scale=1:zprime_tt_m2500_w250,color=#51eb13,scale=1:zprime_tt_m3000_w300,color=#7aeb13,scale=1:zprime_tt_m3500_w350,color=#a3eb13,scale=1:zprime_tt_m4000_w400,color=#cceb13,scale=1:zprime_tt_m4500_w450,color=#ebe113,scale=1:zprime_tt_m5000_w500,color=#ebb813,scale=1:zprime_tt_m6000_w600,color=#eb8e13,scale=1:zprime_tt_m7000_w700,color=#eb6513,scale=1:zprime_tt_m8000_w800,color=#eb3c13,scale=1:zprime_tt_m9000_w900,color=#eb1313,scale=1:tt,unstack \
#     --categories incl,1e,1m \
#     --cms-label simpw \
#     --file-types pdf,png \
#     --workflow local \
#     --yscale log \
#     --plot-suffix for_an \
#     --skip-ratio \
#     --shape-norm \
#     --custom-style-config for_an \
#     --hide-stat-errors \
#     "${common_args[@]}"

# echo claw run cf.PlotVariables1D \
#     --variables gen_ttbar_mass_narrow_ext_ext \
#     --selector ttbar_res_sel \
#     --producers add_prod_cats,ttbar,features,weights \
#     --hist-producer no_btag_and_trigger_weights \
#     --processes sig_wide_all \
#     --process-settings zprime_tt_m400_w120,color=#1313eb,scale=1:zprime_tt_m500_w150,color=#133ceb,scale=1:zprime_tt_m600_w180,color=#1365eb,scale=1:zprime_tt_m700_w210,color=#138eeb,scale=1:zprime_tt_m800_w240,color=#13b8eb,scale=1:zprime_tt_m900_w270,color=#13e1eb,scale=1:zprime_tt_m1000_w300,color=#13ebcc,scale=1:zprime_tt_m1200_w360,color=#13eba3,scale=1:zprime_tt_m1400_w420,color=#13eb7a,scale=1:zprime_tt_m1600_w480,color=#13eb51,scale=1:zprime_tt_m1800_w540,color=#13eb27,scale=1:zprime_tt_m2000_w600,color=#27eb13,scale=1:zprime_tt_m2500_w750,color=#51eb13,scale=1:zprime_tt_m3000_w900,color=#7aeb13,scale=1:zprime_tt_m3500_w1050,color=#a3eb13,scale=1:zprime_tt_m4000_w1200,color=#cceb13,scale=1:zprime_tt_m4500_w1350,color=#ebe113,scale=1:zprime_tt_m5000_w1500,color=#ebb813,scale=1:zprime_tt_m6000_w1800,color=#eb8e13,scale=1:zprime_tt_m7000_w2100,color=#eb6513,scale=1:zprime_tt_m8000_w2400,color=#eb3c13,scale=1:zprime_tt_m9000_w2700,color=#eb1313,scale=1:tt,unstack \
#     --categories incl,1e,1m \
#     --cms-label simpw \
#     --file-types pdf,png \
#     --workflow local \
#     --yscale log \
#     --plot-suffix for_an \
#     --skip-ratio \
#     --shape-norm \
#     --custom-style-config for_an \
#     --hide-stat-errors \
#     "${common_args[@]}"

# for ds in tt_sl_powheg tt_dl_powheg tt_fh_powheg; do
#     echo "Preparing ML events for dataset $ds"
#     echo claw run cf.PrepareMLEventsWrapper \
#         --datasets $ds \
#         --ml-model $ML_MODEL \
#         --producers weights,ml_inputs \
#         --workflow local \
#         --remove-output 0,a,y \
#         "${common_args[@]}"
# done

# # ML training and plotting
# echo law run cf.MLTraining \
#     --ml-model $ML_MODEL \
#     --cf.MLTraining-htcondor-memory 30GB \
#     --cf.MLTraining-htcondor-runtime 4h \
#     --mtt.MLPreTraining-htcondor-memory 35GB \
#     --mtt.MLPreTraining-workflow local \
#     --workflow htcondor \
#     "${common_args[@]}"

# echo law run mtt.PlotMLResultsSingleFold \
#     --ml-model $ML_MODEL \
#     --workflow local \
#     --remove-output 0,a,y \
#     --workflow htcondor \
#     --mtt.PlotMLResultsSingleFold-htcondor-memory 10GB \
#     "${common_args[@]}"

# # for ds in tt st dy w_lnu vv qcd sig_narrow_all sig_medium_all sig_wide_all data; do
# for mass in 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000; do
#     for frac in 0.01 0.1 0.3; do
#         width=$(awk -v m="$mass" -v f="$frac" 'BEGIN{printf "%d", m*f}')
#         echo "Producing ML columns for dataset zprime_tt_m${mass}_w${width}_madgraph"
#         echo claw run cf.ProduceColumnsWrapper \
#             --datasets zprime_tt_m${mass}_w${width}_madgraph \
#             --producers add_ml_cats_$ML_MODEL \
#             --workflow local \
#             --cf.MLEvaluation-workflow local \
#             "${common_args[@]}"
#     done
# done

# # final plots of output node distributions and ttbar mass
#     # --cf.MergeHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} \
# echo law run cf.PlotVariables1D \
#     --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
#     --ml-models $ML_MODEL \
#     --hist-producer no_trigger_weights \
#     --variables ttbar_mass_ext \
#     --categories v12_simplified \
#     --processes tt,st,others,sig \
#     --yscale log \
#     --skip-ratio \
#     --plot-suffix log \
#     --workflow local \
#     --file-types pdf,png \
#     --cms-label simpw \
#     "${common_args[@]}"

# # create datacards
#         # --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} \
# for mass in 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000; do
# # for mass in 500; do
#     for frac in 0.01 0.1 0.3; do
#     # for frac in 0.1; do
#         width=$(awk -v m="$mass" -v f="$frac" 'BEGIN{printf "%d", m*f}')
#         inf_model_split="zp_m${mass}_w${width}__split_eras"
#         # echo "Creating datacard for model $inf_model_split"
#         # claw run cf.CreateDatacards \
#         #     --inference-model $inf_model_split \
#         #     --hist-producer no_trigger_weights \
#         #     --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
#         #     --cf.ProduceColumns-workflow local \
#         #     --workflow local \
#         #     "${common_args[@]}"
#         inf_model_v2="zp_m${mass}_w${width}__v2"
#         echo "Creating datacard for model $inf_model_v2"
#         echo claw run cf.cms.CreateDatacards \
#             --inference-model $inf_model_v2 \
#             --hist-producer no_trigger_weights \
#             --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
#             --cf.ProduceColumns-workflow local \
#             --workflow local \
#             "${common_args[@]}"
#     done
# done

echo claw run cf.PlotVariables1D \
    --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
    --hist-producer no_trigger_weights \
    --variables ttbar_mass_v2 \
    --categories v12_simplified \
    --processes tt,st,others \
    --skip-ratio \
    --workflow local \
    --file-types pdf,png \
    --cms-label simpw \
    --remove-output 0,a,y \
    "${common_args[@]}"

# plot shifted vars
echo claw run cf.PlotShiftedVariables1D \
    --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
    --hist-producer no_trigger_weights \
    --variables ttbar_mass_v2 \
    --categories v12_simplified \
    --processes vv__dy__w_lnu,st,tt,zprime_tt_m3000_w300 \
    --process-settings zprime_tt_m3000_w300,scale=1 \
    --remove-output 0,a,y \
    --skip-legend \
    --yscale log \
    --skip-ratio \
    --plot-suffix fsp \
    --workflow local \
    --file-types pdf,png \
    --cms-label simpw \
    --custom-style-config fit_templates \
    --shift-sources minbias_xs,btag_bc,btag_light,pdf,mur,muf,isr,fsr,muon_reco,muon_id,muon_iso,electron_reco,electron_id_iso_low,electron_id,electron_iso \
    "${common_args[@]}"

echo claw run cf.PlotShiftedVariables1D \
    --producers add_prod_cats,ttbar,features,weights \
    --hist-producer no_trigger_weights \
    --variables jet1_pt,bjet1_pt,fatjet1_eta,jet1_eta,bjet1_eta,fatjet1_eta,electron_pt,muon_pt \
    --categories 1m,1e,1m__0t,1m__1t,1e__0t,1e__1t \
    --processes tt,st,dy,w_lnu,vv,qcd,data \
    --yscale log \
    --workflow htcondor \
    --plot-suffix for_DataMC \
    --file-types pdf,png \
    --cms-label simpw \
    --shift-sources minbias_xs,btag_bc,btag_light,pdf,mur,muf,isr,fsr,muon_reco,muon_id,muon_iso,electron_reco,electron_id_iso_low,electron_id,electron_iso \
    "${common_args[@]}"

echo claw run cf.PlotShiftedVariablesPerConfig1D \
    --producers add_prod_cats,ttbar,features,weights \
    --hist-producer no_trigger_weights \
    --variables jet1_pt,bjet1_pt,fatjet1_eta,jet1_eta,bjet1_eta,fatjet1_eta,electron_pt,muon_pt \
    --categories 1m,1e,1m__0t,1m__1t,1e__0t,1e__1t \
    --processes tt,st,dy,w_lnu,vv,qcd,data \
    --yscale log \
    --plot-suffix for_DataMC \
    --file-types pdf,png \
    --cms-label simpw \
    --shift-sources minbias_xs,btag_bc,btag_light,pdf,mur,muf,isr,fsr,muon_reco,muon_id,muon_iso,electron_reco,electron_id_iso_low,electron_id,electron_iso \
    "${common_args[@]}"

for proc in tt st others; do
# for proc in st; do
    echo "Plotting shifted variables per shift for process $proc"
    echo claw run cf.PlotShiftedVariablesPerShift1D \
        --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
        --hist-producer no_trigger_weights \
        --variables ttbar_mass_v2 \
        --categories v12_simplified \
        --processes $proc \
        --yscale log \
        --plot-suffix uncs_full \
        --workflow local \
        --file-types pdf,png \
        --cms-label simpw \
        --shift-sources minbias_xs,btag_bc,btag_light,pdf,mur,muf,isr,fsr,muon_reco,muon_id,muon_iso,electron_reco,electron_id_iso_low,electron_id,electron_iso \
        "${common_args[@]}"
done

# claw run cf.CreateYieldTable \
#     --categories v12_simplified \
#     --processes tt,st,others \
#     --hist-producer no_weights \
#     --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_v4_2425 \
#     --workflow local \
#     "${common_args[@]}"

echo claw run cf.CreateYieldTable \
    --categories v12_simplified \
    --processes tt,st,dy,w_lnu,vv,qcd,sig_narrow_all,sig_medium_all,sig_wide_all \
    --hist-producer no_trigger_weights \
    --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_v4_2425 \
    --workflow local \
    "${common_args[@]}"

# for mass in 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000; do
for mass in 500; do
    # for frac in 0.01 0.1 0.3; do
    for frac in 0.1; do
        width=$(awk -v m="$mass" -v f="$frac" 'BEGIN{printf "%d", m*f}')
        signal_sample="zprime_tt_m${mass}_w${width}"
        inf_model="zp_m${mass}_w${width}__v2"
        echo "Creating datacard for model $inf_model"
        echo claw run cf.cms.CreateDatacards \
            --inference-model $inf_model \
            --hist-producer no_trigger_weights \
            --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
            --cf.ProduceColumns-workflow local \
            --workflow local \
            "${common_args[@]}"
        
        echo claw run cf.PlotShiftedVariables1D \
            --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
            --hist-producer no_trigger_weights \
            --variables ttbar_mass_v2 \
            --categories v12_simplified \
            --processes tt,st,others,$signal_sample \
            --yscale log \
            --skip-ratio \
            --plot-suffix uncs_full \
            --workflow local \
            --file-types pdf,png \
            --cms-label simpw \
            --shift-sources minbias_xs,btag_bc,btag_light,pdf,mur,muf,isr,fsr,muon_reco,muon_id,muon_iso,electron_reco,electron_id_iso_low,electron_id,electron_iso \
            "${common_args[@]}"

        for proc in tt st others $signal_sample; do
        # for proc in st; do
            echo "Plotting shifted variables per shift for process $proc"
            echo claw run cf.PlotShiftedVariablesPerShift1D \
                --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
                --hist-producer no_trigger_weights \
                --variables ttbar_mass_v2 \
                --categories v12_simplified \
                --processes $proc \
                --yscale log \
                --plot-suffix uncs_full \
                --workflow local \
                --file-types pdf,png \
                --cms-label simpw \
                --shift-sources minbias_xs,btag_bc,btag_light,pdf,mur,muf,isr,fsr,muon_reco,muon_id,muon_iso,electron_reco,electron_id_iso_low,electron_id,electron_iso \
                "${common_args[@]}"
        done
        
        echo claw run mtt.AsymptoticLimits \
            --inference-model $inf_model \
            --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
            --hist-producer no_trigger_weights \
            --workflow local \
            --print-status 0 \
            "${common_args[@]}"
    done
done

# for ds in data_e_c data_e_d data_e_e data_e_f data_e_g data_e_h data_e_i data_mu_c data_mu_d data_mu_e data_mu_f data_mu_g data_mu_h data_mu_i; do
for ds in data_e_c data_e_d data_e_e data_e_f data_e_g data_mu_c data_mu_d data_mu_e data_mu_f data_mu_g; do
    echo "Merging selection stats for dataset $ds"
    echo claw run cf.MergeSelectionStats \
        --selector ttbar_res_sel \
        --calibrators default_nominal \
        --dataset $ds \
        --workflow local \
        "${common_args[@]}"
done