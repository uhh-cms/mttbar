#!/usr/bin/env bash

ML_MODEL=v4_2425  # merged 24/25 model

# common args
common_args=(
    --version v0
    --analysis mtt.config.run3.analysis_mtt.analysis_mtt_new
    --configs run3_mtt_2025_nano_v15_new,run3_mtt_2024_nano_v15_new
    # some tasks don't work with multiple configs:
    # --config run3_mtt_2024_nano_v15_new
    --local-scheduler False
    --workers 6
    --htcondor-memory 6GB
    --poll-interval 5
    --tasks-per-job 10
    # --no-poll
    $@
)

# reduce events
# loop over datasets to avoid having too many htcondor jobs at once (capped at 5000 jobs)
for ds in data_e data_mu zprime_tt dy w_lnu 'qcd,vv' 'tt,st'; do
    echo "Reducing $ds"
    # if needed, you can run the calibration and selection steps separately, but try runninng the ReduceEventsWrapper with the 'pilot' option first, as it will run the other steps automatically if needed.
    # echo claw run cf.CalibrateEventsWrapper \
    #     --datasets $ds \
    #     --cf.CalibrateEvents-workflow htcondor \
    #     --print-status 0 \
    #     "${common_args[@]}"
    # claw run cf.SelectEventsWrapper \
    #     --datasets $ds \
    #     --cf.SelectEvents-workflow htcondor \
    #     "${common_args[@]}"
    echo claw run cf.ReduceEventsWrapper \
        --cf.ReduceEvents-workflow htcondor \
        --cf.ReduceEvents-pilot \
        --datasets $ds \
        --pilot \
        "${common_args[@]}"
done

# produce columns
for prod in ttbar features ml_inputs; do
    for ds in data_e data_mu zprime_tt dy w_lnu 'qcd,vv' 'tt,st'; do
        echo "Producing $prod for $ds"
        echo claw run cf.ProduceColumnsWrapper \
            --datasets $ds \
            --cf.ProduceColumns-producer $prod \
            --cf.ProduceColumns-workflow htcondor \
            --cf.ProduceColumns-selector default \
            --cf.ProduceColumns-pilot \
            --cf.ProduceColumns-htcondor-memory 5GB \
            --cf.MergeReducedEvents-workflow local \
            --cf.MergeSelectionStats-workflow local \
            "${common_args[@]}"
    done
done

# plotting
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
    --no-poll \
    --skip-ratio \
    --custom-style-config my_default \
    --remove-output 0,a,y \
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

# ML workflow
for ds in tt_sl_powheg tt_dl_powheg tt_fh_powheg; do
    echo "Preparing ML events for dataset $ds"
    echo claw run cf.PrepareMLEventsWrapper \
        --datasets $ds \
        --ml-model $ML_MODEL \
        --producers weights,ml_inputs \
        --workflow local \
        "${common_args[@]}"
done

echo law run cf.MLTraining \
    --ml-model $ML_MODEL \
    --cf.MLTraining-htcondor-memory 30GB \
    --cf.MLTraining-htcondor-runtime 4h \
    --mtt.MLPreTraining-htcondor-memory 35GB \
    --mtt.MLPreTraining-workflow local \
    --workflow htcondor \
    "${common_args[@]}"

echo law run mtt.PlotMLResultsSingleFold \
    --ml-model $ML_MODEL \
    --workflow local \
    --remove-output 0,a,y \
    --workflow htcondor \
    --mtt.PlotMLResultsSingleFold-htcondor-memory 10GB \
    "${common_args[@]}"

# for ds in tt st dy w_lnu vv qcd sig_narrow_all sig_medium_all sig_wide_all data; do
for mass in 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000; do
    for frac in 0.01 0.1 0.3; do
        width=$(awk -v m="$mass" -v f="$frac" 'BEGIN{printf "%d", m*f}')
        echo "Producing ML columns for dataset zprime_tt_m${mass}_w${width}_madgraph"
        echo claw run cf.ProduceColumnsWrapper \
            --datasets zprime_tt_m${mass}_w${width}_madgraph \
            --producers add_ml_cats_$ML_MODEL \
            --workflow local \
            --cf.MLEvaluation-workflow local \
            "${common_args[@]}"
    done
done

# final plots of ttbar mass in inference categories
    # Adding this MergeHistograms argument submits all histogramming jobs to htcondor without polling.
    # This will speed up the workflow significantly, but you will need to manually check the status of
    # the jobs and resubmit any failed ones. Use with caution!
    # --cf.MergeHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} \
echo law run cf.PlotVariables1D \
    --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_$ML_MODEL \
    --ml-models $ML_MODEL \
    --hist-producer no_trigger_weights \
    --variables ttbar_mass_ext \
    --categories v12_simplified \
    --processes tt,st,others,sig \
    --yscale log \
    --skip-ratio \
    --plot-suffix log \
    --workflow local \
    --file-types pdf,png \
    --cms-label simpw \
    "${common_args[@]}"


# create datacards
    # See comment about MergeHistograms above. You can also use MergeShiftedHistograms to merge the
    # shifted histograms for systematic uncertainties, but this will also submit all jobs to htcondor without polling.
    # --cf.MergeShiftedHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} \
for mass in 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000; do
    for frac in 0.01 0.1 0.3; do
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
        
        # one plot with syst. uncs. included as green error band
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
            # one plot per shift source, with nominal and up/down variations overlaid
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
    done
done

# yield tables, only works with one config (I think)
echo claw run cf.CreateYieldTable \
    --categories v12_simplified \
    --processes tt,st,dy,w_lnu,vv,qcd,sig_narrow_all,sig_medium_all,sig_wide_all \
    --hist-producer no_trigger_weights \
    --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_v4_2425 \
    --workflow local \
    "${common_args[@]}"
