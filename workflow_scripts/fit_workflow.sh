#!/usr/bin/env bash

common_args=(
    --version 260804_v8__update
    --configs run3_mtt_2025_nano_v15_new,run3_mtt_2024_nano_v15_new
    # --config run3_mtt_2024_nano_v15_new
    --analysis mtt.config.run3.analysis_mtt.analysis_mtt_new
    --local-scheduler False
    --htcondor-memory 6GB
    # --workers 1
    --htcondor-memory 6GB
    # --no-poll
    --poll-interval 1
    --tasks-per-job 10
    --cf.BundleRepo-custom-checksum v_1103_1250
    --hist-producer no_trigger_weights
    --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_v4_2425
    # --print-status 1
    --cf.MergeShiftedHistograms-version 260804_v8
)

# for fit_mode in v3 v2 v2_tt_st_only v2_tt_only; do
for fit_mode in v2__wo_qcd__dec_th; do
    # for mass in 500 1000 2000 4500 9000; do
    # for mass in 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000; do
    for mass in 1200; do
        for frac in 0.01 0.1 0.3; do
        # for frac in 0.1; do
            width=$(awk -v m="$mass" -v f="$frac" 'BEGIN{printf "%d", m*f}')
            signal_sample="zprime_tt_m${mass}_w${width}"
            inf_model="zp_m${mass}_w${width}__${fit_mode}"
            echo claw run cf.cms.CreateDatacards \
                --inference-model $inf_model \
                --cf.ProduceColumns-workflow local \
                --workflow local \
                "${common_args[@]}"
            echo "Running fits for mass=${mass}, width=${width}, fit_mode=${fit_mode}, inf_model=${inf_model}"
            echo claw run mtt.AsymptoticLimits \
                --inference-model ${inf_model} \
                --workflow local \
                --run-mode expected \
                --cminPreFit 1 \
                "${common_args[@]}"
            echo "Running impacts for mass=${mass}, width=${width}, fit_mode=${fit_mode}, inf_model=${inf_model}"
            echo claw run mtt.PlotImpacts \
                --inference-model ${inf_model} \
                --per-page 15 \
                --height 450 \
                --label-size 0.05 \
                --left-margin 0.8 \
                --impacts-mode sens \
                "${common_args[@]}"
        done
    done
    echo claw run mtt.PlotLimits \
        --masses 400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,4500,5000,6000,7000,8000,9000 \
        --width-percentage 10 \
        --variant ${fit_mode} \
        --remove-output 0,a,y \
        --include-run2 lumi-scaled \
        --include-theory \
        "${common_args[@]}"
done

claw run mtt.PlotImpactsComparison \
    --masses 1200 \
    --variants v2__wo_qcd__dec_th \
    --impacts-mode sens \
    --params muf_tt,muf_st,mur_tt,mur_st,pu,btag_light,fsr_tt,fsr_st,isr_tt,isr_st,pdf,xsec_st,xsec_tt,prop_binmu_cr1_bin1,prop_binmu_sr2_bin1,prop_binel_sr2_bin5,prop_binel_cr2_bin0 \
    --width-percentages 1,10,30 \
    --remove-output 0,a,y \
    "${common_args[@]}"

for width in 1 10 30; do
    echo claw run mtt.PlotLimitsComparison \
        --masses 1200 \
        --width-percentage ${width} \
        --variants v2__wo_qcd__dec_th \
        "${common_args[@]}"
done

echo claw run mtt.PlotLimitsComparison \
    --masses 1200 \
    --width-percentage 1,10,30 \
    --variants v2__wo_qcd__dec_th \
    "${common_args[@]}"


# two Impacts plots at the same time for testing
echo claw run mtt.PlotImpactsWrapper \
    --inference-models zp_m1200_w120__v2__wo_qcd__dec_th,zp_m1200_w12__v2__wo_qcd__dec_th,zp_m1200_w360__v2__wo_qcd__dec_th \
    --per-page 15 \
    --height 450 \
    --label-size 0.05 \
    --left-margin 0.8 \
    --impacts-mode sens \
    --workers 3 \
    --workflow htcondor \
    --remove-output 0,a,y \
    --pilot \
    "${common_args[@]}"

# two Impacts plots at the same time for testing
echo claw run mtt.PlotImpacts \
    --inference-model zp_m1400_w140__v2__wo_qcd__dec_th \
    --per-page 15 \
    --height 450 \
    --label-size 0.05 \
    --left-margin 0.8 \
    --impacts-mode sens \
    --workers 3 \
    --workflow htcondor \
    --remove-output 0,a,y \
    "${common_args[@]}"
