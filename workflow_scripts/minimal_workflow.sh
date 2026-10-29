#!/usr/bin/env bash

# A minimal workflow to run the mtt analysis on HTCondor or locally, serving as a template.

common_args=(
    --version test_v1
    --config run3_mtt_2025_nano_v15_new
    # --configs run3_mtt_2025_nano_v15_new,run3_mtt_2024_nano_v15_new  # not all tasks take multiple configs
    --analysis mtt.config.run3.analysis_mtt.analysis_mtt_new
    --cf.BundleRepo-custom-checksum v1  # remember to update this when changing anything in the repo when submitting to htcondor!
    --workers 4
    --poll-interval 5  # poll every 5 minutes
    --tasks-per-job 5  # run 5 tasks ("branches") per job
)

# reduce events
# loop over dataset groups to avoid having too many htcondor jobs at once (capped at 5000 jobs)
# either run ReduceEvents-pilot or Calibrate-Select-Reduce workflow, depending on how stable the ReduceEvents workflow is
for ds in data_e data_mu zprime_tt dy w_lnu 'qcd,vv' 'tt,st'; do
    echo "Reducing events for dataset group: $ds"
    # echo claw run cf.CalibrateEventsWrapper \
    #     --datasets $ds \
    #     --cf.CalibrateEvents-workflow htcondor \
    #     "${common_args[@]}"
    # claw run cf.SelectEventsWrapper \
    #     --datasets $ds \
    #     --cf.SelectEvents-workflow local \
    #     "${common_args[@]}"
    echo claw run cf.ReduceEventsWrapper \
        --cf.ReduceEvents-workflow local \
        --cf.ReduceEvents-pilot \
        --datasets $ds \
        --pilot \
        "${common_args[@]}"
done

# produce columns
for prod in add_prod_cats ttbar features weights; do
    for ds in data_e data_mu zprime_tt dy w_lnu 'qcd,vv' 'tt,st'; do
        echo "Producing columns with $prod for dataset $ds"
        echo claw run cf.ProduceColumnsWrapper \
            --datasets $ds \
            --cf.ProduceColumns-producer $prod \
            --cf.ProduceColumns-workflow htcondor \
            --cf.ProduceColumns-pilot \
            --cf.ProduceColumns-htcondor-memory 8GB \
            --cf.MergeSelectionStats-workflow htcondor \
            --cf.MergeReducedEvents-workflow htcondor \
            --pilot \
            "${common_args[@]}"
    done
done

# plot base kinematics distributions for signal groups, MC bkg, and data
for sig_group in sig sig_narrow sig_wide sig_low_mass sig_medium_mass sig_high_mass sig_very_high_mass; do
    echo "Plotting basic variable distributions for signal group $sig_group"
    echo claw run cf.PlotVariables1D \
        --variables base_kinematics \
        --selector ttbar_res_sel \
        --producers add_prod_cats,ttbar,features,weights \
        --hist-producer no_btag_and_trigger_weights \
        --processes data,bkg,$sig_group \
        --process-settings zprime_tt_m500_w5,scale=stack:zprime_tt_m1000_w10,scale=stack:zprime_tt_m3000_w30,scale=stack:zprime_tt_m7000_w70,scale=stack:zprime_tt_m500_w50,scale=stack:zprime_tt_m1000_w100,scale=stack:zprime_tt_m3000_w300,scale=stack:zprime_tt_m7000_w700,scale=stack:zprime_tt_m500_w150,scale=stack:zprime_tt_m1000_w300,scale=stack:zprime_tt_m3000_w900,scale=stack:zprime_tt_m7000_w2100,scale=stack \
        --categories 1m,1e,1m__0t,1m__1t,1e__0t,1e__1t,chi2p,chi2f \
        --cms-label simpw \
        --file-types pdf,png \
        --workflow htcondor \
        --yscale log \
        --plot-suffix for_an \
        --no-poll \
        --custom-style-config for_an \
        "${common_args[@]}"
done
