#!/usr/bin/env bash

# common args
common_args=(

    --version 260804_v8
    --config run3_mtt_2025_nano_v15_medium_limited_new
    # --configs run3_mtt_2022_preEE_nano_v12_new,run3_mtt_2022_postEE_nano_v12_new
    --analysis mtt.config.run3.analysis_mtt.analysis_mtt_new
    --local-scheduler False
    # --remove-output 0,a,y
    --cf.BundleRepo-custom-checksum v_0813_1620
    --workers 1
    --poll-interval 5
    --tasks-per-job 5
    # --workflow htcsondor
    # --pilot
    # --workflow
        # local
        # htcondor
    
    # --cf.CalibrateEvents-htcondor-memory 7GB
    # --cf.SelectEvents-htcondor-memory 7GB
    # --cf.ReduceEvents-htcondor-memory 4GB
    # --cf.ProduceColumns-htcondor-memory 7GB
    # --cf.MergeHistograms-htcondor-memory 7GB
    $@
)

plot_args=(
### ___ set path parameters ___ ###
    --selector
        default
        # default_without_2d_selection

    --producers
        add_prod_cats,ttbar,features,weights
        # add_prod_cats,ttbar,features,weights,ml_inputs
        # features,weights

    # --ml-models simple

    --processes
        tt,w_lnu,st,dy,vv,qcd,data
        # tt,w_lnu_1j,w_lnu_2j,w_lnu_3j,w_lnu_4j,dy,st,qcd,vv,data,zprime_tt_m7000_w70
        # vv,qcd,st,dy,w_lnu,tt
        # zprime_default,backgrounds
        # bkg,zprime_tt_m1000_w100
        # tt,w_lnu,dy_lep,st,qcd,vv,zprime_tt_m3000_w300
        # tt,zprime_tt_m500_w50,zprime_tt_m1000_w100,zprime_tt_m3000_w300
        # tt
        # $proc
        # w_lnu_1j,w_lnu_2j,w_lnu_3j,w_lnu_4j

#     --datasets
#         tt_fh_powheg

    --hist-producer all_weights

    --process-settings
        # tt,unstack:w_lnu,unstack:dy,unstack:st,unstack:qcd,unstack:vv,unstack:data,unstack:zprime_tt_m7000_w70,unstack
        "zprime_tt_m500_w50,color1=#000000,scale=2000:zprime_tt_m1000_w100,color1=#cccccc,scale=2000:zprime_tt_m3000_w300,color1=#666666,scale=2000:zprime_tt_m7000_w70,color1=#666666,scale=2000"
#         "tt,unstack:zprime_tt_m500_w50,color1=#000000,scale=2000:zprime_tt_m1000_w100,color1=#cccccc,scale=2000:zprime_tt_m3000_w300,color1=#666666,scale=2000"

### ___ set plot parameters ___ ###
    --categories
        # 1m,1e
        1m__0j,1m__1j,1m__2j,1m__3j
        # 1m,1e,1m__0t,1m__1t,1e__0t,1e__1t,1m__0t__chi2pass,1m__1t__chi2pass,1e__0t__chi2pass,1e__1t__chi2pass,1m__chi2pass,1e__chi2pass
#         1m,1m__dnn_tt,1m__dnn_st,1m__dnn_dy_lep,1m__dnn_w_lnu

    # --variables
    #     jet_lep_pt_rel_zoom,jet_lep_delta_r_zoom
    #     muon_pt,electron_pt
#         chi2_lt100
        # jet_lep_pt_rel_zoom-jet_lep_delta_r_zoom
#         "simple.score_*"
#         'mli_n_*','mli_AN_*'

    --yscale log
    # --skip-ratio
    # --shape-norm
    --cms-label pw
    --file-types pdf,png
    # --plot-suffix
        # norm__lin
            # log_flipped
#           resolved
        #   wo_2d
#     --hide-errors
    # --remove-output 0,a,y

    "${common_args[@]}"
)

# yield_table_args=(
#     --selector default
#     --producers add_prod_cats,ttbar,features,weights
#     # --processes zprime_tt_m7000_w70
#     --processes tt,w_lnu,dy,st,qcd,vv,data,zprime_tt_m7000_w70
#     --table-format latex_raw
#     --workflow htcondor
#     --hist-producer all_weights
#     "${common_args[@]}"
# )

for ds in st; do
    echo "Reducing $ds"
    echo claw run cf.ReduceEventsWrapper \
    --cf.ReduceEvents-workflow local \
    --cf.ReduceEvents-pilot \
    --cf.ReduceEvents-htcondor-memory 7GB \
    --datasets $ds \
    "${common_args[@]}"
done


for prod in ttbar features ml_inputs; do
    echo claw run cf.ProduceColumnsWrapper \
        --datasets st \
        --cf.ProduceColumns-producer $prod \
        --cf.ProduceColumns-workflow htcondor \
        --cf.ProduceColumns-selector default \
        --cf.ProduceColumns-pilot \
        --cf.ProduceColumns-htcondor-memory 5GB \
        --cf.MergeReducedEvents-workflow local \
        --cf.MergeSelectionStats-workflow local \
        "${common_args[@]}"
done

# for prod in ttbar ; do
#     claw run cf.ProduceColumnsWrapper \
#         --datasets bkg,data,zprime_tt_m7000_w70 \
#         --cf.ProduceColumns-producer $prod \
#         --cf.ProduceColumns-workflow local \
#         --cf.ProduceColumns-selector default \
#         --cf.ProduceColumns-pilot \
#         "${common_args[@]}"
# done

        
for var in 'n_bjet'; do
    echo "plotting $var"
    echo law run cf.PlotVariables1D \
        --variables $var \
        --datasets bkg,data \
        "${plot_args[@]}"
done

    # --cf.MergeHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar} \
# law run cf.PlotVariables1D \
#     --variables AN_v12_mli_fatjet_tau32_1,AN_v12_mli_fatjet_tau32_2,AN_v12_mli_fatjet_tau32_3,AN_v12_mli_fatjet_tau21_1,AN_v12_mli_fatjet_tau21_2,AN_v12_mli_fatjet_tau21_3 \
#     "${plot_args[@]}"
    # --categories 1m__1t__chi2pass__dnn_tt,1m__0t__chi2pass__dnn_tt,1e__1t__chi2pass__dnn_tt,1e__0t__chi2pass__dnn_tt,1m__1t__chi2pass__dnn_st,1m__0t__chi2pass__dnn_st,1e__1t__chi2pass__dnn_st,1e__0t__chi2pass__dnn_st,1m__1t__chi2pass__dnn_other,1m__0t__chi2pass__dnn_other,1e__1t__chi2pass__dnn_other,1e__0t__chi2pass__dnn_other \
    # --processes qcd,st,dy,w_lnu,tt \


# echo law run cf.PlotVariables1D \
#     --selector default \
#     --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_v1_AN_v12 \
#     --ml-models v1_AN_v12 \
#     --hist-producer all_weights \
#     --variables ttbar_mass_ext,mlscore.tt,mlscore.st,mlscore.other \
#     --categories v12_simplified \
#     --processes qcd,st,dy,w_lnu,tt,zprime_tt_m7000_w70 \
#     --yscale log \
#     --skip-ratio \
#     --plot-suffix log \
#     --workflow local \
#     --file-types pdf,png \
#     --cms-label simpw \
#     "${common_args[@]}"

# law run mtt.PlotMLResultsSingleFold \
#     --ml-model v1_AN_v12 \
#     --fold 0 \
#     --workflow htcondor \
#     "${common_args[@]}"

# for proc in tt bkg data; do
#     echo "Plotting $proc"
#     law run cf.PlotVariables2D \
#         --processes $proc \
#         --variables mli_AN_jet_energy_1-mli_AN_jet_pt_1,mli_AN_jet_energy_1-mli_AN_jet_eta_1,mli_AN_jet_pt_1-mli_AN_jet_eta_1,mli_AN_jet_energy_1-mli_AN_jet_mass_1,mli_AN_jet_energy_1-mli_AN_jet_phi_1,mli_AN_jet_energy_2-mli_AN_jet_pt_2,mli_AN_jet_energy_2-mli_AN_jet_eta_2,mli_AN_jet_pt_2-mli_AN_jet_eta_2,mli_AN_jet_energy_2-mli_AN_jet_mass_2,mli_AN_jet_energy_2-mli_AN_jet_phi_2,mli_AN_jet_energy_3-mli_AN_jet_pt_3,mli_AN_jet_energy_3-mli_AN_jet_eta_3,mli_AN_jet_pt_3-mli_AN_jet_eta_3,mli_AN_jet_energy_3-mli_AN_jet_mass_3,mli_AN_jet_energy_3-mli_AN_jet_phi_3,mli_AN_jet_energy_4-mli_AN_jet_pt_4,mli_AN_jet_energy_4-mli_AN_jet_eta_4,mli_AN_jet_pt_4-mli_AN_jet_eta_4,mli_AN_jet_energy_4-mli_AN_jet_mass_4,mli_AN_jet_energy_4-mli_AN_jet_phi_4,mli_AN_jet_energy_5-mli_AN_jet_pt_5,mli_AN_jet_energy_5-mli_AN_jet_eta_5,mli_AN_jet_pt_5-mli_AN_jet_eta_5,mli_AN_jet_energy_5-mli_AN_jet_mass_5,mli_AN_jet_energy_5-mli_AN_jet_phi_5 \
#         "${plot_args[@]}"
# done

# for cats in 1m,1m__0t,1m__1t,1m__0t__chi2pass,1m__1t__chi2pass,1m__chi2pass,1e,1e__0t,1e__1t,1e__0t__chi2pass,1e__1t__chi2pass,1e__chi2pass 1m,1e 1m__0t,1e__0t 1m__1t,1e__1t 1m__0t__chi2pass,1e__0t__chi2pass 1m__1t__chi2pass,1e__1t__chi2pass 1m__chi2pass,1e__chi2pass; do
# # for cats in 1m,1e 1m__0t,1e__0t 1m__1t,1e__1t 1m__0t__chi2pass,1e__0t__chi2pass 1m__1t__chi2pass,1e__1t__chi2pass 1m__chi2pass,1e__chi2pass; do
#     echo "Creating yield table for categories $cats"
#     echo claw run cf.CreateYieldTable \
#         --categories $cats \
#         --remove-output 0,a,y \
#         "${yield_table_args[@]}"
# done

# for proc in qcd tt zprime_tt_m1000_w100 zprime_tt_m3000_w300 zprime_tt_m500_w50; do
# for proc in qcd tt; do
#     echo "Plotting $proc"
#     # law run cf.PlotVariables2D \
#     #     --processes $proc \
#     #     --variables jet_lep_pt_rel_zoom-jet_lep_delta_r_zoom \
#     #     "${plot_args[@]}"
# done

# for mass in 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2500 3000 3500 4000 4500 5000 6000 7000 8000 9000; do
#     law run cf.CreateDatacards \
#     --inference-model simple_zprime_tt_m${mass}_w$((mass/10)) \
#     "${common_args[@]}"
# done
# --cf.CreateCutflowHistograms-{workflow=htcondor,pilot,no-poll,remote-claw-sandbox=venv_columnar,version=test_v1} \
# --cf.CreateCutflowHistograms-version test_v1 \

# for proc in qcd,dy,vv,w_lnu,st,tt,zprime_tt_m7000_w70; do
#     law run cf.PlotCutflow \
#     --yscale log \
#     --processes $proc \
#     --process-settings tt,unstack:st,unstack:dy,unstack:w_lnu,unstack:qcd,unstack:vv,unstack:zprime_tt_m7000_w70,unstack:zprime_tt_m4000_w40,unstack:zprime_tt_m500_w5,unstack \
#     --categories incl \
#     --variable mc_weight \
#     --file-type png,pdf \
#     --selector-steps Lepton,MET,Jet,BJet,JetLepton2DCut,AllHadronicVeto,DileptonVeto,METFilters \
#     --cms-label simpw \
#     --workflow local \
#     --shape-norm \
#     --remove-output 0,a,y \
#     "${common_args[@]}"
# done
for sig_group in zprime_tt_m3000_w300; do
# for sig_group in sig sig_narrow sig_wide sig_low_mass sig_medium_mass sig_high_mass sig_very_high_mass; do
    law run cf.PlotCutflow \
        --yscale log \
        --processes tt,qcd,st,w_lnu,dy,$sig_group \
        --process-settings tt,unstack:st,unstack:dy,unstack:w_lnu,unstack:qcd,unstack:vv,unstack:$sig_group \
        --selector-steps Lepton,MET,Jet,BJet,JetLepton2DCut,AllHadronicVeto,DileptonVeto,cleanup \
        --categories 1m,1e,incl \
        --variable mc_weight \
        --file-type png,pdf \
        --cms-label simpw \
        --workflow local \
        --shape-norm \
        --custom-style-config "cutflow" \
        "${common_args[@]}"
done

# law run cf.PlotCutflowVariables1D \
#     --selector default \
#     --processes tt,w_lnu_1j,w_lnu_2j,w_lnu_3j,w_lnu_4j \
#     --categories incl,1m,1e \
#     --variables cf_muon_pt,cf_electron_pt,cf_muon_eta,cf_electron_eta \
#     --file-type png,pdf \
#     --yscale log \
#     --cms-label simpw \
#     --selector-steps Lepton,Jet,BJet,MET \
#     --per-plot steps \
#     --cf.ReduceEvents-version test_v1 \
#     --cf.SelectEvents-version test_v1 \
#     --cf.CalibrateEvents-version test_v1 \
#     "${common_args[@]}"

# for inf_model in an_v12_simplified__m500_w5 an_v12_simplified__m4000_w40 an_v12_simplified__m4500_w45 an_v12_simplified__m7000_w70; do
#     echo "Creating datacard for model $inf_model"
#     law run cf.CreateDatacards \
#         --inference-model $inf_model \
#         --hist-producer all_weights \
#         --selector default \
#         --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_v3_AN_v12 \
#         --workflow local \
#         --remove-output 0,a,y \
#         "${common_args[@]}"
# done

yield_table_args=(
    --selector ttbar_res_sel
    --producers add_prod_cats,ttbar,features,weights,ml_inputs,add_ml_cats_v4_2425
    --processes tt,w_lnu,dy,st,qcd,vv,sig,sig_narrow,sig_wide
    # --table-format latex_raw
    --workflow local
    --hist-producer no_trigger_weights
    "${common_args[@]}"
)

# for cats in 1m,1m__0t,1m__1t,1m__0t__chi2pass,1m__1t__chi2pass,1m__chi2pass,1e,1e__0t,1e__1t,1e__0t__chi2pass,1e__1t__chi2pass,1e__chi2pass 1m,1e 1m__0t,1e__0t 1m__1t,1e__1t 1m__0t__chi2pass,1e__0t__chi2pass 1m__1t__chi2pass,1e__1t__chi2pass 1m__chi2pass,1e__chi2pass; do
for cats in v12_simplified; do
    echo "Creating yield table for categories $cats"
    claw run cf.CreateYieldTable \
        --categories $cats \
        "${yield_table_args[@]}"
done

for proc in data zprime_tt_m1000_w100; do
    for var in cf_n_muon cf_n_electron cf_n_veto_electron cf_n_veto_muon; do
        echo "Plotting cutflow variable $var for process $proc"
        echo claw run cf.PlotCutflowVariables1D \
            --selector ttbar_res_sel \
            --categories 1m,1e,incl \
            --variables $var \
            --per-plot steps \
            --processes $proc \
            --process-settings tt,unstack:st,unstack:dy,unstack:w_lnu,unstack:qcd,unstack:vv,unstack \
            --selector-steps Lepton,MET,DileptonVeto,all \
            --file-type png,pdf \
            --cms-label simpw \
            --yscale log \
            --workflow local \
            "${common_args[@]}"
    done

    for var in  cf_n_jet cf_n_bjet cf_n_lightjet cf_n_toptag cf_jet1_pt; do
        echo "Plotting cutflow variable $var for process $proc"
        echo claw run cf.PlotCutflowVariables1D \
            --selector ttbar_res_sel \
            --categories 1m,1e,incl \
            --variables $var \
            --per-plot steps \
            --processes $proc \
            --process-settings tt,unstack:st,unstack:dy,unstack:w_lnu,unstack:qcd,unstack:vv,unstack \
            --selector-steps MET,Jet,BJet,JetLepton2DCut,AllHadronicVeto,all \
            --file-type png,pdf \
            --cms-label simpw \
            --yscale log \
            --workflow local \
            --remove-output 0,a,y \
            "${common_args[@]}"
    done
done