#
# set some common variables
# (for sourcing inside derived scripts)
#

# -- version tag
export my_config="run2_2017_nano_v9_limited"
export my_version="test"

# -- background process & dataset
# standard model ttbar (semileptonic decay channel)
#export my_process="tt"
#export my_dataset="tt_sl_powheg"

# single-top t-channel with top quark
#export my_process="st"
#export my_dataset="st_tchannel_t_4f_powheg"

# diboson
#export my_process="vv"
#export my_dataset="ww_pythia"

# bsm zprime decaying to ttbar
export my_process="zprime_tt_m3000_w300"
export my_dataset="${my_process}_madgraph"

export all_processes="tt,st,dy,w_lnu,vv,qcd,zprime_tt_m3000_w300"
export all_selector_steps="Lepton,MET,Jet,BJet,JetLepton2DCut,AllHadronicVeto,DileptonVeto,METFilters"

export all_variables="jet1_pt"
export all_variables_ttbar_reco="${all_variables},chi2_lt100,ttbar_mass"

# categories
export all_categories=$(echo 1{e,m} 1{e,m}__{1,0}t 1{e,m}__{1,0}t__chi2{pass,fail} | tr " " ",")
export test_categories="1e,1m"

# print or run commands depending on env var PRINT
_law=$(type -fp law)
law () {
    if [ -z $PRINT ]; then
        ${_law} "$@"
    else
        echo law "$@"
    fi
}

_mtt_inspect=$(type -fp mtt_inspect)
mtt_inspect () {
    if [ -z $PRINT ]; then
        ${_mtt_inspect} "$@"
    else
        echo mtt_inspect "$@"
    fi
}
