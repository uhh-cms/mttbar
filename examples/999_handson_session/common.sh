#
# set some common variables
# (for sourcing inside derived scripts)
#

# version tag
export my_version="test"

# signal process & dataset:
#   - Zprime with m = 3 TeV, w = 10%
export my_sig_process="zprime_tt_m3000_w300"
export my_sig_dataset="zprime_tt_m3000_w300_madgraph"

# background process & dataset
#   - standard model ttbar (semileptonic decay channel)
export my_bkg_process="tt"
export my_bkg_dataset="tt_sl_powheg"

export all_bkg_processes="tt,w_lnu,dy_lep,st,qcd,vv"
export all_processes="${all_bkg_processes},zprime_tt_m3000_w300"

# categories
export my_categories="1m"

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
