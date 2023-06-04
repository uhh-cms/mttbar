#!/bin/sh
action () {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    source ${this_dir}/common.sh

    fname="$CF_STORE_LOCAL/analysis_mtt/cf.MergeHistograms/run2_2017_nano_v9/$my_bkg_dataset/nominal/calib__skip_jecunc/sel__default/prod__features__weights/$my_version/hist__jet1_pt.pickle"

    if [ ! -f $fname ]; then
        echo "[ERROR] File not found: $fname"
        return 1
    fi

    mtt_inspect "$fname"
}

action "$@"

