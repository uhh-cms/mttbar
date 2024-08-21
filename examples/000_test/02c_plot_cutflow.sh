#!/bin/sh
action () {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    source ${this_dir}/common.sh

    args=(
        --config $my_config
        --version $my_version
        --processes $my_process
        #--processes $all_processes
        --categories $test_categories
        --selector-steps $all_selector_steps
        --shape-norm
        --process-settings
            "tt,unstack:w_lnu,unstack:dy,unstack:st,unstack:qcd,unstack:vv,unstack"
        --hide-errors
        --skip-ratio
        --yscale "log"
        "$@"
    )

    law run cf.PlotCutflow "${args[@]}"
}

action "$@"

