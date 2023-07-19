#!/bin/sh
action () {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    source ${this_dir}/common.sh

    args=(
        --version $my_version
        --processes $my_process
        --datasets $my_dataset
        --variables $all_variables
        --categories $test_categories
        --skip-ratio
        "$@"
    )

    law run cf.PlotVariables1D "${args[@]}"
}

action "$@"

