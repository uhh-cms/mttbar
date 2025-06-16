#!/bin/sh
action () {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    cf_sandbox venv_columnar_dev python ${this_dir}/calc_sf.py "$@"

    # use this instead of the above for an interactive session
    #cf_sandbox venv_columnar_dev ipython -i ${this_dir}/calc_sf.py "$@"
}

action "$@"
