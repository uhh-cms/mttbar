#!/usr/bin/env bash

# Script that sets up a standalone combine environment in $CF_COMBINE_BASE.
# For more info on functionality and parameters, see the generic setup script _setup_combine.sh.

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set variables and source the combine setup
    export CF_SANDBOX_FILE="${CF_SANDBOX_FILE:-${this_file}}"
    export CF_COMBINE_GIT_URL="${CF_COMBINE_GIT_URL:-https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git}"
    # export CF_COMBINE_HARVESTER_GIT_URL="${CF_COMBINE_HARVESTER_GIT_URL:-https://github.com/cms-analysis/CombineHarvester.git}"
    export CF_COMBINE_SCRAM_ARCH="el9_amd64_gcc13"
    export CF_COMBINE_CMSSW_VERSION="CMSSW_16_0_0"  # from combine docu 03.09.26
    # export CF_COMBINE_HARVESTER_VERSION="${CF_COMBINE_HARVESTER_VERSION:-v3.0.0-pre1}"
    export CF_COMBINE_VERSION="${CF_COMBINE_VERSION:-v11.0.0}"  # from combine docu 03.09.26
    export CF_COMBINE_ENV_NAME="$( basename "${this_file%.sh}" )"
    export CF_COMBINE_FLAG="1"  # increment when content changed# define custom install and setup functions

    cf_cmssw_custom_install() {
        PYTHONPATH="" CF_VENV_BASE="${CMSSW_BASE}/venvs" cf_create_venv combine_cmssw &&
        source "${CMSSW_BASE}/venvs/combine_cmssw/bin/activate" "" &&
        PYTHONPATH="" "${CMSSW_BASE}/venvs/combine_cmssw/bin/python" -m pip install -r "${this_dir}/combine_cmssw.txt" &&
        CF_VENV_BASE="${CMSSW_BASE}/venvs" cf_make_venv_relocatable combine_cmssw &&
        mkdir -p "${CMSSW_BASE}/python/HiggsAnalysis/CombinedLimit" &&
        for f in "${CMSSW_BASE}"/src/HiggsAnalysis/CombinedLimit/python/*.py; do
            ln -sf "$f" "${CMSSW_BASE}/python/HiggsAnalysis/CombinedLimit/$(basename "$f")"
        done
    }
    cf_cmssw_custom_setup() {
        source "${CMSSW_BASE}/venvs/combine_cmssw/bin/activate" "" &&
        local combine_site_packages="$( python -c 'import site; print(site.getsitepackages()[0])' )" &&
        export PYTHONPATH="${combine_site_packages}:${PYTHONPATH}" &&
        export PYTHONPATH="${PYTHONPATH}:$( python -c "import os; print(os.path.normpath('$( root-config --libdir )'))" )"
    }
    # cf_cmssw_custom_setup() {
    #     source "${CMSSW_BASE}/venvs/combine_cmssw/bin/activate" "" &&
    #     export PYTHONPATH="${PYTHONPATH}:$( python -c "import os; print(os.path.normpath('$( root-config --libdir )'))" )"
    # }

    source "${this_dir}/_setup_combine.sh" cmssw "$@"
}
action "$@"
