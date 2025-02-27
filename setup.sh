#!/usr/bin/env bash

setup_mtt() {
    # Runs the project setup, leading to a collection of environment variables starting with either
    #   - "CF_", for controlling behavior implemented by columnflow, or
    #   - "MTT_", for features provided by the analysis repository itself.
    # Check the setup.sh in columnflow for documentation of the "CF_" variables. The purpose of all
    # "MTT_" variables is documented below.
    #
    # The setup also handles the installation of the software stack via virtual environments, and
    # optionally an interactive setup where the user can configure certain variables.
    #
    #
    # Arguments:
    #   1. The name of the setup. "default" (which is itself the default when no name is set)
    #      triggers a setup with good defaults, avoiding all queries to the user and the writing of
    #      a custom setup file. See "interactive_setup()" for more info.
    #
    #
    # Optinally preconfigured environment variables:
    #   None yet.
    #
    #
    # Variables defined by the setup and potentially required throughout the analysis.
    #   MTT_BASE
    #       The absolute analysis base directory. Used to infer file locations relative to it.

    #
    # prepare local variables
    #

    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local orig="${PWD}"
    local setup_name="${1:-default}"
    local setup_is_default="false"
    [ "${setup_name}" = "default" ] && setup_is_default="true"

    # zsh options
    if ${shell_is_zsh}; then
        emulate -L bash
        setopt globdots
    fi


    #
    # global variables
    # (MTT = mttbar, CF = columnflow)
    #

    # start exporting variables
    export MTT_BASE="${this_dir}"
    export CF_BASE="${this_dir}/modules/columnflow"
    export CF_SETUP_NAME="${setup_name}"

    # load cf setup helpers
    CF_SKIP_SETUP="1" source "${CF_BASE}/setup.sh" "" || return "$?"

    # interactive setup
    if [ "${CF_REMOTE_JOB}" != "1" ]; then
        cf_setup_interactive_body() {
            # start querying for variables
            query CF_CERN_USER "CERN username" "$( whoami )"
            export_and_save CF_CERN_USER_FIRSTCHAR "\${CF_CERN_USER:0:1}"
            query LAW_CONFIG_FILE "Law configuration file" "\$MTT_BASE/law.cfg"
            query CF_DATA "Local data directory" "\$MTT_BASE/data" "./data"
            query CF_STORE_NAME "Relative path used in store paths (see next queries)" "mtt_store"
            query CF_STORE_LOCAL "Default local output store" "\$CF_DATA/\$CF_STORE_NAME"
            query CF_WLCG_CACHE_ROOT "Local directory for caching remote files" "" "''"
            export_and_save CF_WLCG_USE_CACHE "$( [ -z "${CF_WLCG_CACHE_ROOT}" ] && echo false || echo true )"
            export_and_save CF_WLCG_CACHE_CLEANUP "${CF_WLCG_CACHE_CLEANUP:-false}"
            query CF_SOFTWARE_BASE "Local directory for installing software" "\$CF_DATA/software"
            query CF_JOB_BASE "Local directory for storing job files" "\$CF_DATA/jobs"
            query CF_VENV_SETUP_MODE_UPDATE "Automatically update virtual envs if needed" "False"
            [ "${CF_VENV_SETUP_MODE_UPDATE}" != "True" ] && export_and_save CF_VENV_SETUP_MODE "update"
            unset CF_VENV_SETUP_MODE_UPDATE
            query CF_VOMS "Virtual-organization" "cms:/cms/dcms"
            export_and_save CF_TASK_NAMESPACE "${CF_TASK_NAMESPACE:-cf}"
            query CF_LOCAL_SCHEDULER "Use a local scheduler for law tasks" "True"
            if [ "${CF_LOCAL_SCHEDULER}" != "True" ]; then
                query CF_SCHEDULER_HOST "Address of a central scheduler for law tasks" "127.0.0.1"
                query CF_SCHEDULER_PORT "Port of a central scheduler for law tasks" "8082"
            else
                export_and_save CF_SCHEDULER_HOST "127.0.0.1"
                export_and_save CF_SCHEDULER_PORT "8082"
            fi
            query MTT_BUNDLE_CMSSW "Install and bundle CMSSW sandboxes for job submission?" "True"
            # store directory the setup was run in for the first time
            export_and_save MTT_ORIG_BASE "${MTT_BASE}"
        }
        cf_setup_interactive "${CF_SETUP_NAME}" "${MTT_BASE}/.setups/${CF_SETUP_NAME}.sh" || return "$?"
    fi

    # continue the fixed setup
    export CF_REPO_BASE="${MTT_BASE}"
    export CF_REPO_BASE_ALIAS="MTT_BASE"
    export CF_CONDA_BASE="${CF_CONDA_BASE:-${CF_SOFTWARE_BASE}/conda}"
    export CF_VENV_BASE="${CF_VENV_BASE:-${CF_SOFTWARE_BASE}/venvs}"
    export CF_CMSSW_BASE="${CF_CMSSW_BASE:-${CF_SOFTWARE_BASE}/cmssw}"
    export CF_CI_JOB="$( [ "${GITHUB_ACTIONS}" = "true" ] && echo 1 || echo 0 )"
    export CF_ORIG_PATH="${PATH}"
    export CF_ORIG_PYTHONPATH="${PYTHONPATH}"
    export CF_ORIG_PYTHON3PATH="${PYTHON3PATH}"
    export CF_ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"

    # show a warning in case no CF_REPO_BASE_ALIAS is set
    if [ -z "${CF_REPO_BASE_ALIAS}" ]; then
        cf_color yellow "the variable CF_REPO_BASE_ALIAS is unset"
        cf_color yellow "please consider setting it to the name of the variable that refers to your analysis base directory"
    fi


    #
    # minimal local software setup
    #

    cf_setup_software_stack "${CF_SETUP_NAME}" || return "$?"


    #
    # common variables
    #

    cf_setup_common_variables || return "$?"


    #
    # git hooks
    #

    cf_setup_git_hooks || return "$?"

    # ammend paths that are not covered by the central cf setup
    export PATH="${MTT_BASE}/bin:${PATH}"
    export PYTHONPATH="${MTT_BASE}:${MTT_BASE}/modules/cmsdb:${PYTHONPATH}"

    # initialze submodules
    if [ -d "${MTT_BASE}/.git" ]; then
        for m in $( ls -1q "${MTT_BASE}/modules" ); do
            cf_init_submodule "${MTT_BASE}/modules/${m}"
        done
    fi


    #
    # law setup
    #

    export LAW_HOME="${MTT_BASE}/.law"
    export LAW_CONFIG_FILE="${LAW_CONFIG_FILE:-${MTT_BASE}/law.cfg}"

    if which law &> /dev/null; then
        # source law's bash completion scipt
        source "$( law completion )" ""

        # silently index
        law index -q
    fi
}

main() {
    # Invokes the main action of this script, catches possible error codes and prints a message.

    # run the actual setup
    if setup_mtt "$@"; then
        cf_color green "m(ttbar) analysis successfully set up"
        return "0"
    else
        local code="$?"
        cf_color red "setup failed with code ${code}"
        return "${code}"
    fi
}

# entry point
if [ "${MTT_SKIP_SETUP}" != "1" ]; then
    main "$@"
fi
