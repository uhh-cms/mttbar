#!/usr/bin/env bash

# Script that installs, removes and / or sources an environment containing the HiggsAnalysis-CombinedLimit
# tool, a.k.a. `combine`. Distinctions are made depending on whether the installation is already present,
# and whether the script is called as part of a remote (law) job (CF_REMOTE_JOB=1).
#
# Six environment variables are expected to be set before this script is called:
#   CF_SANDBOX_FILE
#       The path of the file that contained the sandbox definition and that sourced _this_ script.
#       It is used to derive a hash for defining the installation directory and to set the value of
#       the LAW_SANDBOX variable.
#   CF_COMBINE_GIT_URL
#       A URL pointing to a combine Git repository.
#   CF_COMBINE_VERSION
#       The desired version of combine to set up (must be a valid revision in the combine Git repository).
#   CF_COMBINE_CMSSW_VERSION
#       The desired version of CMSSW to set up for use with combine. Only required if this script is
#       called with install_type 'cmssw' (see below).
#   CF_COMBINE_SCRAM_ARCH
#       The desired SCRAM architecture for which to set up CMSSW. Only required if this script is
#       called with install_type 'cmssw' (see below).
#   CF_COMBINE_BASE
#       The location where the environment containing combine should be installed.
#   CF_COMBINE_ENV_NAME
#       The name of the environment to prevent collisions between multiple environments using the
#       same version of combine.
#   CF_COMBINE_FLAG
#       An incremental integer value stored in the installed combine environment to detect whether it
#       needs to be updated.
#
# Arguments:
#   1. install_type
#      How to install combine. The following values are accepted:
#        - ''/standalone: combine is installed in a standalone way
#        - lcg:           combine is compiled against an LCG stack
#        - cmssw:         A CMSSW environment is created and combine is installed
#                         inside it (requires CF_COMBINE_CMSSW_VERSION to be set)
#
#   2. mode
#      The setup mode. Different values are accepted:
#        - ''/install: The combine environment is installed when not existing yet and sourced.
#        - clear:      The combine environment is removed when existing.
#        - reinstall:  The combine environment is removed first, then reinstalled and sourced.
#        - update:     The combine environment is removed first in case it is outdated, then
#                      reinstalled and sourced.
#      Please note that if the mode is empty ('') and the environment variable CF_SANDBOX_SETUP_MODE
#      is defined, its value is used instead.
#
# Note on remote jobs:
# When the CF_REMOTE_JOB variable is found to be "1" (usually set by a remote job bootstrap script),
# no mode is supported and no installation will happen but the desired combine setup is reused from a
# pre-compiled combine bundle that is fetched from a local or remote location and unpacked.

setup_combine() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local orig_dir="$( pwd )"

    # source the main setup script to access helpers
    CF_SKIP_SETUP="1" MTT_SKIP_SETUP="1" source "${this_dir}/../setup.sh" "" || return "$?"


    #
    # get and check arguments
    #

    local install_type="${1:-standalone}"
    local mode="${2:-}"

    # default mode
    if [ -z "${mode}" ]; then
        if [ ! -z "${CF_SANDBOX_SETUP_MODE}" ]; then
            mode="${CF_SANDBOX_SETUP_MODE}"
        else
            mode="install"
        fi
    fi

    # force install mode for remote jobs
    [ "${CF_REMOTE_JOB}" = "1" ] && mode="install"

    # value checks
    if [ "${mode}" != "install" ] && [ "${mode}" != "clear" ] && [ "${mode}" != "reinstall" ] && [ "${mode}" != "update" ]; then
        >&2 echo "unknown combine setup mode '${mode}'"
        return "1"
    fi
    if [ "${install_type}" != "standalone" ] && [ "${install_type}" != "lcg" ] && [ "${install_type}" != "cmssw" ]; then
        >&2 echo "unknown combine install_type '${install_type}' (valid: cmssw, lcg, standalone)"
        return "1"
    fi


    #
    # check required global variables
    #

    local sandbox_file="${CF_SANDBOX_FILE}"
    unset CF_SANDBOX_FILE
    if [ -z "${sandbox_file}" ]; then
        >&2 echo "CF_SANDBOX_FILE is not set but required by ${this_file}"
        return "10"
    fi
    if [ -z "${CF_COMBINE_VERSION}" ]; then
        >&2 echo "CF_COMBINE_VERSION is not set but required by ${this_file} to set up combine"
        return "11"
    fi
    if [ -z "${CF_COMBINE_CMSSW_VERSION}" ] && [ "${install_type}" == "cmssw" ]; then
        >&2 echo "CF_COMBINE_CMSSW_VERSION is not set but required by ${this_file} to set up CMSSW for combine"
        return "12"
    fi
    if [ -z "${CF_COMBINE_SCRAM_ARCH}" ] && [ "${install_type}" == "cmssw" ]; then
        >&2 echo "CF_COMBINE_SCRAM_ARCH is not set but required by ${this_file} to setup CMSSW for combine"
        return "13"
    fi
    if [ -z "${CF_COMBINE_BASE}" ]; then
        >&2 echo "CF_COMBINE_BASE is not set but required by ${this_file} to set up combine"
        return "14"
    fi
    if [ -z "${CF_COMBINE_ENV_NAME}" ]; then
        >&2 echo "CF_COMBINE_ENV_NAME is not set but required by ${this_file} to set up combine"
        return "15"
    fi
    if [ -z "${CF_COMBINE_FLAG}" ]; then
        >&2 echo "CF_COMBINE_FLAG is not set but required by ${this_file} to set up combine"
        return "16"
    fi


    #
    # define variables
    #

    local install_hash="$( cf_sandbox_file_hash "${sandbox_file}" )"
    local combine_env_name_hashed="${CF_COMBINE_ENV_NAME}_${install_hash}"
    local install_base="${CF_COMBINE_BASE}/${combine_env_name_hashed}"
    local install_path="${install_base}/${CF_COMBINE_VERSION}"
    local install_path_repr="\$CF_COMBINE_BASE/${combine_env_name_hashed}/${CF_COMBINE_VERSION}"
    local pending_flag_file="${CF_COMBINE_BASE}/pending_${combine_env_name_hashed}_${CF_COMBINE_VERSION}"

    export CF_SANDBOX_FLAG_FILE="${install_path}/cf_flag"


    #
    # start the setup
    #

    # ensure CF_COMBINE_BASE exists
    mkdir -p "${CF_COMBINE_BASE}"

    if [ "${CF_REMOTE_JOB}" != "1" ]; then
        # optionally remove the current installation
        if [ "${mode}" = "clear" ] || [ "${mode}" = "reinstall" ]; then
            echo "removing current installation at ${install_path} (mode '${mode}')"
            rm -rf "${install_path}"

            # optionally stop here
            [ "${mode}" = "clear" ] && return "0"
        fi

        # in local environments, install from scratch
        if [ ! -d "${install_path}" ]; then
            # from here onwards, files and directories could be created and in order to prevent race
            # conditions from multiple processes, guard the setup with the pending_flag_file and
            # sleep for a random amount of seconds between 0 and 10 to further reduce the chance of
            # simultaneously starting processes getting here at the same time
            local sleep_counter="0"
            sleep "$( python3 -c 'import random;print(random.random() * 10)')"
            # when the file is older than 30 minutes, consider it a dangling leftover from a
            # previously failed installation attempt and delete it.
            if [ -f "${pending_flag_file}" ]; then
                local flag_file_age="$(( $( date +%s ) - $( date +%s -r "${pending_flag_file}" )))"
                [ "${flag_file_age}" -ge "1800" ] && rm -f "${pending_flag_file}"
            fi
            # start the sleep loop
            while [ -f "${pending_flag_file}" ]; do
                # wait at most 20 minutes
                sleep_counter="$(( $sleep_counter + 1 ))"
                if [ "${sleep_counter}" -ge 120 ]; then
                    >&2 echo "combine ${CF_COMBINE_VERSION} is being set up in different process, but number of sleeps exceeded"
                    return "20"
                fi
                cf_color yellow "combine ${CF_COMBINE_VERSION} already being set up in different process, sleep ${sleep_counter} / 120"
                sleep 10
            done
        fi

        # create the pending_flag to express that the venv state might be changing
        touch "${pending_flag_file}"
        clear_pending() {
            rm -f "${pending_flag_file}"
        }

        # checks to be performed if the venv already exists
        if [ -d "${install_path}" ]; then
            # get the current version
            local current_version="$( cat "${CF_SANDBOX_FLAG_FILE}" | grep -Po "version \K\d+.*" )"
            if [ -z "${current_version}" ]; then
                >&2 echo "the flag file ${CF_SANDBOX_FLAG_FILE} does not contain a valid version"
                clear_pending
                return "21"
            fi

            if [ "${current_version}" != "${CF_COMBINE_FLAG}" ]; then
                if [ "${mode}" = "update" ]; then
                    # remove the venv in case an update is requested
                    echo "removing current installation at ${install_path_repr}"
                    echo "(mode '${mode}', installed version ${current_version}, requested version ${CF_COMBINE_FLAG})"
                    rm -rf "${install_path}"
                else
                    >&2 echo
                    >&2 echo "WARNING: outdated combine environment '${combine_env_name_hashed}'"
                    >&2 echo "WARNING: (installed version ${current_version}, requested version ${CF_COMBINE_FLAG})"
                    >&2 echo "WARNING: located at ${install_path_repr}"
                    >&2 echo "WARNING: please consider updating it by adding 'update' to the source command"
                    >&2 echo "WARNING: or by setting the environment variable 'CF_SANDBOX_SETUP_MODE=update'"
                    >&2 echo
                fi
            fi
        fi

        # install when missing
        # following: https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit
        if [ ! -d "${install_path}" ]; then
            echo
            cf_color cyan "installing combine ${CF_COMBINE_VERSION} to ${install_base} (install type: $install_type)"

            if [ "$install_type" == "cmssw" ]; then
                (
                    mkdir -p "${install_path}"
                    cd "${install_path}"

                    # set up and source a CMSSW work area
                    source "/cvmfs/cms.cern.ch/cmsset_default.sh" ""
                    export SCRAM_ARCH="${CF_COMBINE_SCRAM_ARCH}"
                    scramv1 project CMSSW "${CF_COMBINE_CMSSW_VERSION}"
                    cd "${CF_COMBINE_CMSSW_VERSION}/src"
                    eval "$( scramv1 runtime -sh )"

                    # clone the combine repo
                    git clone "$CF_COMBINE_GIT_URL" HiggsAnalysis/CombinedLimit || {
                        >&2 echo "failed to clone combine git repository from URL ${CF_COMBINE_GIT_URL}"
                        clear_pending
                        return "3001"
                    }

                    # check out the specified combine version
                    cd HiggsAnalysis/CombinedLimit
                    git checkout "${CF_COMBINE_VERSION}" || {
                        >&2 echo "failed to check out revision ${CF_COMBINE_VERSION} from git repository"
                        clear_pending
                        return "3002"
                    }

                    # # clone the combine harvester repo
                    # cd ${CMSSW_BASE}/src
                    # git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester || {
                    #     >&2 echo "failed to clone CombineHarvester git repository from URL ${CF_COMBINE_HARVESTER_GIT_URL}"
                    #     clear_pending
                    #     return "3003"
                    # }

                    # # check out the specified combine harvester version
                    # cd CombineHarvester
                    # git checkout "${CF_COMBINE_HARVESTER_VERSION}" || {
                    #     >&2 echo "failed to check out revision ${CF_COMBINE_HARVESTER_VERSION} from git repository"
                    #     clear_pending
                    #     return "3004"
                    # }

                    # compile
                    cd ${CMSSW_BASE}
                    scram b -j4

                    # write the flag into a file
                    echo "version ${CF_COMBINE_FLAG}" > "${CF_SANDBOX_FLAG_FILE}"
                    rm -f "${pending_flag_file}"
                )
            else
                (
                    mkdir -p "${install_path}"
                    cd "${install_path}"

                    # clone the combine repo
                    git clone "$CF_COMBINE_GIT_URL" HiggsAnalysis/CombinedLimit || {
                        >&2 echo "failed to clone combine git repository from URL ${CF_COMBINE_GIT_URL}"
                        clear_pending
                        return "3001"
                    }

                    # check out the specified combine version
                    cd HiggsAnalysis/CombinedLimit
                    git checkout "${CF_COMBINE_VERSION}" || {
                        >&2 echo "failed to check out revision ${CF_COMBINE_VERSION} from git repository"
                        clear_pending
                        return "3002"
                    }

                    # source the combine environment and compile
                    if [ "$install_type" == "standalone" ]; then
                        . env_standalone.sh
                        make -j 4
                    elif [ "$install_type" == "lcg" ]; then
                        . env_lcg.sh
                        make LCG=1 -j 4
                    elif [ "$install_type" == "cmssw" ]; then
                        mkdir -p "${install_path}"
                        cd "${install_path}"
                        source "/cvmfs/cms.cern.ch/cmsset_default.sh" ""
                        export SCRAM_ARCH="${CF_COMBINE_SCRAM_ARCH}"
                        scramv1 project CMSSW "${CF_COMBINE_CMSSW_VERSION}"
                        cd "${CF_COMBINE_CMSSW_VERSION}/src"
                        eval "$( scramv1 runtime -sh )"
                        scram b
                    else
                        >&2 echo "invalid compile mode '${install_type}'"
                        clear_pending
                        return "3003"
                    fi

                    # write the flag into a file
                    echo "version ${CF_COMBINE_FLAG}" > "${CF_SANDBOX_FLAG_FILE}"
                    rm -f "${pending_flag_file}"
                )
            fi
            local ret="$?"
            [ "${ret}" != "0" ] && clear_pending && return "${ret}"


            # custom install hook and second install pass
            (
                local src_dir
                if [ "$install_type" == "cmssw" ]; then
                    src_dir="${install_path}/${CF_COMBINE_CMSSW_VERSION}/src"
                else
                    src_dir="${install_path}"
                fi
                cd "${src_dir}"
                source "/cvmfs/cms.cern.ch/cmsset_default.sh" ""
                eval "$( scramv1 runtime -sh )"

                if command -v cf_cmssw_custom_install &> /dev/null; then
                    echo -e "\nrunning cf_cmssw_custom_install"
                    cf_cmssw_custom_install &&
                    source "/cvmfs/cms.cern.ch/cmsset_default.sh" "" &&
                    cd "${src_dir}" &&
                    scram b
                elif [ ! -z "${cf_cmssw_custom_install}" ] && [ -f "${cf_cmssw_custom_install}" ]; then
                    echo -e "\nsourcing cf_cmssw_custom_install file"
                    source "${cf_cmssw_custom_install}" "" &&
                    source "/cvmfs/cms.cern.ch/cmsset_default.sh" "" &&
                    cd "${src_dir}" &&
                    scram b
                fi
            )
        fi

        # remove the pending_flag
        clear_pending
    fi

    # handle remote job environments
    if [ "${CF_REMOTE_JOB}" = "1" ]; then
        # fetch, unpack and set up the bundle
        if [ ! -d "${install_path}" ]; then
            # fetch the bundle and unpack it
            echo "looking for combine sandbox bundle for ${CF_COMBINE_ENV_NAME}"
            local sandbox_names=(${CF_JOB_COMBINE_SANDBOX_NAMES})
            local sandbox_uris=(${CF_JOB_COMBINE_SANDBOX_URIS})
            local sandbox_patterns=(${CF_JOB_COMBINE_SANDBOX_PATTERNS})
            local found_sandbox="false"
            for (( i=0; i<${#sandbox_names[@]}; i+=1 )); do
                if [ "${sandbox_names[i]}" = "${CF_COMBINE_ENV_NAME}" ]; then
                    echo "found bundle ${CF_COMBINE_ENV_NAME}, index ${i}, pattern ${sandbox_patterns[i]}, uri ${sandbox_uris[i]}"
                    (
                        mkdir -p "${install_base}" &&
                        cd "${install_base}" &&
                        law_wlcg_get_file "${sandbox_uris[i]}" "${sandbox_patterns[i]}" "combine.tgz"
                    ) || return "$?"
                    found_sandbox="true"
                    break
                fi
            done
            if ! ${found_sandbox}; then
                >&2 echo "combine sandbox ${CF_COMBINE_ENV_NAME} not found in job configuration, stopping"
                return "22"
            fi

            # unpack the bundled combine repo
            if [ "$install_type" == "cmssw" ]; then
                (
                    echo "unpacking bundle to ${install_path}"
                    cd "${install_base}" &&
                    source "/cvmfs/cms.cern.ch/cmsset_default.sh" "" &&
                    export SCRAM_ARCH="${CF_SCRAM_ARCH}" &&
                    scramv1 project CMSSW "${CF_CMSSW_VERSION}" &&
                    cd "${CF_CMSSW_VERSION}" &&
                    tar -xzf "../combine.tgz" &&
                    rm "../combine.tgz" &&
                    cd "src" &&
                    eval "$( scramv1 runtime -sh )" &&
                    scram b python
                ) || return "$?"
            else
                (
                    echo "unpacking combine bundle to ${install_path}"
                    cd "${install_base}" &&
                    tar -xzf "../combine.tgz" &&
                    rm "../combine.tgz"
                ) || return "$?"
            fi

            # write the flag into a file
            echo "version ${CF_COMBINE_FLAG}" > "${CF_SANDBOX_FLAG_FILE}"
        fi
    fi

    # at this point, the src path must exist
    if [ "$install_type" == "cmssw" ] && [ ! -d "${install_path}/${CF_COMBINE_CMSSW_VERSION}/src" ]; then
        >&2 echo "src directory not found in combine CMSSW installation at ${install_path}"
        return "30"
    fi

    # source the combine setup
    if [ "$install_type" == "cmssw" ]; then
        source "/cvmfs/cms.cern.ch/cmsset_default.sh" "" || return "$?"
        export SCRAM_ARCH="${CF_COMBINE_SCRAM_ARCH}"
        export CMSSW_VERSION="${CF_COMBINE_CMSSW_VERSION}"
        cd "${install_path}/${CF_COMBINE_CMSSW_VERSION}/src"
        eval "$( scramv1 runtime -sh )"
        export PYTHONPATH="${PYTHON3PATH}:${PYTHONPATH}"
    elif [ "$install_type" == "standalone" ]; then
        cd "${install_path}/HiggsAnalysis/CombinedLimit"
        . env_standalone.sh
    elif [ "$install_type" == "lcg" ]; then
        cd "${install_path}/HiggsAnalysis/CombinedLimit"
        . env_lcg.sh
    fi
    # custom setup
    if command -v cf_cmssw_custom_setup &> /dev/null; then
        cf_cmssw_custom_setup || return "$?"
    elif [ ! -z "${cf_cmssw_custom_setup}" ] && [ -f "${cf_cmssw_custom_setup}" ]; then
        source "${cf_cmssw_custom_setup}" "" || return "$?"
    fi

    cd "${orig_dir}"

    export PYTHONPATH="${CF_PERSISTENT_PATH}:$( echo ${PYTHONPATH} | sed "s|${CF_CONDA_PYTHONPATH}||g" )"
    export PATH="${CF_PERSISTENT_PATH}:${PATH}"

    # mark this as a bash sandbox for law
    export LAW_SANDBOX="bash::$( cf_sandbox_file_hash -p "${sandbox_file}" )"

    return "0"
}
setup_combine "$@"
