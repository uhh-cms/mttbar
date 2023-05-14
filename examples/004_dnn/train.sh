#!/usr/bin/env bash

#
# example commands for training a DNN model
#

# law args
args=(
    --version test_dnn
    --ml-model simple
    --workers 8
    --workflow htcondor
    --local-scheduler False
    $@
)

law run cf.MLTraining \
    --configs run2_2017_nano_v9 \
    "${args[@]}"
