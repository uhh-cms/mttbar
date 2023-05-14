#!/usr/bin/env bash

#
# example commands for evaluating a DNN model
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

law run cf.MLEvaluation \
    --producers default,ml_inputs \
    "${args[@]}"
