# coding: utf-8

"""
ML model using the MLClassifierBase and Mixins
Taken from hbw analysis (dl derived ml model).
"""

from __future__ import annotations

from columnflow.types import Union

import law

from columnflow.util import maybe_import, DotDict

from mtt.ml.base import MLClassifierBase
from mtt.ml.mixins import DenseModelMixin, ModelFitMixin


np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


class DenseClassifier(DenseModelMixin, ModelFitMixin, MLClassifierBase):
    _default__processes: tuple = (
        "tt",
        "st",
        "dy",
        "w_lnu",
    )
    train_nodes: dict = {
        "tt": {"ml_id": 0},
        "st": {"ml_id": 1},
        "dy": {"ml_id": 2},
        "w_lnu": {"ml_id": 3},
    }
    _default__class_factors: dict = {
        "tt": 1,
        "st": 1,
        "dy": 1,
        "w_lnu": 1,
    }

    input_features = (
        "n_jet",
        "n_fatjet",
    ) + tuple([
        f"jet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "mass", "btagUParTAK4B")
        for i in range(5)
    ]) + tuple([
        f"fatjet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ]) + tuple([
        f"lepton_{var}"
        for var in ("energy", "pt", "eta", "phi")
    ]) + tuple([
        f"met_{var}"
        for var in ("pt", "phi")
    ])

    preparation_producer_name = "prepml"

    folds: int = 5
    negative_weights: str = "ignore"

    # overwriting DenseModelMixin parameters
    # same as simple dnn from ml_mtt_backup
    _default__activation: str = "relu"
    _default__layers: tuple = (512, 512)
    _default__dropout: float = 0.5
    _default__learningrate: float = 0.00050

    # overwriting ModelFitMixin parameters
    _default__callbacks: set = {
        "backup", "checkpoint", "reduce_lr",
        # "early_stopping",
    }
    remove_backup: bool = True
    _default__reduce_lr_factor: float = 0.5
    _default__reduce_lr_patience: int = 2
    _default__epochs: int = 100
    _default__batchsize: int = 2 ** 12
    steps_per_epoch: Union[int, str] = "iter_smallest_process"

    # parameters to add into the `parameters` attribute to determine the 'parameters_repr' and to store in a yaml file
    bookkeep_params: set[str] = {
        # base params
        "data_loader", "input_features", "train_val_test_split",
        "processes", "sub_process_class_factors", "class_factors", "train_nodes",
        "negative_weights", "folds",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate",
        # ModelFitMixin
        "callbacks",
        "reduce_lr_factor", "reduce_lr_patience", "reduce_lr_min_delta", "reduce_lr_mode", "reduce_lr_monitor",
        "early_stopping_monitor", "early_stopping_min_delta",
        "epochs", "batchsize",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        # base params
        "processes", "class_factors", "sub_process_class_factors",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate",
        # ModelFitMixin
        "callbacks",
        "reduce_lr_factor", "reduce_lr_patience", "reduce_lr_min_delta", "reduce_lr_mode", "reduce_lr_monitor",
        "early_stopping_monitor", "early_stopping_min_delta",
        "epochs", "batchsize",
    }

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def cast_ml_param_values(self):
        super().cast_ml_param_values()

    def setup(self) -> None:
        super().setup()


#################################
#                               #
# config dicts                  #
# add more configurations here  #
#                               #
#################################

# which processes should be used for training
procs_config = DotDict({
    "default": DenseClassifier._default__processes,
})

# which input features should be used for training
input_features_config = DotDict({
    "default": DenseClassifier.input_features,
})

# how to setup the train nodes (aka output classes)
train_nodes_config = DotDict({
    "default": DenseClassifier.train_nodes,
    "mergedbkgs": {
        "tt": {
            "ml_id": 0,
            "color": "#D62727",  # red
        },
        "st": {
            "ml_id": 1,
            "color": "#1F77B4",  # blue
        },
        "other": {
            "ml_id": 2,
            "sub_processes": ["dy", "w_lnu"],
            "label": "Other",
            "color": "#2BA02B",  # green
            "class_factor_mode": "xsec",
        },
    },
})

# how to weight the different classes during training to balance out class imbalances
class_factors_config = DotDict({
    "default": DenseClassifier._default__class_factors,
    "ones": {},  # defaults to 1 (NOTE: do not try to use defaultdict! does not work with hash generation) FIXME?
    "benchmark_from_hbw": {
        "tt": 8,
        "st": 2,
        "dy": 2,
        "w_lnu": 1,
    },
})

# which configs/eras to use for training
configs_config = DotDict({
    "24": lambda self, requested_configs: ["run3_mtt_2024_nano_v15_new"],
})

#################################
#                               #
# derived MLModels              #
#                               #
#################################

# only used as first setup testing
simple_from_hbw = DenseClassifier.derive("simple_from_hbw", cls_dict={
    "training_configs": configs_config["24"],
    "processes": procs_config.default,
    "input_features": input_features_config.default,
    "class_factors": class_factors_config.default,
    "train_nodes": train_nodes_config.default,
})

# first try of gridsearching base -> odd validation weights
v1_251219 = DenseClassifier.derive("v1_251219", cls_dict={
    "training_configs": configs_config["24"],
    "learningrate": 0.0001,
    "epochs": 100,
    "batchsize": 2 ** 10,
    "dropout": 0.3,
})

v1_mergedbkgs = v1_251219.derive("v1_mergedbkgs", cls_dict={
    "processes": [
        "tt",
        "st",
        "dy",
        "w_lnu",
    ],
    "input_features": input_features_config.default,
    "class_factors": {
        "tt": 1,
        "st": 1,
        "dy": 1,
        "w_lnu": 1,
    },
    "train_nodes": {
        "tt": {"ml_id": 0},
        "st": {"ml_id": 1},
        "other": {
            "ml_id": 2,
            "sub_processes": ["dy", "w_lnu"],
            "label": "Other",
            "color": "#5E8FFC",  # blue
            "class_factor_mode": "xsec",
        },
    },
})

# DNN from ANv12/code on github:
#   https://github.com/jabuschh/ZprimeClassifier/blob/8c3a8eee0e1682605b0f03503636e339ae2ed543/steer_inputs_DNN.py
#   https://github.com/jabuschh/ZprimeClassifier/blob/8c3a8eee0e1682605b0f03503636e339ae2ed543/Training.py#L103
# now using reasonable validation weights and early stopping
# also, using 5-fold cross validation as in other models (DIFFERENT FROM ANv12!)
v1_AN_v12 = DenseClassifier.derive("v1_AN_v12", cls_dict={
    "training_configs": configs_config["24"],
    "learningrate": 0.0005,
    "epochs": 500,
    "batchsize": 2 ** 15,
    "train_val_test_split": (0.6, 0.2, 0.2),
    "train_nodes": train_nodes_config.mergedbkgs,
    "dropout": 0.5,
    "reduce_lr_kwargs": {
        # FIXME these were not used in current training, need to set them properly by removing 'reduce_lr_' prefix in keys
        # but keep for now to reproduce old results
        "reduce_lr_factor": 0.5,
        "reduce_lr_patience": 50,
        "reduce_lr_min_delta": 0.001,
        "reduce_lr_mode": "min",
    },
    "early_stopping_kwargs": {
        "monitor": "val_loss",
        "min_delta": 0.005,
        "patience": 10,
        "start_from_epoch": 10,
    },
    "callbacks": {
        "backup", "checkpoint", "reduce_lr",
        "early_stopping",
    }
})

v2_AN_v12 = v1_AN_v12.derive("v2_AN_v12", cls_dict={
    "learningrate": 0.0001,
    "reduce_lr_kwargs": {
        "monitor": "val_loss",
        "factor": 0.5,
        "patience": 5,
        "min_delta": 0.01,
        "mode": "min",
    },
    "early_stopping_kwargs": {
        "monitor": "val_loss",
        "min_delta": 0.0005,
        "patience": 30,
        "start_from_epoch": 20,
        "mode": "min",
    },
})















# #
# # grid search models
# #

# from mtt.util import build_param_product


# def physics_weights(strategy="balanced"):
#     if strategy == "balanced":
#         return {"tt": 1, "st": 1, "w_lnu": 1, "dy": 1}
#     elif strategy == "benchmark_from_hbw":
#         return {"tt": 8, "st": 2, "w_lnu": 2, "dy": 1}
#     elif strategy == "benchmark_from_hbw_inverted":
#         return {"tt": 1, "st": 2, "w_lnu": 2, "dy": 8}
#     else:
#         raise ValueError(f"Unknown strategy: {strategy}")


# example_grid_search = {
#     "layers": [(64, 64), (128, 128), (256, 256), (512, 512)],
#     "learningrate": [0.00500, 0.00050],
#     "negative_weights": ["ignore"],
#     "epochs": [500],
#     "batchsize": [1024, 2048, 4096],
#     "dropout": [0.1, 0.3, 0.5],
#     "sub_process_class_factors": [
#         physics_weights("balanced"),
#         physics_weights("benchmark_from_hbw"),
#         physics_weights("benchmark_from_hbw_inverted"),
#     ],  # weighting should not change AUCs, so optimize it separately
# }

# param_product_v1 = build_param_product(example_grid_search, lambda i: f"dense_gridsearch_v1_{i}")
# param_product_v1_mergedbkgs = build_param_product(example_grid_search, lambda i: f"dense_gridsearch_v1_mergedbkgs_{i}")

# # to use these derived models, include this file in the law.cfg (ml_modules)
# for model_name, params in param_product_v1.items():
#     dense_model = v1_251219.derive(model_name, cls_dict=params)

# for model_name, params in param_product_v1_mergedbkgs.items():
#     dense_model = v1_mergedbkgs.derive(model_name, cls_dict=params)

# # store model names as tuple to be exportable for scripts
# grid_search_models = tuple(param_product_v1.keys())
# grid_search_models_mergedbkgs = tuple(param_product_v1_mergedbkgs.keys())
