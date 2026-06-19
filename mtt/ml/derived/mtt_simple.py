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
        f"jet_{i + 1}_{var}"
        for var in ("energy", "pt", "eta", "phi", "mass", "btagUParTAK4B")
        for i in range(5)
    ]) + tuple([
        f"fatjet_{i + 1}_{var}"
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
    # full feature lists
    "default": DenseClassifier.input_features,
    "no_btag_scores": tuple([
        "n_jet",
        "n_fatjet",
        "n_bjet",
    ]) + tuple([
        f"jet_{i + 1}_{var}"
        for var in ("energy", "pt", "eta", "phi", "mass")
        for i in range(5)
    ]) + tuple([
        f"fatjet_{i + 1}_{var}"
        for var in ("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ]) + tuple([
        f"lepton_{var}"
        for var in ("energy", "pt", "eta", "phi")
    ]) + tuple([
        f"met_{var}"
        for var in ("pt", "phi")
    ]),
    "btag_buckets": tuple([
        "n_jet",
        "n_fatjet",
        "n_bjet",
    ]) + tuple([
        f"jet_{i + 1}_{var}"
        for var in ("energy", "pt", "eta", "phi", "mass", "btagUParTAK4B_buckets")
        for i in range(5)
    ]) + tuple([
        f"fatjet_{i + 1}_{var}"
        for var in ("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ]) + tuple([
        f"lepton_{var}"
        for var in ("energy", "pt", "eta", "phi")
    ]) + tuple([
        f"met_{var}"
        for var in ("pt", "phi")
    ]),
    "highlevel": tuple([
        "n_jet",
        "n_fatjet",
        "n_bjet",
        "ht_jets",
        "ht_fatjets",
        "ht_sum",
        "st",
        "leading_jets_deltar",
        "leading_fatjets_deltar",
        "leading_jetfatjet_deltar",
    ]) + tuple([
        f"jet_{i + 1}_{var}"
        for var in ("pt", "eta", "mass")
        for i in range(5)
    ]) + tuple([
        f"fatjet_{i + 1}_{var}"
        for var in ("pt", "eta", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ]) + tuple([
        f"lepton_{var}"
        for var in ("pt", "eta",)
    ]) + tuple([
        f"met_{var}"
        for var in ("pt", "phi")
    ]),
    # atomic feature lists
    "ak4_jets_all": tuple([
        f"jet_{i + 1}_{var}"
        for var in ("energy", "pt", "eta", "phi", "mass")
        for i in range(5)
    ]),
    "ak4_jets_minimal": tuple([
        f"jet_{i + 1}_{var}"
        for var in ("pt", "eta", "mass")
        for i in range(5)
    ]),
    "ak8_fatjets_all": tuple([
        f"fatjet_{i + 1}_{var}"
        for var in ("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ]),
    "ak8_fatjets_minimal": tuple([
        f"fatjet_{i + 1}_{var}"
        for var in ("pt", "eta", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ]),
    "lepton_all": tuple([
        f"lepton_{var}"
        for var in ("energy", "pt", "eta", "phi")
    ]),
    "lepton_minimal": tuple([
        f"lepton_{var}"
        for var in ("pt", "eta")
    ]),
    "met_all": tuple([
        f"met_{var}"
        for var in ("pt", "phi")
    ]),
    "btag_scores": tuple([
        f"jet_{i + 1}_btagUParTAK4B_buckets"
        for i in range(5)
    ]),
    "event_level_all": (
        "n_jet",
        "n_fatjet",
        "n_bjet",
        "ht_jets",
        "ht_fatjets",
        "ht_sum",
        "st",
        "leading_jets_deltar",
        "leading_fatjets_deltar",
        "leading_jetfatjet_deltar",
    ),
    "jet_multiplicity": (
        "n_jet",
        "n_fatjet",
        "n_bjet",
    ),
    "ht_variables": (
        "ht_jets",
        "ht_fatjets",
        "ht_sum",
        "st",
    ),
    "deltaR_variables": (
        "leading_jets_deltar",
        "leading_fatjets_deltar",
        "leading_jetfatjet_deltar",
    ),
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
    "naive_balanced": {
        # not reasonable, don't use
        "tt": 250,
        "st": 20,
        "dy": 2,
        "w_lnu": 1,
    },
    "more_balanced": {
        "tt": 10,
        "st": 4,
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
    },
})

v2_AN_v12 = v1_AN_v12.derive("v2_AN_v12", cls_dict={
    # training without t channel single top
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

# same parameters as v2_AN_v12 but with t channel single top included
# renaming to keep output files separate from v2_AN_v12
v3_AN_v12 = v2_AN_v12.derive("v3_AN_v12")

# new training without b tagging scores
# drop `AN_v12` from name since this is now different from the original ANv12 model
# also, learningrate and callbacks need to be adapted differently without b tagging scores
v4_no_btag = v3_AN_v12.derive("v4_no_btag", cls_dict={
    "input_features": tuple([
        "n_jet",
        "n_fatjet",
    ]) + tuple([
        f"jet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "mass")
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
    ]),
    "learningrate": 0.0005,
    "batchsize": 2 ** 13,
    "reduce_lr_kwargs": {
        "monitor": "val_loss",
        "factor": 0.5,
        "patience": 5,
        "min_delta": 0.001,
        "mode": "min",
    },
    "steps_per_epoch": 400.0,
})

v5 = v2_AN_v12.derive("v5", cls_dict={
    "steps_per_epoch": 20.0,
    "input_features": tuple([
        "n_jet",
        "n_fatjet",
    ]) + tuple([
        f"jet_{i + 1}_{var}"
        for var in ("energy", "pt", "eta", "phi", "mass", "btagUParTAK4B_buckets")
        for i in range(5)
    ]) + tuple([
        f"fatjet_{i + 1}_{var}"
        for var in ("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ]) + tuple([
        f"lepton_{var}"
        for var in ("energy", "pt", "eta", "phi")
    ]) + tuple([
        f"met_{var}"
        for var in ("pt", "phi")
    ]),
})

v6 = v2_AN_v12.derive("v6", cls_dict={
    "steps_per_epoch": 20.0,
    "input_features": input_features_config.no_btag_scores,
    "learningrate": 0.0001,
    "reduce_lr_kwargs": {
        "monitor": "val_loss",
        "factor": 0.5,
        "patience": 5,
        "min_delta": 0.001,
        "mode": "min",
    },
    "early_stopping_kwargs": {
        "monitor": "val_loss",
        "min_delta": 0.001,
        "patience": 8,
        "start_from_epoch": 20,
        "mode": "min",
    },
})

chatties_baseline = DenseClassifier.derive("chatties_baseline", cls_dict={
    "training_configs": configs_config["24"],
    "layers": (128, 64, 32),
    "learningrate": 1e-3,
    "batchsize": 4096,
    "dropout": 0.3,
    "train_nodes": train_nodes_config.mergedbkgs,
    "epochs": 100,
    "early_stopping_kwargs": {
        "monitor": "val_f1_score",
        "min_delta": 0.001,
        "patience": 10,
        "start_from_epoch": 20,
        "mode": "max",
    },
    "reduce_lr_kwargs": {
        "monitor": "val_loss",
        "factor": 0.5,
        "patience": 5,
        "min_delta": 0.001,
        "mode": "min",
    },
    "steps_per_epoch": 200.0,
    "callbacks": {
        "backup", "checkpoint", "reduce_lr",
        "early_stopping",
    },
    "train_val_test_split": (0.6, 0.2, 0.2),
    "input_features": input_features_config.no_btag_scores,
    "loss": "focal_loss",
    "focal_loss_alpha": 1.0,
    "focal_loss_gamma": 2.0,
})

tt_st_classifier = DenseClassifier.derive("tt_st_classifier", cls_dict={
    "training_configs": configs_config["24"],
    "processes": ["tt", "st"],
    "input_features": input_features_config.no_btag_scores,
    "class_factors": class_factors_config.default,
    "train_nodes": {
        "tt": {"ml_id": 0},
        "st": {"ml_id": 1},
    },
    "learningrate": 0.0001,
    "epochs": 100,
    "batchsize": 2 ** 10,
    "dropout": 0.3,
    "reduce_lr_kwargs": {
        "monitor": "val_loss",
        "factor": 0.5,
        "patience": 5,
        "min_delta": 0.001,
        "mode": "min",
    },
    "early_stopping_kwargs": {
        "monitor": "val_loss",
        "min_delta": 0.001,
        "patience": 8,
        "start_from_epoch": 20,
        "mode": "min",
    },
    "layers": (512, 512),
    "train_val_test_split": (0.6, 0.2, 0.2),
    "steps_per_epoch": 100.0,
    "callbacks": {
        "backup", "checkpoint", "reduce_lr",
        "early_stopping",
    },
})

tt_st_highlevel = tt_st_classifier.derive("tt_st_highlevel", cls_dict={
    "input_features": tuple([
        "n_jet",
        "n_fatjet",
        "n_bjet",
        "ht_jets",
        "ht_fatjets",
        "ht_sum",
        "st",
        "leading_jets_deltar",
        "leading_fatjets_deltar",
        "leading_jetfatjet_deltar",
    ]) + tuple([
        f"jet_{i + 1}_{var}"
        for var in ("pt", "eta", "mass")
        for i in range(5)
    ]) + tuple([
        f"fatjet_{i + 1}_{var}"
        for var in ("pt", "eta", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ]) + tuple([
        f"lepton_{var}"
        for var in ("pt", "eta",)
    ]) + tuple([
        f"met_{var}"
        for var in ("pt", "phi")
    ]),
    "batchsize": 4096,
})

v5_red_stats = v5.derive("v5_red_stats", cls_dict={
    # "steps_per_epoch": 10.0,
    "batchsize": 4096,
})

v5_balanced = v5.derive("v5_balanced", cls_dict={
    "class_factors": class_factors_config.naive_balanced,
    "input_features": (
        input_features_config.ak4_jets_all
        + input_features_config.ak8_fatjets_all
        + input_features_config.lepton_all
        + input_features_config.met_all
        + input_features_config.jet_multiplicity
    ),
})

v5_more_balanced = v5.derive("v5_more_balanced", cls_dict={
    "class_factors": class_factors_config.more_balanced,
    "input_features": (
        input_features_config.ak4_jets_all
        + input_features_config.ak8_fatjets_all
        + input_features_config.lepton_all
        + input_features_config.met_all
        + input_features_config.jet_multiplicity
    ),
})

# first optuna run results 11.03.26
# {
# 'n_layers': 2,
# 'units_l0': 1024, 'dropout_l0': 0.15000000000000002,
# 'units_l1': 128, 'dropout_l1': 0.05,
# 'learning_rate': 0.002725378204287723, 'batch_size': 512, 'epochs': 90,
# 'early_stopping_patience': 9,
# 'learning_rate_reduction_factor': 0.1, 'learning_rate_reduction_patience': 3
# }

v6_optuna = DenseClassifier.derive("v6_optuna", cls_dict={
    "class_factors": class_factors_config.default,
    "input_features": (
        input_features_config.ak4_jets_all
        + input_features_config.btag_scores
        + input_features_config.ak8_fatjets_all
        + input_features_config.lepton_all
        + input_features_config.met_all
        + input_features_config.jet_multiplicity
    ),
    "layers": (1024, 128),
    "dropout": 0.15,
    "learningrate": 0.002725378204287723,
    "batchsize": 512,
    "epochs": 90,
    "early_stopping_kwargs": {
        "monitor": "val_loss",
        "min_delta": 0.001,
        "patience": 9,
        "start_from_epoch": 20,
        "mode": "min",
    },
    "reduce_lr_kwargs": {
        "monitor": "val_loss",
        "factor": 0.1,
        "patience": 3,
        "min_delta": 0.001,
        "mode": "min",
    },
    "steps_per_epoch": 50.0,
    "train_val_test_split": (0.6, 0.2, 0.2),
})

v7 = v2_AN_v12.derive("v7", cls_dict={
    "steps_per_epoch": 50.0,
    "batchsize": 4096,
    "input_features": (
        input_features_config.ak4_jets_all
        + input_features_config.btag_scores
        + input_features_config.ak8_fatjets_all
        + input_features_config.lepton_all
        + input_features_config.met_all
        + input_features_config.jet_multiplicity
    ),
    "folds": 3,
})

v8 = DenseClassifier.derive("v8", cls_dict={
    "steps_per_epoch": 50.0,
    "batchsize": 4096,
    "input_features": (
        input_features_config.ak4_jets_all
        + input_features_config.btag_scores
        + input_features_config.ak8_fatjets_all
        + input_features_config.lepton_all
        + input_features_config.met_all
        + input_features_config.jet_multiplicity
    ),
    "folds": 5,
    "class_factors": {
        "tt": 1,
        "dy": 1,
        "w_lnu": 1,
    },
    "train_nodes": {
        "tt": {"ml_id": 0},
        "other": {
            "ml_id": 1,
            "sub_processes": ["dy", "w_lnu"],
            "label": "Other",
            "color": "#5E8FFC",  # blue
            "class_factor_mode": "xsec",
        },
    },
    "train_val_test_split": (0.6, 0.2, 0.2),
    "processes": [
        "tt",
        "dy",
        "w_lnu",
    ],
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
