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


class DenseClassifierDL(DenseModelMixin, ModelFitMixin, MLClassifierBase):
    _default__processes: tuple = (
        "tt",
        "st",
        "dy",
        "w_lnu",
    )
    train_nodes: dict = {
        "tt": {"ml_id": 1},
        "st": {"ml_id": 2},
        "dy": {"ml_id": 3},
        "w_lnu": {"ml_id": 4},
    }
    _default__class_factors: dict = {
        "tt": 1,
        "st": 1,
        "dy": 1,
        "w_lnu": 1,
    }

    input_features = [
        "n_jet",
        "n_fatjet",
    ] + [
        f"jet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "mass", "btagUParTAK4B")
        for i in range(5)
    ] + [
        f"fatjet_{var}_{i + 1}"
        for var in ("energy", "pt", "eta", "phi", "msoftdrop", "tau21", "tau32")
        for i in range(3)
    ] + [
        f"lepton_{var}"
        for var in ("energy", "pt", "eta", "phi")
    ] + [
        f"met_{var}"
        for var in ("pt", "phi")
    ]

    preparation_producer_name = "prepml"

    folds: int = 5
    negative_weights: str = "ignore"

    # overwriting DenseModelMixin parameters
    # same as simple dnn from before
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
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
        "epochs", "batchsize",
    }

    # parameters that can be overwritten via command line
    settings_parameters: set[str] = {
        # base params
        "processes", "class_factors", "sub_process_class_factors",
        # DenseModelMixin
        "activation", "layers", "dropout", "learningrate",
        # ModelFitMixin
        "callbacks", "reduce_lr_factor", "reduce_lr_patience",
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


#
# configs
#

processes = DotDict({
    "simple": [
        "tt",
        "st",
        "dy",
        "w_lnu",
    ],
})
input_features = DotDict({
    "default": DenseClassifierDL.input_features,
})

class_factors = {
    "default": DenseClassifierDL._default__class_factors,
    "ones": {},  # defaults to 1 (NOTE: do not try to use defaultdict! does not work with hash generation)
    "benchmark_from_hbw": {
        "tt": 8,
        "st": 2,
        "dy": 2,
        "w_lnu": 1,
    },
}

configs = DotDict({
    "24": lambda self, requested_configs: ["run3_mtt_2024_nano_v15_new"],
})

#
# derived MLModels
#

simple_from_hbw = DenseClassifierDL.derive("simple_from_hbw", cls_dict={
    "training_configs": configs["24"],
    "input_features": input_features.default,
    "processes": (
        "tt",
        "st",
        "dy",
        "w_lnu",
    ),
    "train_nodes": {
        "tt": {"ml_id": 0},
        "st": {"ml_id": 1},
        "dy": {
            "ml_id": 2,
            # "sub_processes": ["dy_m10to50", "dy_m50toinf"],  # FIXME combine dy + w_lnu here?
            # "label": "DY",
            # "color": color_palette["yellow"],
            # "class_factor_mode": "xsec",
        },
        "w_lnu": {"ml_id": 3},
    },
})
