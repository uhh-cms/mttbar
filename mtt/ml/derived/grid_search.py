# coding: utf-8
"""

"""

from mtt.util import build_param_product
from mtt.ml.derived.mtt_simple import v1_251219, v1_mergedbkgs


def physics_weights(strategy="balanced"):
    if strategy == "balanced":
        return {"tt": 1.0, "st": 1.0, "w_lnu": 1.0, "dy": 1.0}
    elif strategy == "benchmark_from_hbw":
        return {"tt": 8.0, "st": 2.0, "w_lnu": 2.0, "dy": 1.0}
    elif strategy == "benchmark_from_hbw_inverted":
        return {"tt": 1.0, "st": 2.0, "w_lnu": 2.0, "dy": 8.0}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# weights = lambda bkg_weight: {
#     "tt": bkg_weight,
#     "st": bkg_weight,
#     "w_lnu": bkg_weight,
#     "dy": bkg_weight,
# }


example_grid_search = {  # 4*2*2*1*3*3*1 = 144 trainings
    "layers": [(64, 64), (128, 128), (256, 256), (512, 512)],
    "learningrate": [0.00500, 0.00050],
    "negative_weights": ["ignore"],
    "epochs": [500],
    "batchsize": [1024, 2048, 4096],
    "dropout": [0.1, 0.3, 0.5],
    "sub_process_class_factors": [
        physics_weights("balanced"),
        physics_weights("benchmark_from_hbw"),
        physics_weights("benchmark_from_hbw_inverted"),
    ],  # weighting should not change AUCs, so optimize it separately
}

param_product_v1 = build_param_product(example_grid_search, lambda i: f"dense_gridsearch_v1_{i}")
param_product_v1_mergedbkgs = build_param_product(example_grid_search, lambda i: f"dense_gridsearch_v1_mergedbkgs_{i}")

# to use these derived models, include this file in the law.cfg (ml_modules)
for model_name, params in param_product_v1.items():
    dense_model = v1_251219.derive(model_name, cls_dict=params)

for model_name, params in param_product_v1_mergedbkgs.items():
    dense_model = v1_mergedbkgs.derive(model_name, cls_dict=params)

# store model names as tuple to be exportable for scripts
grid_search_models = tuple(param_product_v1.keys())
grid_search_models_mergedbkgs = tuple(param_product_v1_mergedbkgs.keys())
