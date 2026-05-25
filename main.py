"""CLI entry-point for the mrmr_eval tutorial repo.

Mirrors the structure of ``_public_paper_repo/mrmr_paper/main.py`` but with
a slimmed method registry and only the binary OpenLLM / HELM / GLUE data
sources.  Designed to be importable from ``tutorial.ipynb`` so the same
``DEFAULT_METHODS`` definition is reused there.

Quick start::

    python main.py --dataset_name ifeval --coreset_size 10% \\
        --num_run 3 --methods default --no-multi_process --no-use_git
"""

import os

from zarth_utils.config import Config
from method_runner import MethodRunner
from data_utils import openllm_datasets, helm_datasets, glue_datasets
from benchpred import all_methods


def _mrmr_method_name(k_prefix: str, order: int, objective: str) -> str:
    """Build an mRMR method-registry key.

    ``order == 3`` is the default ``mi_k`` and uses the bare class name
    (e.g. ``mrmr_MIQ_y``).  Higher ``order`` values produce numbered
    variants (e.g. ``mrmr5_MIQ_y``).
    """
    if int(order) == 3:
        return f"{k_prefix}mrmr_{objective}_y"
    return f"{k_prefix}mrmr{int(order)}_{objective}_y"


binary_mrmr = [
    _mrmr_method_name(k_prefix, order, "MIQ")
    # for order in [3, 4, 5, 6, 7, 8, 9]
    for order in [5]
    for k_prefix in ("", "k")
]
# MID variants are also available — uncomment the next block to include them.
# binary_mrmr_mid = [
#     _mrmr_method_name(k_prefix, order, "MID")
#     for order in [3, 4, 5, 6, 7, 8, 9]
#     for k_prefix in ("", "k")
# ]

# irt_methods = ["gpirt1", "gpirt5", "gpirt"]
irt_methods = ["gpirt1"] #, "gpirt5", "gpirt"]

other_baselines = [
    # "metabench",  # commented out: very slow even for small datasets
    "lasso",
    "random_search_and_learn",
    "krandom_search_and_learn",
    "random_sampling_and_learn",
    "krandom_sampling_and_learn",
    "random_sampling",
]

# anchor_points = ["anchor_points_weighted", "anchor_points_predictor"]
anchor_points = ["anchor_points_weighted"] #, "anchor_points_predictor"]

# NOTE: These will only run if the base methods have been run first 
#       (they look at the base methods' coreset files in results/)
refit_methods = ["kanchor_points_weighted+", "kgpirt1+"]

DEFAULT_METHODS = binary_mrmr + irt_methods + other_baselines + anchor_points + refit_methods


def _build_config():
    """Build the experiment configuration from CLI args / defaults."""
    config = Config(
        default_config_dict={
            "data_source": "openllm",
            "dataset_name": "ifeval",
            "datasets": openllm_datasets,
            "dir_results": "./results",
            "exp_suffix": "",
            "coreset_size": "10%",
            "methods": list(DEFAULT_METHODS),
            "model_split_method": "interpolation",
            "num_train_models": 30,
            "seed_start": 0,
            "num_run": 3,
            "multi_process": True,
            "use_git": False,
        },
        use_argparse=True,
    )

    # Convenience: when --dataset_name is supplied, run only that one dataset.
    if (
        getattr(config, "dataset_name", None)
        and config.datasets == openllm_datasets
        and config.dataset_name in openllm_datasets
    ):
        config.datasets = [config.dataset_name]
    elif config.datasets == ["openllm_datasets"]:
        config.datasets = openllm_datasets
        config.data_source = "openllm"
    elif config.datasets == ["helm_datasets"]:
        config.datasets = helm_datasets
        config.data_source = "helm"
    elif config.datasets == ["glue_datasets"]:
        config.datasets = glue_datasets
        config.data_source = "glue"

    if config.methods == ["default"]:
        config.methods = list(DEFAULT_METHODS)
    elif config.methods == ["binary_mrmr"]:
        config.methods = binary_mrmr
    elif config.methods == ["irt_methods"]:
        config.methods = irt_methods
    elif config.methods == ["other_baselines"]:
        config.methods = other_baselines
    elif config.methods == ["anchor_points"]:
        config.methods = anchor_points
    elif config.methods == ["all"]:
        config.methods = list(all_methods.keys())

    # Validate methods early so we fail before launching multiprocessing.
    for method in config.methods:
        if method not in all_methods:
            raise ValueError(
                f"Method {method!r} is not registered in benchpred.all_methods. "
                f"Available: {sorted(all_methods.keys())}"
            )

    return config


def main():
    config = _build_config()

    print("Config:")
    print(config.to_dict())
    print("Datasets:", config.datasets)
    print("Methods:", config.methods)
    print("Model split method:", config.model_split_method)
    print("Seed start:", config.seed_start)
    print("Number of runs:", config.num_run)
    print("Multi process:", config.multi_process)
    print("Coreset size:", config.coreset_size)
    print("Number of train models:", config.num_train_models)

    os.makedirs(config.dir_results, exist_ok=True)

    runner = MethodRunner(config)
    runner.run_all_datasets()

    print("Experiment completed! Results saved to:", config.dir_results)


if __name__ == "__main__":
    main()
