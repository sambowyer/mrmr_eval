"""Thin per-source DataLoader abstraction used by ``method_runner.MethodRunner``.

Slimmed from the paper repo: supports only OpenLLM / HELM / GLUE (binary).
"""

import os
import joblib as jbl
import numpy as np
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

from data_utils import HelmLite, OpenLLM, load_glue_predictions


class DataLoader(ABC):
    """Abstract base class for data loading."""

    @abstractmethod
    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Return ``(scores, model_names, true_acc)`` for *dataset_name*.

        - ``scores``: 2D array of shape ``(num_models, num_items)``
        - ``model_names``: list of length ``num_models``
        - ``true_acc``: 1D array of shape ``(num_models,)``
        """


class OpenLLMLoader(DataLoader):
    """Loader for OpenLLM leaderboard datasets."""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        scores_path = os.path.join("data", "scores", f"{dataset_name}.jbl")
        scores = jbl.load(scores_path).astype(np.float32)

        model_names_path = os.path.join("data", "scores", f"{dataset_name}_models.jbl")
        model_info = jbl.load(model_names_path)
        model_names = model_info

        true_acc = scores.mean(axis=1)
        return scores, model_names, true_acc


class HelmsLoader(DataLoader):
    """Loader for HELM Lite benchmark datasets."""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        helm = HelmLite(tasks=[dataset_name])
        helm.download_and_check()
        dataset = helm.get_datasets()

        scores = np.array(dataset["acc"]).T.astype(np.float32)
        model_names = helm.models
        true_acc = scores.mean(axis=1)
        return scores, model_names, true_acc


class GlueLoader(DataLoader):
    """Loader for GLUE benchmark datasets."""

    def load(self, dataset_name: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
        all_preds, gold_labels = load_glue_predictions(dataset_name)

        scores = np.zeros((all_preds.shape[0], all_preds.shape[1]), dtype=np.float32)
        for m in range(all_preds.shape[0]):
            scores[m] = all_preds[m].argmax(-1) == gold_labels

        model_names = [f"model_{i}" for i in range(scores.shape[0])]
        true_acc = scores.mean(axis=1)
        return scores, model_names, true_acc


class LoaderRegistry:
    """Registry mapping a data-source string to a ``DataLoader`` instance."""

    _loaders: Dict[str, DataLoader] = {
        "openllm": OpenLLMLoader(),
        "helm": HelmsLoader(),
        "glue": GlueLoader(),
    }

    @classmethod
    def get_loader(cls, source: str) -> DataLoader:
        loader = cls._loaders.get(source.lower())
        if loader is None:
            raise ValueError(
                f"Unknown data source: {source}.  Available: "
                f"{list(cls._loaders.keys())}"
            )
        return loader

    @classmethod
    def register_loader(cls, source: str, loader: DataLoader):
        cls._loaders[source.lower()] = loader
