import warnings

import torch
import random
import numpy as np
from abc import abstractmethod
from abc import ABC
from sklearn.linear_model import RidgeCV

# PyTorch probes CUDA even when we only need CPU (e.g. IRT on CPU).  On
# machines with mismatched drivers this emits a noisy UserWarning once per
# process/worker.
warnings.filterwarnings(
    "ignore",
    message=".*CUDA initialization.*",
    category=UserWarning,
    module=r"torch\.cuda",
)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class _MeanPredictor:
    """Trivial predictor that returns the row-wise mean of the input features.

    Used by mean-only methods (RandomSampling, KCenterGreedy, Herding, etc.)
    so that ``refit_regressor`` returns an object with a sklearn-compatible
    ``.predict()`` interface.
    """

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X.mean(axis=1)


class _WeightedSumPredictor:
    """Predictor that returns a weighted sum of input features.

    Used by methods whose native prediction is a weighted combination of
    coreset scores (IRT anchor-point methods, DoubleOptimize, etc.).
    """

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X * self.weights).sum(axis=1)


class BenchPred(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        # num_model = source_full_scores.shape[0]
        # num_data = source_full_scores.shape[1]
        #
        # assert num_model > 1
        # assert num_data > 1
        # assert coreset_size > 1
        # assert num_data > coreset_size
        #
        # set_random_seed(seed)
        pass

    @abstractmethod
    def get_coreset(self):
        pass

    @abstractmethod
    def predict(self, target_coreset_scores):
        # if len(target_coreset_scores.shape) == 1:
        #     target_coreset_scores = target_coreset_scores.reshape(1, -1)
        pass

    @abstractmethod
    def save(self, path_save):
        pass

    @abstractmethod
    def load(self, path_load):
        pass

    def refit_regressor(self, source_full_scores):
        """Build a new regressor on *source_full_scores* using the existing
        coreset (from a previous ``fit`` call).

        The default implementation trains a ``RidgeCV`` to predict the mean
        score from the coreset columns.  Subclasses with their own regressor
        builders should override this to match their default prediction model.

        Args:
            source_full_scores: Score matrix of shape (M_source, N_questions)
                for the new k' data.

        Returns:
            A fitted regressor with a sklearn-compatible ``.predict()`` method.
            Does **not** mutate ``self``.
        """
        coreset = self.get_coreset()
        X = source_full_scores[:, coreset]
        y = source_full_scores.mean(axis=1)
        rgs = RidgeCV(alphas=np.logspace(-2, 2, 9))
        rgs.fit(X, y.reshape(-1, 1))
        return rgs
