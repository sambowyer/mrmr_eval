"""Backfill methods that reuse coresets from pre-existing IRT / anchor-points
methods and refit a Ridge or Kernel Ridge regressor.

These methods load a base method's checkpoint (to extract coreset indices),
then fit a RidgeCV or KernelRidge regressor on those coreset columns -- the
same regression pipeline used by the mrmr / kmrmr families.

The runner passes ``base_ckpt_path`` to ``fit()`` when it detects
``_base_method_key`` on the method class.
"""

import numpy as np
import joblib as jbl
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, LeaveOneOut

from .base import BenchPred


def _is_binary_scores(scores: np.ndarray) -> bool:
    finite = scores[np.isfinite(scores)]
    return finite.size > 0 and len(np.unique(finite)) <= 2


class BackfillRidgePred(BenchPred):
    """Load a pre-existing method's coreset and fit RidgeCV on it.

    ``_base_method_key`` must be set (via subclass or factory) to the
    ``all_methods`` key of the base method whose checkpoint will be loaded.
    """

    _base_method_key = None

    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.rgs = None
        self._binary = None
        self.alpha_ = None
        self.alpha_range_ = None

    def _build_regressor(self, X: np.ndarray, y: np.ndarray):
        alphas = (
            np.logspace(-1, 1, 5) if self._binary
            else np.logspace(-2, 2, 9)
        )
        rgs = RidgeCV(alphas=alphas)
        rgs.fit(X, y.reshape(-1, 1))
        self.alpha_ = float(rgs.alpha_)
        self.alpha_range_ = (float(alphas.min()), float(alphas.max()))
        return rgs

    def fit(self, source_full_scores, coreset_size, seed=42,
            base_ckpt_path=None, *args, **kwargs):
        if base_ckpt_path is None:
            raise ValueError(
                f"BackfillRidgePred ({self._base_method_key}+) requires "
                "base_ckpt_path; the runner should supply it automatically."
            )

        from . import all_methods
        base_cls = all_methods[self._base_method_key]
        base_method = base_cls()
        base_method.load(base_ckpt_path)
        self.compressed_data_indices = np.asarray(base_method.get_coreset())

        self._binary = _is_binary_scores(source_full_scores)
        X = source_full_scores[:, self.compressed_data_indices]
        y = source_full_scores.mean(axis=1)
        self.rgs = self._build_regressor(X, y)
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)
        return self.rgs.predict(target_coreset_scores).ravel()

    def refit_regressor(self, source_full_scores):
        coreset = self.get_coreset()
        X = source_full_scores[:, coreset]
        y = source_full_scores.mean(axis=1)
        return self._build_regressor(X, y)

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.rgs), path_save)

    def load(self, path_load):
        self.compressed_data_indices, self.rgs = jbl.load(path_load)
        return self


class BackfillKRRPred(BackfillRidgePred):
    """Like BackfillRidgePred but uses Kernel Ridge Regression with a
    polynomial kernel (same pipeline as kmrmr).
    """

    _degree = 2

    def _build_regressor(self, X: np.ndarray, y: np.ndarray):
        alphas = (
            np.logspace(-1, 1, 5) if self._binary
            else np.logspace(-2, 2, 9)
        )
        rgs = GridSearchCV(
            KernelRidge(kernel="poly", degree=self._degree, coef0=1),
            param_grid={"alpha": alphas},
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
        )
        rgs.fit(X, y.ravel())
        self.alpha_ = float(rgs.best_params_["alpha"])
        self.alpha_range_ = (float(alphas.min()), float(alphas.max()))
        return rgs


def _make_backfill_ridge(base_method_key):
    """Create a BackfillRidgePred subclass bound to *base_method_key*."""

    class _Variant(BackfillRidgePred):
        _base_method_key = base_method_key

    name = f"BackfillRidge_{base_method_key}"
    _Variant.__name__ = name
    _Variant.__qualname__ = name
    _Variant.__module__ = __name__
    _Variant.__doc__ = (
        f"Backfill Ridge regressor using coreset from '{base_method_key}'."
    )
    return _Variant


def _make_backfill_krr(base_method_key, degree=2):
    """Create a BackfillKRRPred subclass bound to *base_method_key*."""

    _degree = degree

    class _Variant(BackfillKRRPred):
        _base_method_key = base_method_key

    _Variant._degree = _degree

    suffix = "" if _degree == 2 else str(_degree)
    name = f"BackfillKRR{suffix}_{base_method_key}"
    _Variant.__name__ = name
    _Variant.__qualname__ = name
    _Variant.__module__ = __name__
    _Variant.__doc__ = (
        f"Backfill Kernel Ridge (degree-{_degree}) regressor using "
        f"coreset from '{base_method_key}'."
    )
    return _Variant
