import os
import warnings
import numpy as np
import joblib as jbl
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from tqdm import tqdm
from .base import BenchPred, _MeanPredictor, set_random_seed


def _is_binary_scores(scores: np.ndarray) -> bool:
    """Check whether a score matrix contains only binary (0/1) values."""
    finite = scores[np.isfinite(scores)]
    return finite.size > 0 and len(np.unique(finite)) <= 2


class RandomSampling(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)

        self.compressed_data_indices = np.random.permutation(num_data)[
            :coreset_size
        ]
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return target_coreset_scores.mean(1)

    def refit_regressor(self, source_full_scores):
        return _MeanPredictor()

    def save(self, path_save):
        jbl.dump(self.compressed_data_indices, path_save)

    def load(self, path_load):
        self.compressed_data_indices = jbl.load(path_load)
        return self


class RandomSamplingAndLearn(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.rgs = None
        self._binary = None

    def _make_regressor(self):
        alphas = (
            np.logspace(-1, 1, 5) if self._binary
            else np.logspace(-2, 2, 9)
        )
        return RidgeCV(alphas=alphas)

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)
        self._binary = _is_binary_scores(source_full_scores)

        self.compressed_data_indices = np.random.permutation(num_data)[
            :coreset_size
        ]

        self.rgs = self._make_regressor()
        self.rgs.fit(
            source_full_scores[:, self.compressed_data_indices],
            source_full_scores.mean(1),
        )
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return self.rgs.predict(target_coreset_scores)

    def refit_regressor(self, source_full_scores):
        coreset = self.get_coreset()
        X = source_full_scores[:, coreset]
        y = source_full_scores.mean(axis=1)
        rgs = self._make_regressor()
        rgs.fit(X, y)
        return rgs

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.rgs), path_save)

    def load(self, path_load):
        self.compressed_data_indices, self.rgs = jbl.load(path_load)
        return self


class SampleFirstAndLearn(BenchPred):
    """Take the first *coreset_size* questions (in benchmark order) and learn."""

    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.rgs = None
        self._binary = None

    def _make_regressor(self):
        alphas = (
            np.logspace(-1, 1, 5) if self._binary
            else np.logspace(-2, 2, 9)
        )
        return RidgeCV(alphas=alphas)

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        self._binary = _is_binary_scores(source_full_scores)

        self.compressed_data_indices = np.arange(coreset_size)

        self.rgs = self._make_regressor()
        self.rgs.fit(
            source_full_scores[:, self.compressed_data_indices],
            source_full_scores.mean(1),
        )
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return self.rgs.predict(target_coreset_scores)

    def refit_regressor(self, source_full_scores):
        coreset = self.get_coreset()
        X = source_full_scores[:, coreset]
        y = source_full_scores.mean(axis=1)
        rgs = self._make_regressor()
        rgs.fit(X, y)
        return rgs

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.rgs), path_save)

    def load(self, path_load):
        self.compressed_data_indices, self.rgs = jbl.load(path_load)
        return self


class RandomSearchAndLearn(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.rgs = None
        self.search_metrics = None
        self._binary = None

    def _make_regressor(self):
        alphas = (
            np.logspace(-1, 1, 5) if self._binary
            else np.logspace(-2, 2, 9)
        )
        return RidgeCV(alphas=alphas)

    def fit(
        self,
        source_full_scores,
        coreset_size,
        num_search=10000, # 250,
        seed=42,
        *args,
        **kwargs
    ):
        num_model = source_full_scores.shape[0]
        num_data = source_full_scores.shape[1]

        assert num_model > 1
        assert num_data > 1
        assert coreset_size > 1
        assert num_data > coreset_size

        set_random_seed(seed)
        self._binary = _is_binary_scores(source_full_scores)

        order = np.random.permutation(num_model)
        tr_models = order[: int(num_model * 0.75)]
        val_models = order[int(num_model * 0.75) :]
        val_true = source_full_scores.mean(1)[val_models]

        mae_list = []
        rmse_list = []
        spearman_list = []
        kendall_list = []

        best_idxs, best_gap = None, 1e9
        for _ in tqdm(range(num_search), desc="Random Search and Learn"):
            selected_idxs = np.random.permutation(num_data)[:coreset_size]
            rgs = self._make_regressor()
            rgs.fit(
                source_full_scores[tr_models][:, selected_idxs],
                source_full_scores.mean(1)[tr_models],
            )
            estimated_scores = rgs.predict(source_full_scores[:, selected_idxs])
            val_pred = estimated_scores[val_models]

            residuals = val_pred - val_true
            mae = float(np.mean(np.fabs(residuals)))
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            sp_corr = float(spearmanr(val_true, val_pred).statistic)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kt_corr = float(kendalltau(val_true, val_pred).statistic)

            mae_list.append(mae)
            rmse_list.append(rmse)
            spearman_list.append(sp_corr)
            kendall_list.append(kt_corr)

            if mae < best_gap:
                best_gap = mae
                best_idxs = selected_idxs

        mae_arr = np.array(mae_list)
        rmse_arr = np.array(rmse_list)
        spearman_arr = np.array(spearman_list)
        kendall_arr = np.array(kendall_list)

        self.search_metrics = {
            "mae": mae_arr,
            "rmse": rmse_arr,
            "spearman": spearman_arr,
            "kendall": kendall_arr,
            "mae_mean": float(np.mean(mae_arr)),
            "mae_std": float(np.std(mae_arr)),
            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_std": float(np.std(rmse_arr)),
            "spearman_mean": float(np.mean(spearman_arr)),
            "spearman_std": float(np.std(spearman_arr)),
            "kendall_mean": float(np.mean(kendall_arr)),
            "kendall_std": float(np.std(kendall_arr)),
        }

        self.compressed_data_indices = best_idxs
        self.rgs = self._make_regressor()
        self.rgs.fit(
            source_full_scores[:, self.compressed_data_indices],
            source_full_scores.mean(1),
        )
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return self.rgs.predict(target_coreset_scores)

    def refit_regressor(self, source_full_scores):
        coreset = self.get_coreset()
        X = source_full_scores[:, coreset]
        y = source_full_scores.mean(axis=1)
        rgs = self._make_regressor()
        rgs.fit(X, y)
        return rgs

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.rgs), path_save)
        if self.search_metrics is not None:
            sm_path = os.path.join(
                os.path.dirname(path_save), "search_metrics.jbl",
            )
            jbl.dump(self.search_metrics, sm_path)

    def load(self, path_load):
        self.compressed_data_indices, self.rgs = jbl.load(path_load)
        sm_path = os.path.join(
            os.path.dirname(path_load), "search_metrics.jbl",
        )
        if os.path.isfile(sm_path):
            self.search_metrics = jbl.load(sm_path)
        else:
            self.search_metrics = None
        return self


class SmallSearchAndLearn(RandomSearchAndLearn):
    """Like RandomSearchAndLearn but searches only 1000 coresets instead of 10,000."""

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        return super().fit(source_full_scores, coreset_size, num_search=1000, seed=seed, *args, **kwargs)


def _make_krr_random_variant(base_cls, degree=2):
    """Create a KRR subclass of a random sampling+learn method."""

    _degree = degree

    class _KRRVariant(base_cls):
        def _make_regressor(self):
            alphas = (
                np.logspace(-1, 1, 5) if self._binary
                else np.logspace(-2, 2, 9)
            )
            return GridSearchCV(
                KernelRidge(kernel="poly", degree=_degree, coef0=1),
                param_grid={"alpha": alphas},
                cv=LeaveOneOut(),
                scoring="neg_mean_squared_error",
            )

        def fit(self, *args, **kwargs):
            result = super().fit(*args, **kwargs)
            self.search_metrics = None
            return result

    suffix = "" if _degree == 2 else str(_degree)
    new_name = f"K{suffix}{base_cls.__name__}"
    _KRRVariant.__name__ = new_name
    _KRRVariant.__qualname__ = new_name
    _KRRVariant.__module__ = __name__
    return _KRRVariant
