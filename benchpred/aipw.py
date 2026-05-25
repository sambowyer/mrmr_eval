import numpy as np
import joblib as jbl
from sklearn.linear_model import Ridge
from .base import BenchPred, set_random_seed


class _AIPWPredictor:
    def __init__(self, ref_scores, coreset_idx, rest_indices):
        self.ref_scores = ref_scores
        self.coreset_idx = coreset_idx
        self.rest_indices = rest_indices

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        ret = []
        for i in range(X.shape[0]):
            x_train = self.ref_scores[:, self.coreset_idx].T
            y_train = X[i]
            x_test = self.ref_scores[:, self.rest_indices].T
            rgs = Ridge(alpha=10)
            rgs.fit(x_train, y_train.reshape(-1, 1))
            y_pred_train = rgs.predict(x_train).squeeze()
            y_pred_test = rgs.predict(x_test).squeeze()
            n = len(self.coreset_idx)
            N = len(self.rest_indices)
            ppi_part = (y_train - y_pred_train).mean() / (1 + n / N)
            ppi_part += y_pred_test.mean()
            ret.append(ppi_part)
        return np.array(ret)


class AIPWPred(BenchPred):
    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.source_full_scores = None

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
        self.source_full_scores = source_full_scores
        return self

    def get_coreset(self):
        return self.compressed_data_indices

    def predict(self, target_coreset_scores):
        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        num_data = self.source_full_scores.shape[1]
        rest_indices = np.array(
            [i for i in range(num_data) if i not in self.compressed_data_indices]
        )

        ret = []
        num_target_model = target_coreset_scores.shape[0]
        for i in range(num_target_model):
            x_train = self.source_full_scores[:, self.compressed_data_indices].T
            y_train = target_coreset_scores[i]
            x_test = self.source_full_scores[:, rest_indices].T

            rgs = Ridge(alpha=10)
            rgs.fit(x_train, y_train.reshape(-1, 1))

            y_pred_train = rgs.predict(x_train).squeeze()
            y_pred_test = rgs.predict(x_test).squeeze()

            n = len(self.compressed_data_indices)
            N = len(rest_indices)
            ppi_part = (y_train - y_pred_train).mean() / (1 + n / N)
            ppi_part += y_pred_test.mean()
            ret.append(ppi_part)

        return np.array(ret)

    def refit_regressor(self, source_full_scores):
        coreset_idx = self.get_coreset()
        ref_scores = source_full_scores.copy()
        num_data = ref_scores.shape[1]
        rest_indices = np.array(
            [i for i in range(num_data) if i not in coreset_idx]
        )
        return _AIPWPredictor(ref_scores, coreset_idx, rest_indices)

    def save(self, path_save):
        jbl.dump((self.compressed_data_indices, self.source_full_scores), path_save)

    def load(self, path_load):
        self.compressed_data_indices, self.source_full_scores = jbl.load(path_load)
        return self
