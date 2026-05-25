import numpy as np
import pickle
from .base import BenchPred, set_random_seed


class _SortSearchPredictor:
    def __init__(self, coreset_idx, n_samples):
        self.coreset_idx = coreset_idx
        self.n_samples = n_samples

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        agg = np.zeros(X.shape[0], dtype=np.float32)
        for i in range(X.shape[0]):
            pred = X[i].copy()
            pred[pred == 0] = -1
            cumsum = np.cumsum(pred)
            idx = np.argmax(cumsum)
            agg[i] = float(self.coreset_idx[idx]) / float(self.n_samples)
        return agg


class SortAndSearch(BenchPred):
    """
    Sort & Search (S&S) Efficient Lifelong Model Evaluation
    Implements insertM (efficiently evaluating new models)
    """

    def __init__(self, ranking_mode="sum"):
        super().__init__()
        self.ranking_mode = ranking_mode
        self.order = None
        self.coreset_idx = None
        self.n_samples = None

    def fit(self, source_full_scores, coreset_size, seed=42, *args, **kwargs):
        """
        source_full_scores: numpy array [num_models, num_samples], binary {0,1}
        coreset_size: Number of samples to use as coreset (n')
        """
        assert source_full_scores.ndim == 2
        n_models, n_samples = source_full_scores.shape
        assert coreset_size > 1 and coreset_size < n_samples

        set_random_seed(seed)
        self.n_samples = n_samples

        # 1. Sort: get global ranking order of samples
        if self.ranking_mode == "sum":
            sum_ranking = source_full_scores.sum(axis=0)
            self.order = np.flip(np.argsort(sum_ranking))
        elif self.ranking_mode == "recursive_sum":
            self.order = self._recursive_sum_ranking(source_full_scores)
        else:
            raise ValueError(f"Unknown ranking_mode: {self.ranking_mode}")

        # 2. Search: pick coreset indices (uniform sampling over sorted samples)
        self.coreset_idx = self._uniform_sampling(n_samples, coreset_size)

    def get_coreset(self):
        """
        Returns the indices of the coreset (to be evaluated by the new model)
        """
        return self.order[self.coreset_idx]

    def predict(self, target_coreset_scores):
        """
        Predicts average accuracy for one or more models.

        target_coreset_scores: shape [num_models, coreset_size] or [coreset_size]
        Returns: array of shape [num_models,] (or scalar if single model)
        """
        arr = np.array(target_coreset_scores, dtype=np.float32)
        single_input = False
        if arr.ndim == 1:
            arr = arr[None, :]
            single_input = True

        num_models = arr.shape[0]
        agg_accuracies = np.zeros(num_models, dtype=np.float32)
        for i in range(num_models):
            coreset_pred = arr[i]
            # DP-search: treat 0 as -1, 1 as +1
            pred = coreset_pred.copy()
            pred[pred == 0] = -1
            cumsum = np.cumsum(pred)
            idx = np.argmax(cumsum)
            threshold_full = self.coreset_idx[idx]
            agg_accuracies[i] = float(threshold_full) / float(self.n_samples)
        if single_input:
            return agg_accuracies[0]
        return agg_accuracies

    def refit_regressor(self, source_full_scores):
        return _SortSearchPredictor(self.coreset_idx, self.n_samples)

    def save(self, path_save):
        with open(path_save, "wb") as f:
            pickle.dump(
                {
                    "order": self.order,
                    "coreset_idx": self.coreset_idx,
                    "n_samples": self.n_samples,
                    "ranking_mode": self.ranking_mode,
                },
                f,
            )

    def load(self, path_load):
        with open(path_load, "rb") as f:
            state = pickle.load(f)
            self.order = state["order"]
            self.coreset_idx = state["coreset_idx"]
            self.n_samples = state["n_samples"]
            self.ranking_mode = state["ranking_mode"]

    @staticmethod
    def _uniform_sampling(query_len, num_queries):
        # Uniformly sample indices over [0, query_len)
        step = query_len // num_queries
        start = (
            step // 2
            if query_len == step * num_queries
            else (query_len - step * num_queries)
        )
        sampled_points = np.arange(start, query_len, step)
        assert len(sampled_points) == num_queries
        return sampled_points

    @staticmethod
    def _dynamic_programming_threshold(A):
        # A: shape [num_models, num_samples]
        # For each model (row), find the threshold index maximizing cumsum (with 0 -> -1)
        Aopt = A.copy()
        Aopt[Aopt == 0] = -1
        idx = np.argmax(np.cumsum(Aopt, axis=1), axis=1)
        return idx  # shape: [num_models]

    def _recursive_sum_ranking(self, A):
        idx = np.arange(A.shape[0])
        # Two step approximation works well enough
        sum_bins = A[idx].sum(axis=0)
        order = np.flip(np.argsort(sum_bins))

        # an array of size m --> indexes of thresholds for each model in the ordered matrix
        thresh_ordered = self._dynamic_programming_threshold(A[:, order])
        # permute to fix ordering
        sum_bins_ordered = sum_bins[order]
        uniq_bins = np.unique(sum_bins_ordered)

        # we look at each bin, take all the thresholds that lie within each bin, and take the model sums for those thresholds, and then compute the sum again
        for bin in uniq_bins:
            idx = np.nonzero(sum_bins_ordered == bin)[0]
            # find thresh in this idx
            # across models
            thresh_idx = np.nonzero(
                np.all(
                    [[thresh_ordered >= idx.min()], [thresh_ordered <= idx.max()]],
                    axis=0,
                )
            )[1]
            A_New = A[thresh_idx][:, order[idx]]
            improved_bins = A_New.sum(axis=0)
            # new ordering within current bin
            new_order = np.flip(np.argsort(improved_bins))
            # order withing current bin
            order[idx] = order[idx[new_order]]
        return order


class SortAndSearchSum(SortAndSearch):
    def __init__(self):
        super().__init__(ranking_mode="sum")


class SortAndSearchRecursiveSum(SortAndSearch):
    def __init__(self):
        super().__init__(ranking_mode="recursive_sum")
