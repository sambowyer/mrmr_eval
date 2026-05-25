import numpy as np
import joblib as jbl
from scipy.special import psi
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from tqdm import tqdm

from .base import BenchPred, set_random_seed


class MRMRPred(BenchPred):
    """MRMR (Minimum Redundancy Maximum Relevance) coreset selection.

    Implements MID (relevance minus mean redundancy) and MIQ (relevance divided
    by redundancy) schemes.  Relevance is always measured against the per-model
    average benchmark score (``_y`` target).  Binary score matrices use the Ross
    (2014) estimator; continuous matrices use the PCA-corrected KSG (LNC)
    estimator.  A Ridge regressor is fit on the selected coreset to predict the
    average score.
    """

    def __init__(self):
        super().__init__()
        self.compressed_data_indices = None
        self.rgs = None
        self.miq_scheme = False  # False for MID, True for MIQ
        self.mi_k = 3  # k nearest neighbours for Ross MI estimator
        self.selection_metrics = None
        self._binary = None
        self.alpha_ = None
        self.alpha_range_ = None

    def _build_regressor(self, X: np.ndarray, y: np.ndarray):
        """Build and fit the prediction model.

        Override in subclasses to swap the regressor (e.g. GLM).

        Args:
            X: Feature matrix of shape (M, K)
            y: Target vector of shape (M,)

        Returns:
            Fitted regressor with a sklearn-compatible .predict() method.
        """
        # if self._binary:
        #     # rgs = Ridge(alpha=10)
        #     rgs = RidgeCV(alphas=np.logspace(-1, 1, 5))
        # else:
        #     rgs = RidgeCV(alphas=np.logspace(-2, 2, 9))
        alphas = (
            np.logspace(-1, 1, 5) if self._binary
            else np.logspace(-2, 2, 9)
        )
        rgs = RidgeCV(alphas=alphas)
        rgs.fit(X, y.reshape(-1, 1))
        self.alpha_ = float(rgs.alpha_)
        self.alpha_range_ = (float(alphas.min()), float(alphas.max()))
        return rgs

    @staticmethod
    def _mutual_information_discrete_discrete(x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Mutual Information between two binary response patterns.

        Args:
            x1: First binary array (0/1 valued)
            x2: Second binary array (0/1 valued)

        Returns:
            Mutual information value (non-negative)
        """
        n = len(x1)
        if n == 0:
            return 0.0

        s1 = x1.sum()
        s2 = x2.sum()
        s11 = np.dot(x1, x2)

        p1 = np.array([n - s1, s1]) / n
        p2 = np.array([n - s2, s2]) / n
        p_joint = np.array([
            [n - s1 - s2 + s11, s2 - s11],
            [s1 - s11,          s11],
        ]) / n

        mi = 0.0
        for i in range(2):
            for j in range(2):
                if p_joint[i, j] > 0:
                    mi += p_joint[i, j] * np.log(p_joint[i, j] / (p1[i] * p2[j] + 1e-15))
        return max(0.0, mi)

    @staticmethod
    def _mutual_information_discrete_discrete_batch(
        X: np.ndarray, x2: np.ndarray
    ) -> np.ndarray:
        """Vectorised MI between each column of X and a single vector x2 (all binary).

        Args:
            X: Binary matrix of shape (n_samples, n_features)
            x2: Single binary vector of shape (n_samples,)

        Returns:
            Array of MI values of shape (n_features,)
        """
        n = X.shape[0]
        if n == 0:
            return np.zeros(X.shape[1])

        x2 = x2.astype(np.float64)

        s1 = X.sum(axis=0)            # (n_features,)
        s2 = x2.sum()                  # scalar
        s11 = X.T @ x2                 # (n_features,)

        p1_1 = s1 / n
        p1_0 = 1.0 - p1_1
        p2_1 = s2 / n
        p2_0 = 1.0 - p2_1

        # 2x2 joint probabilities for each feature
        p_11 = s11 / n
        p_10 = p1_1 - p_11
        p_01 = p2_1 - p_11
        p_00 = 1.0 - p1_1 - p2_1 + p_11

        eps = 1e-15
        mi = np.zeros(X.shape[1])
        for p_ij, m_ij in [
            (p_00, p1_0 * p2_0),
            (p_01, p1_0 * p2_1),
            (p_10, p1_1 * p2_0),
            (p_11, p1_1 * p2_1),
        ]:
            mask = p_ij > 0
            mi[mask] += p_ij[mask] * np.log(p_ij[mask] / (m_ij[mask] + eps))

        return np.maximum(mi, 0.0)

    @staticmethod
    def _mutual_information_ross_estimator(
        x: np.ndarray, y: np.ndarray, k: int = 3
    ) -> float:
        """Ross (2014) estimator for MI between binary x and continuous y.

        Args:
            x: Binary array
            y: Continuous array
            k: Number of nearest neighbors to use

        Returns:
            Mutual information estimate (non-negative)
        """
        n = len(x)
        if n == 0:
            return 0.0

        y0 = y[x == 0].reshape(-1, 1)
        y1 = y[x == 1].reshape(-1, 1)
        nx0, nx1 = len(y0), len(y1)

        # Fallback if categories are too small for k-NN
        if nx0 < k + 1 or nx1 < k + 1:
            return 0.0

        tree0 = cKDTree(y0)
        tree1 = cKDTree(y1)
        tree_all = cKDTree(y.reshape(-1, 1))

        m_list = []
        for i in range(n):
            val = y[i].reshape(1, -1)
            tree_same = tree1 if x[i] == 1 else tree0
            # Get distance to kth neighbor in the SAME class
            dist, _ = tree_same.query(val, k=k + 1)

            # Use max distance to capture all neighbors within that radius
            max_dist = np.max(dist)
            # Count neighbors in ALL classes within the same radius
            m = tree_all.query_ball_point(
                val.reshape(1, -1), max_dist - 1e-15, return_length=True
            )
            m_list.append(m)

        avg_psi_nx = (nx0 * psi(nx0) + nx1 * psi(nx1)) / n
        avg_psi_m = np.mean([psi(m) for m in m_list])
        return max(0.0, psi(n) - avg_psi_nx + psi(k) - avg_psi_m)

    @staticmethod
    def _is_binary_scores(scores: np.ndarray) -> bool:
        """Check whether a score matrix contains only binary (0/1) values."""
        finite = scores[np.isfinite(scores)]
        return finite.size > 0 and len(np.unique(finite)) <= 2

    # ------------------------------------------------------------------
    # Gao et al. (AISTATS 2015) PCA-based Local Nonuniformity Correction
    # ------------------------------------------------------------------

    @staticmethod
    def _mutual_information_lnc(
        x: np.ndarray, y: np.ndarray, k: int = 5, alpha: float = 0.25
    ) -> float:
        """MI via KSG + PCA local nonuniformity correction (LNC).

        Gao, Ver Steeg & Galstyan (AISTATS 2015) correct the KSG
        estimate by replacing the axis-aligned max-norm rectangle with a
        PCA-aligned rectangle at each point.  When the PCA volume is
        much smaller than the max-norm volume (ratio < alpha), a
        positive correction is applied.

        Args:
            x: Continuous array of shape (n,).
            y: Continuous array of shape (n,).
            k: Number of nearest neighbours.
            alpha: Threshold for local nonuniformity test.

        Returns:
            Mutual information estimate (non-negative, nats).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)
        if n < k + 1:
            return 0.0

        intens = 1e-10
        x = x + intens * np.random.default_rng(0).standard_normal(n)
        y = y + intens * np.random.default_rng(1).standard_normal(n)

        xy = np.column_stack([x, y])
        tree = cKDTree(xy)
        dd, ii = tree.query(xy, k=k + 1, p=np.inf)
        eps = dd[:, -1]

        # Gather all neighbour patches: (n, k+1, 2)
        all_nbr = xy[ii]
        centres = all_nbr[:, 0:1, :]
        centred = all_nbr - centres

        # Per-axis max distances (needed for LNC rectangle volume)
        dvec_x = np.maximum(np.max(np.abs(centred[:, :, 0]), axis=1), 1e-15)
        dvec_y = np.maximum(np.max(np.abs(centred[:, :, 1]), axis=1), 1e-15)

        # KSG1 MI — use eps (Chebyshev distance) for both marginal counts
        r = np.maximum(eps - 1e-15, 0.0)
        tree_x = cKDTree(x.reshape(-1, 1))
        tree_y = cKDTree(y.reshape(-1, 1))
        nx = tree_x.query_ball_point(
            x.reshape(-1, 1), r, p=np.inf, return_length=True
        ) - 1
        ny = tree_y.query_ball_point(
            y.reshape(-1, 1), r, p=np.inf, return_length=True
        ) - 1
        nx = np.maximum(nx, 1)
        ny = np.maximum(ny, 1)
        mi_ksg = float(
            psi(k) + psi(n) - np.mean(psi(nx + 1) + psi(ny + 1))
        )

        # LNC correction — vectorised
        nbr_only = centred[:, 1:, :]                          # (n, k, 2)
        cov = np.einsum('nki,nkj->nij', nbr_only, nbr_only)  # (n, 2, 2)
        cov /= k

        _, eigvecs = np.linalg.eigh(cov)                      # (n, 2, 2)
        projected = np.einsum('nki,nij->nkj', centred, eigvecs)  # (n, k+1, 2)
        max_proj = np.maximum(np.max(np.abs(projected), axis=1), 1e-30)  # (n, 2)

        log_V_pca = np.sum(np.log(max_proj), axis=1)          # (n,)
        log_V_rect = np.log(dvec_x) + np.log(dvec_y)          # (n,)

        log_alpha = np.log(max(alpha, 1e-30))
        lnc_mask = log_V_pca < log_V_rect + log_alpha
        correction = np.sum(log_V_rect[lnc_mask] - log_V_pca[lnc_mask]) / n

        return max(0.0, mi_ksg + correction)

    @staticmethod
    def _mutual_information_lnc_batch(
        X: np.ndarray, x2: np.ndarray, k: int = 5, alpha: float = 0.25
    ) -> np.ndarray:
        """Batch LNC MI between each column of X and a single vector x2.

        Args:
            X: Matrix of shape (n_samples, n_features).
            x2: Vector of shape (n_samples,).
            k: Number of nearest neighbours.
            alpha: LNC threshold parameter.

        Returns:
            Array of MI values of shape (n_features,).
        """
        n_features = X.shape[1]
        mi_values = np.empty(n_features)
        for j in range(n_features):
            mi_values[j] = MRMRPred._mutual_information_lnc(
                X[:, j], x2, k=k, alpha=alpha
            )
        return mi_values

    def _get_mi_estimators(self, _binary: bool):
        """Return (relevance_fn, redundancy_batch_fn) for MI estimation.

        Binary score matrices use the Ross (2014) estimator + a fast
        discrete-discrete batch estimator.  Continuous score matrices use
        the PCA-corrected KSG (Local Nonuniformity Correction) of Gao et
        al. (AISTATS 2015), which gives the cleanest MI estimates on the
        clustered, low-entropy outputs typical of LLM evaluations.

        Args:
            _binary: True when the score matrix contains only binary values.

        Returns:
            Tuple of (relevance_fn, redundancy_batch_fn).
            - relevance_fn(x, y, k=...) -> float
            - redundancy_batch_fn(X, x2) -> np.ndarray
        """
        if _binary:
            return (
                self._mutual_information_ross_estimator,
                self._mutual_information_discrete_discrete_batch,
            )
        return (
            lambda x, y, k=self.mi_k: self._mutual_information_lnc(x, y, k),
            lambda X, x2: self._mutual_information_lnc_batch(X, x2, k=self.mi_k),
        )

    def fit(
        self,
        source_full_scores: np.ndarray,
        coreset_size: int,
        seed: int = 42,
        miq_scheme: bool = False,
        only_relevance: bool = False,
        *args,
        **kwargs,
    ) -> "MRMRPred":
        """Fit MRMR feature selection and Ridge regression model.

        Relevance is always measured against the per-model average benchmark
        score (the ``_y`` target).  The fitted regressor also predicts that
        same average score from the selected coreset.

        Args:
            source_full_scores: Matrix of shape (M models, N questions)
            coreset_size: Number of features to select
            seed: Random seed for reproducibility
            miq_scheme: False for MID, True for MIQ scheme
            only_relevance: If True, greedily maximise relevance (MI with target
                only); redundancy is neither computed nor used.

        Returns:
            self
        """
        # Validate input
        if source_full_scores is None:
            raise ValueError("source_full_scores cannot be None")

        if len(source_full_scores.shape) < 2:
            raise ValueError("source_full_scores must be a 2D array")

        num_model, num_data = source_full_scores.shape

        if num_model <= 1:
            raise ValueError("Need at least 2 models")
        if num_data <= 1:
            raise ValueError("Need at least 2 data points")
        if coreset_size <= 1:
            raise ValueError("coreset_size must be > 1")
        if num_data <= coreset_size:
            raise ValueError("num_data must be > coreset_size")

        set_random_seed(seed)
        self.miq_scheme = miq_scheme
        self.only_relevance = only_relevance

        # Detect binary vs continuous features and choose MI estimators
        _binary = self._is_binary_scores(source_full_scores)
        self._binary = _binary

        # Mean imputation if there are any NaNs
        if not _binary and np.any(np.isnan(source_full_scores)):
            source_full_scores = source_full_scores.copy()
            col_means = np.nanmean(source_full_scores, axis=0)
            inds = np.where(np.isnan(source_full_scores))
            source_full_scores[inds] = col_means[inds[1]]

        _mi_rel, _mi_red_batch = self._get_mi_estimators(_binary)

        # Relevance target and regression target are both mean benchmark score.
        relevance_target = source_full_scores.mean(-1)
        regression_target = relevance_target

        # Precompute relevance for every point (MI with relevance_target); it does not change during selection
        relevance_per_idx = np.array(
            [
                _mi_rel(
                    source_full_scores[:, idx], relevance_target, k=self.mi_k
                )
                for idx in tqdm(
                    range(num_data), desc="Relevance (MI)", unit="point"
                )
            ]
        )

        if only_relevance:
            remaining_set: set[int] = set(range(num_data))
            selected_indices_flat: list[int] = []
            coreset_rel_only: list[float] = []
            coreset_red_only: list[float] = []
            for _ in tqdm(
                range(coreset_size),
                desc="MRMR relevance-only",
                unit="feature",
            ):
                rem_arr = np.array(sorted(remaining_set))
                pick = int(rem_arr[int(np.argmax(relevance_per_idx[rem_arr]))])
                selected_indices_flat.append(pick)
                remaining_set.remove(pick)
                coreset_rel_only.append(float(relevance_per_idx[pick]))
                coreset_red_only.append(0.0)
            self.compressed_data_indices = np.array(selected_indices_flat)
            self.selection_metrics = {
                "all_relevance_min": float(np.min(relevance_per_idx)),
                "all_relevance_max": float(np.max(relevance_per_idx)),
                "all_relevance_mean": float(np.mean(relevance_per_idx)),
                "all_relevance_median": float(np.median(relevance_per_idx)),
                "all_relevance_std": float(np.std(relevance_per_idx)),
                "coreset_relevance": coreset_rel_only,
                "coreset_redundancy": coreset_red_only,
            }
            X = source_full_scores[:, self.compressed_data_indices]
            self.rgs = self._build_regressor(X, regression_target)
            return self

        # Global relevance statistics (across ALL questions)
        coreset_relevance = []
        coreset_redundancy = []

        # Initialize selection variables
        selected_indices = []
        remaining_indices = set(range(num_data))

        # Running sum of MI(idx, selected) for each candidate; updated incrementally
        redundancy_sum = np.zeros(num_data)

        # First iteration - select feature with highest relevance to target
        best_idx = int(np.argmax(relevance_per_idx))
        selected_indices.append(best_idx)
        remaining_indices.discard(best_idx)
        coreset_relevance.append(float(relevance_per_idx[best_idx]))
        coreset_redundancy.append(0.0)

        # Update redundancy sums: add MI(each remaining, newly selected) for the first selection
        remaining_arr = np.array(sorted(remaining_indices))
        redundancy_sum[remaining_arr] += _mi_red_batch(
            source_full_scores[:, remaining_arr], source_full_scores[:, best_idx]
        )

        # Greedy selection of remaining features
        for _ in tqdm(
            range(1, coreset_size),
            desc="MRMR selection",
            unit="feature",
        ):
            num_selected = len(selected_indices)
            remaining_arr = np.array(sorted(remaining_indices))

            mean_red = redundancy_sum[remaining_arr] / num_selected
            rel = relevance_per_idx[remaining_arr]
            if self.miq_scheme:
                scores = rel / (mean_red + 1e-10)
            else:
                scores = rel - mean_red

            winner_pos = np.argmax(scores)
            winner = remaining_arr[winner_pos]
            best_idx = int(winner)
            selected_indices.append(best_idx)
            remaining_indices.discard(best_idx)
            coreset_relevance.append(float(rel[winner_pos]))
            coreset_redundancy.append(float(mean_red[winner_pos]))

            remaining_arr = np.array(sorted(remaining_indices))
            if len(remaining_arr) > 0:
                redundancy_sum[remaining_arr] += _mi_red_batch(
                    source_full_scores[:, remaining_arr], source_full_scores[:, best_idx]
                )

        self.compressed_data_indices = np.array(selected_indices)
        self.selection_metrics = {
            "all_relevance_min": float(np.min(relevance_per_idx)),
            "all_relevance_max": float(np.max(relevance_per_idx)),
            "all_relevance_mean": float(np.mean(relevance_per_idx)),
            "all_relevance_median": float(np.median(relevance_per_idx)),
            "all_relevance_std": float(np.std(relevance_per_idx)),
            "coreset_relevance": coreset_relevance,
            "coreset_redundancy": coreset_redundancy,
        }

        # Train regressor on selected features (always predicting mean score)
        X = source_full_scores[:, self.compressed_data_indices]
        self.rgs = self._build_regressor(X, regression_target)

        return self

    def get_coreset(self) -> np.ndarray:
        """Get the selected feature indices.

        Returns:
            Array of selected feature indices
        """
        if self.compressed_data_indices is None:
            raise ValueError("Model has not been fitted yet")
        return self.compressed_data_indices

    def predict(self, target_coreset_scores: np.ndarray) -> np.ndarray:
        """Predict using the trained Ridge model.

        Args:
            target_coreset_scores: Matrix of shape (M_target models, K coreset features)

        Returns:
            Predicted scores
        """
        if target_coreset_scores is None:
            raise ValueError("target_coreset_scores cannot be None")

        if len(target_coreset_scores.shape) == 1:
            target_coreset_scores = target_coreset_scores.reshape(1, -1)

        return self.rgs.predict(target_coreset_scores).ravel()

    def refit_regressor(self, source_full_scores):
        coreset = self.get_coreset()
        X = source_full_scores[:, coreset]
        y = source_full_scores.mean(axis=1)
        return self._build_regressor(X, y)

    def save(self, path_save: str) -> "MRMRPred":
        """Save the model to disk.

        Args:
            path_save: Path to save the model

        Returns:
            self
        """
        if path_save is None:
            raise ValueError("path_save cannot be None")

        jbl.dump((self.compressed_data_indices, self.rgs), path_save)
        return self

    def load(self, path_load: str) -> "MRMRPred":
        """Load the model from disk.

        Args:
            path_load: Path to load the model from

        Returns:
            self
        """
        if path_load is None:
            raise ValueError("path_load cannot be None")

        self.compressed_data_indices, self.rgs = jbl.load(path_load)
        return self


class MRMRPred_MID_y(MRMRPred):
    """MRMR with MID scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["miq_scheme"] = False
        return super().fit(*args, **kwargs)


class MRMRPred_MIQ_y(MRMRPred):
    """MRMR with MIQ scheme and average score as target."""

    def fit(self, *args, **kwargs):
        kwargs["miq_scheme"] = True
        return super().fit(*args, **kwargs)


class MRMRPred_MI_y(MRMRPred):
    """MRMR relevance-only: greedily maximise MI to the target (no redundancy)."""

    def fit(self, *args, **kwargs):
        kwargs["only_relevance"] = True
        return super().fit(*args, **kwargs)


# ---------------------------------------------------------------------------
# Variant factories: k-NN MI variants (different `mi_k`) and Kernel Ridge
# Regression (KRR) variants.  These programmatically generate subclasses of
# every public MRMR class so the registry can expose mrmr<k>_<scheme>_<tgt>
# and kmrmr<k>_<scheme>_<tgt> names without hand-writing each class.
# ---------------------------------------------------------------------------

_MRMR_BASE_CLASSES = [
    MRMRPred,
    MRMRPred_MID_y,
    MRMRPred_MIQ_y,
    MRMRPred_MI_y,
]


def _make_mi_k_variant(base_cls, k):
    """Create a subclass that overrides ``mi_k`` for the MI estimator."""

    class _Variant(base_cls):
        def __init__(self):
            super().__init__()
            self.mi_k = k

    new_name = base_cls.__name__.replace("MRMR", f"MRMR{k}", 1)
    _Variant.__name__ = new_name
    _Variant.__qualname__ = new_name
    _Variant.__module__ = __name__
    _Variant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(using k={k} MI neighbours)"
    )
    return _Variant


for _base_cls in _MRMR_BASE_CLASSES:
    for _k in (4, 5, 6, 7, 8, 9):
        _variant = _make_mi_k_variant(_base_cls, _k)
        globals()[_variant.__name__] = _variant


def _make_krr_variant(base_cls, degree=2):
    """Create a subclass that uses Kernel Ridge Regression instead of Ridge."""

    _degree = degree  # capture for closure

    class _KRRVariant(base_cls):
        def _build_regressor(self, X, y):
            alphas = (
                np.logspace(-1, 1, 5) if self._binary
                else np.logspace(-2, 2, 9)
            )
            rgs = GridSearchCV(
                KernelRidge(kernel="poly", degree=_degree, coef0=1),
                param_grid={"alpha": alphas},
                cv=LeaveOneOut(),
                scoring="neg_mean_squared_error",
            )
            rgs.fit(X, y.ravel())
            self.alpha_ = float(rgs.best_params_["alpha"])
            self.alpha_range_ = (float(alphas.min()), float(alphas.max()))
            return rgs

    suffix = "" if _degree == 2 else str(_degree)
    new_name = base_cls.__name__.replace("MRMR", f"K{suffix}MRMR", 1)
    _KRRVariant.__name__ = new_name
    _KRRVariant.__qualname__ = new_name
    _KRRVariant.__module__ = __name__
    _KRRVariant.__doc__ = (
        f"{(base_cls.__doc__ or base_cls.__name__).strip()} "
        f"(using Kernel Ridge Regression with degree-{_degree} polynomial kernel)"
    )
    return _KRRVariant
