import numpy as np
import pickle
from scipy.special import expit
from scipy.optimize import minimize
from pygam import LinearGAM, s
from tqdm import tqdm
from .base import BenchPred


def fit_irt(X, irt_type="2PL", n_iters=2):
    """
    Fit a 2PL IRT model via alternating optimization.
    X: (n_subjects, n_items) binary response matrix
    irt_type is ignored (only 2PL supported)
    n_iters: number of alternations between items and abilities

    Returns:
      params: dict with
        'a': np.ndarray of length m  (discriminations, clipped ≥0.01)
        'b': np.ndarray of length m  (difficulties)
    """
    n, m = X.shape

    # initialize parameters
    a = np.ones(m)
    b = np.zeros(m)
    theta = np.zeros(n)

    # small constant for numerical stability
    eps = 1e-8

    # negative log‐lik for all items, used to update a and b jointly
    def all_items_nll(ab_flat):
        ai = ab_flat[:m]
        bi = ab_flat[m:]
        th = theta[:, None]
        P = expit(ai[None, :] * (th - bi[None, :]))
        # sum over subjects and items
        return -np.sum(X * np.log(P + eps) + (1 - X) * np.log(1 - P + eps))

    # negative log‐lik for abilities given fixed a, b
    def theta_nll(t_i, i):
        p = expit(a * (t_i - b))
        xi = X[i]
        return -np.sum(xi * np.log(p + eps) + (1 - xi) * np.log(1 - p + eps))

    for _ in tqdm(range(n_iters)):
        # M-step: optimize all item params (a and b) jointly
        init = np.concatenate([a, b])
        bounds = [(0.01, 5.0)] * m + [(-4.0, 4.0)] * m
        res = minimize(all_items_nll, init, method="L-BFGS-B", bounds=bounds)
        ab_opt = res.x
        a = np.clip(ab_opt[:m], 0.01, 5.0)
        b = ab_opt[m:]

        # E-step: optimize each theta_i separately
        for i in range(n):
            res_i = minimize(
                lambda t: theta_nll(t, i),
                x0=theta[i],
                bounds=[(-4.0, 4.0)],
                method="L-BFGS-B",
            )
            theta[i] = res_i.x

    return {"a": a, "b": b}


def estimate_abilities(params, X_sub, method="MAP"):
    """
    Estimate subject abilities for a fitted 2PL model.
    params: dict with 'a' and 'b' for m_sub items
    X_sub: (n_subjects, m_sub) binary responses
    method: 'MAP' or 'ML' (flat prior)
    Returns: theta_est of length n_subjects
    """
    a = params["a"]
    b = params["b"]
    n, m = X_sub.shape
    eps = 1e-8
    theta_hat = np.zeros(n)

    def negloglik(th, xi):
        p = expit(a * (th - b))
        return -np.sum(xi * np.log(p + eps) + (1 - xi) * np.log(1 - p + eps))

    for i in range(n):
        xi = X_sub[i]
        res = minimize(
            negloglik, x0=0.0, args=(xi,), bounds=[(-4.0, 4.0)], method="L-BFGS-B"
        )
        theta_hat[i] = res.x

    return theta_hat


def compute_fisher_information(params, theta_grid):
    """
    Compute Fisher information for each item over a grid of θ.

    Args:
      params : dict with
        - params["a"]: array of item discriminations, shape (m,)
        - params["b"]: array of item difficulties,     shape (m,)
      theta_grid : 1D array of θ values, shape (T,)

    Returns:
      info : np.ndarray of shape (m, T), where
        info[i, t] = a[i]^2 * p * (1 - p),  p = σ(a[i]*(θ_t - b[i]))
    """
    a = params["a"]  # shape (m,)
    b = params["b"]  # shape (m,)
    theta = theta_grid  # shape (T,)

    # Broadcast to compute p_{i,t} = σ(a_i * (θ_t - b_i))
    # a[:, None] shape (m,1), theta[None,:] (1,T), b[:,None] (m,1)
    logits = a[:, None] * (theta[None, :] - b[:, None])
    p = expit(logits)

    # Fisher info: a^2 * p * (1 - p)
    info = (a[:, None] ** 2) * p * (1 - p)
    return info


class _MetaBenchPredictor:
    def __init__(self, coreset_params, ability_method, gam):
        self.coreset_params = coreset_params
        self.ability_method = ability_method
        self.gam = gam

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        theta_new = estimate_abilities(
            self.coreset_params, X, method=self.ability_method
        )
        Xnew = np.vstack([theta_new, X.mean(axis=1)]).T
        return self.gam.predict(Xnew)


class MetaBench(BenchPred):
    def __init__(self):
        super().__init__()
        self.irt_type = "2PL"
        self.ability_est_method = "MAP"
        self.full_params = None
        self.theta_train = None
        self.coreset_items = None
        self.coreset_params = None
        self.gam = None

    def fit(
        self,
        source_full_scores,  # np.ndarray shape (n_models, n_items), binary
        coreset_size,
        seed=42,
        irt_type="2PL",
        ability_est_method="MAP",
    ):
        np.random.seed(seed)
        self.irt_type = irt_type
        self.ability_est_method = ability_est_method.upper()

        # 1) Fit full-bank IRT
        full_params = fit_irt(source_full_scores, irt_type=irt_type)
        self.full_params = full_params

        # 2) Estimate abilities on full bank (training set)
        theta = estimate_abilities(
            full_params, source_full_scores, method=self.ability_est_method
        )
        self.theta_train = theta

        # 3) Partition theta_train into quantiles, select most informative item per bin
        quantiles = np.quantile(theta, np.linspace(0, 1, coreset_size + 1))
        fis = compute_fisher_information(full_params, theta)
        # shape (n_items, n_models)
        selected = set()
        for i in range(coreset_size):
            lo, hi = quantiles[i], quantiles[i + 1]
            mask = (theta >= lo) & (theta < hi)
            if not np.any(mask):
                continue
            avg_info = fis[:, mask].mean(axis=1)  # per item
            for idx in np.argsort(avg_info)[::-1]:
                if idx not in selected:
                    selected.add(idx)
                    break
            if len(selected) >= coreset_size:
                break

        self.coreset_items = np.array(sorted(selected))

        # 4) Extract coreset params
        sub_scores = source_full_scores[:, self.coreset_items]
        co_params = {
            "a": full_params["a"][self.coreset_items],
            "b": full_params["b"][self.coreset_items],
        }
        self.coreset_params = co_params

        # 5) Re-estimate abilities on coreset
        theta_co = estimate_abilities(
            co_params, sub_scores, method=self.ability_est_method
        )

        # 6) Fit a 2-term GAM:  y = mean(full_scores) ~ s(theta_co) + s(mean(sub_scores))
        y = source_full_scores.mean(axis=1)
        X = np.vstack([theta_co, sub_scores.mean(axis=1)]).T
        self.gam = LinearGAM(s(0) + s(1), fit_intercept=True).fit(X, y)

    def get_coreset(self):
        return self.coreset_items

    def predict(self, target_coreset_scores):
        # target_coreset_scores: shape (n_new, m_co)
        theta_new = estimate_abilities(
            self.coreset_params, target_coreset_scores, method=self.ability_est_method
        )
        Xnew = np.vstack([theta_new, target_coreset_scores.mean(axis=1)]).T
        return self.gam.predict(Xnew)

    def refit_regressor(self, source_full_scores):
        coreset = self.get_coreset()
        sub_scores = source_full_scores[:, coreset]
        theta_co = estimate_abilities(
            self.coreset_params, sub_scores, method=self.ability_est_method
        )
        y = source_full_scores.mean(axis=1)
        X_train = np.vstack([theta_co, sub_scores.mean(axis=1)]).T
        gam = LinearGAM(s(0) + s(1), fit_intercept=True).fit(X_train, y)
        return _MetaBenchPredictor(self.coreset_params, self.ability_est_method, gam)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "irt_type": self.irt_type,
                    "ability_est_method": self.ability_est_method,
                    "full_params": self.full_params,
                    "theta_train": self.theta_train,
                    "coreset_items": self.coreset_items,
                    "coreset_params": self.coreset_params,
                    "gam": self.gam,
                },
                f,
            )

    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.__dict__.update(d)
