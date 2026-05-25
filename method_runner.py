"""MethodRunner: orchestrate (dataset x method x seed) trials.

Slimmed from the paper repo's ``method_runner.py``: drops cross-k / pass@k
back-fill plumbing and the ``extrapolation``/``easier_extrapolation``/
``stratified``/``timing`` model-split methods.  Keeps the multi-process
worker pool, the lightweight tqdm proxy used inside workers, and the
result-directory layout::

    {dir_results}/{split_method}/coreset_{size}/nmodels_{label}/
        {dataset}/{method}/{seed}/{result.jbl, ckpt.jbl, record*}
"""

import os
import sys
import time
import logging
import warnings
import threading
import traceback
import numpy as np
import joblib as jbl
from joblib import Parallel, delayed
from typing import Dict, List, Tuple
from multiprocessing import cpu_count, Manager
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message=".*CUDA initialization.*",
    category=UserWarning,
    module=r"torch\.cuda",
)

from scipy.stats import spearmanr, pearsonr, kendalltau
from zarth_utils.config import Config
from zarth_utils.recorder import Recorder
from zarth_utils.nn_utils import set_random_seed, get_all_paths
from zarth_utils.timer import Timer

from benchpred import all_methods
from data_loader import LoaderRegistry

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _get_error_logger():
    """Return a module-level logger that writes ERROR messages to logs/errors.log."""
    logger = logging.getLogger("method_runner_errors")
    if not logger.handlers:
        os.makedirs(_LOG_DIR, exist_ok=True)
        fh = logging.FileHandler(os.path.join(_LOG_DIR, "errors.log"))
        fh.setLevel(logging.ERROR)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.ERROR)
    return logger


class _TqdmProxy:
    """Lightweight tqdm stand-in for worker processes.

    Instead of writing progress bars to the terminal (which causes ANSI
    escape-code conflicts across processes), this records progress into a
    shared ``Manager().dict()`` that the main process polls to render bars.
    Updates to the shared dict are rate-limited to avoid IPC overhead.
    """

    _MIN_INTERVAL = 0.1

    def __init__(self, iterable=None, desc='', total=None,
                 _progress_dict=None, _worker_id=None, **kwargs):
        self.iterable = iterable
        self.total = total
        if self.total is None and hasattr(iterable, '__len__'):
            self.total = len(iterable)
        self.desc = desc
        self.n = 0
        self._postfix = ''
        self._progress_dict = _progress_dict
        self._worker_id = _worker_id
        self._last_report = 0
        self._report(force=True)

    def _report(self, force=False):
        if self._progress_dict is None or self._worker_id is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_report) < self._MIN_INTERVAL:
            return
        self._last_report = now
        try:
            self._progress_dict[self._worker_id] = {
                'n': self.n, 'total': self.total, 'desc': self.desc,
                'postfix': self._postfix,
            }
        except Exception:
            pass

    def __iter__(self):
        if self.iterable is None:
            return
        for obj in self.iterable:
            yield obj
            self.n += 1
            self._report()
        self._report(force=True)

    def update(self, n=1):
        self.n += n
        self._report()

    def set_description(self, desc=None, refresh=True):
        self.desc = desc or ''
        if refresh:
            self._report(force=True)

    def set_postfix(self, *args, **kwargs):
        pass

    def set_postfix_str(self, s='', refresh=True):
        self._postfix = s
        if refresh:
            self._report(force=True)

    def close(self):
        self._report(force=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _SilentTqdm:
    """No-op tqdm stand-in for compact multiprocessing mode.

    Workers still patch ``benchpred`` modules to use this class so inner
    method loops do not emit per-trial progress bars (which clutter notebooks
    and fight with Jupyter's output handling).
    """

    def __init__(self, iterable=None, desc='', total=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        if self.iterable is None:
            return
        yield from self.iterable

    def update(self, n=1):
        pass

    def set_description(self, desc=None, refresh=True):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def set_postfix_str(self, s='', refresh=True):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MethodRunner:
    """Runner for benchmark methods across datasets."""

    def __init__(self, config: Config):
        self.config = config
        self.loader = LoaderRegistry.get_loader(config.data_source)

        num_train_models = getattr(config, "num_train_models", "default")
        nmodels_label = (
            "nmodels_default"
            if num_train_models == "default"
            else f"nmodels_{num_train_models}"
        )
        self.results_base = os.path.join(
            config.dir_results,
            config.model_split_method,
            f"coreset_{config.coreset_size}",
            nmodels_label,
        )

    @staticmethod
    def resolve_coreset_size(coreset_size, num_data: int) -> int:
        """Resolve ``coreset_size`` to an integer.

        Accepts an int, a numeric string (e.g. ``"100"``), or a percentage
        string (e.g. ``"10%"``) which is interpreted as a fraction of
        ``num_data``.
        """
        coreset_str = str(coreset_size)
        if coreset_str.endswith("%"):
            pct = float(coreset_str[:-1])
            return max(1, int(pct / 100.0 * num_data))
        return int(coreset_str)

    @staticmethod
    def _binned_interpolation_split(num_models, true_acc, num_train_models,
                                    num_bins=10):
        """Pick source models with equal representation across accuracy bins.

        25% of models become source by default; otherwise ``num_train_models``
        specifies the source count.  The remainder become target models.
        """
        if num_train_models == "default":
            total_train = int(0.25 * num_models)
        else:
            total_train = int(num_train_models)

        min_acc, max_acc = float(true_acc.min()), float(true_acc.max())
        bin_edges = np.linspace(min_acc, max_acc + 1e-10, num_bins + 1)
        bin_ids = np.digitize(true_acc, bin_edges) - 1
        bin_ids = np.clip(bin_ids, 0, num_bins - 1)

        bins = [[] for _ in range(num_bins)]
        for idx in range(num_models):
            bins[bin_ids[idx]].append(idx)
        for b in bins:
            np.random.shuffle(b)

        non_empty = [i for i in range(num_bins) if bins[i]]
        per_bin = total_train // len(non_empty)
        remainder = total_train - per_bin * len(non_empty)

        bonus_bins = set(np.random.choice(
            len(non_empty), size=remainder, replace=False
        )) if remainder > 0 else set()
        quotas = {}
        for j, bi in enumerate(non_empty):
            extra = 1 if j in bonus_bins else 0
            quotas[bi] = min(per_bin + extra, len(bins[bi]))

        allocated = sum(quotas.values())
        while allocated < total_train:
            expandable = [i for i in non_empty if quotas[i] < len(bins[i])]
            if not expandable:
                break
            deficit = total_train - allocated
            extra_per = deficit // len(expandable)
            extra_rem = deficit % len(expandable)
            bonus_exp = set(np.random.choice(
                len(expandable), size=extra_rem, replace=False
            )) if extra_rem > 0 else set()
            for j, bi in enumerate(expandable):
                add = extra_per + (1 if j in bonus_exp else 0)
                add = min(add, len(bins[bi]) - quotas[bi])
                quotas[bi] += add
            allocated = sum(quotas.values())

        source_models = []
        for bi in non_empty:
            source_models.extend(bins[bi][: quotas[bi]])
        source_set = set(source_models)
        target_models = [i for i in range(num_models) if i not in source_set]
        return source_models, target_models

    def _compute_split(self, num_models, true_acc, num_train_models,
                       model_names=None):
        """Compute source/target model split for the current RNG state."""
        if self.config.model_split_method == "interpolation":
            num_target_models = int(0.25 * num_models)
            target_models = list(
                np.random.permutation(num_models)[:num_target_models]
            )
            source_models = [
                i for i in range(num_models) if i not in target_models
            ]
        elif self.config.model_split_method == "binned_interpolation":
            source_models, target_models = self._binned_interpolation_split(
                num_models, true_acc, num_train_models
            )
        else:
            raise NotImplementedError(
                f"Unknown split method: {self.config.model_split_method}.  "
                "The minimal mrmr_eval repo only supports 'interpolation' "
                "and 'binned_interpolation'."
            )

        if (
            num_train_models != "default"
            and self.config.model_split_method != "binned_interpolation"
        ):
            n = int(num_train_models)
            if n < len(source_models):
                source_models = list(
                    np.random.choice(source_models, size=n, replace=False)
                )

        return source_models, target_models

    @staticmethod
    def _run_single_trial(method_name, seed, scores, source_models, target_models,
                          true_acc, coreset_size, results_base, dataset_name,
                          exp_suffix, config_dict, use_git,
                          tqdm_position_queue=None, progress_dict=None,
                          mp_progress_mode="terminal"):
        """Execute a single ``(method, seed)`` trial.  Safe for multiprocessing."""
        tqdm_pos = None
        _tqdm_func = None
        if tqdm_position_queue is not None:
            tqdm_pos = tqdm_position_queue.get()
            compact = mp_progress_mode == "compact"

            if compact:
                def _make_tqdm(*args, **kwargs):
                    return _SilentTqdm(*args, **kwargs)
            else:
                _trial_label = f"[W{tqdm_pos}] {method_name}"

                def _make_tqdm(*args, **kwargs):
                    inner_desc = kwargs.get('desc', '')
                    kwargs['desc'] = (
                        f"{_trial_label}: {inner_desc}" if inner_desc
                        else _trial_label
                    )
                    kwargs['_progress_dict'] = progress_dict
                    kwargs['_worker_id'] = tqdm_pos
                    return _TqdmProxy(*args, **kwargs)

            _tqdm_func = _make_tqdm

            for _name, _mod in list(sys.modules.items()):
                if (_mod is not None
                        and _name.startswith('benchpred')
                        and callable(getattr(_mod, 'tqdm', None))):
                    _mod.tqdm = _make_tqdm

        try:
            return MethodRunner._do_single_trial(
                method_name, seed, scores, source_models, target_models,
                true_acc, coreset_size, results_base, dataset_name,
                exp_suffix, config_dict, use_git, _tqdm_func=_tqdm_func,
            )
        except Exception:
            _get_error_logger().error(
                "Trial FAILED | method=%s | dataset=%s | seed=%s | "
                "coreset_size=%s | results_base=%s\n%s",
                method_name, dataset_name, seed, coreset_size, results_base,
                traceback.format_exc(),
            )
            return None
        finally:
            if tqdm_pos is not None:
                tqdm_position_queue.put(tqdm_pos)

    @staticmethod
    def _is_result_stale(result_path: str) -> bool:
        """Detect cached results from older / buggy runs that should be ignored.

        Currently catches the boolean-prediction bug that affected runs made
        before the data_loader started casting scores to float32: sklearn's
        RidgeCV propagated the bool dtype, predictions were all ``True``, and
        MAE/RMSE ended up around 0.65.
        """
        try:
            r = jbl.load(result_path)
        except Exception:
            return True  # Corrupt -> stale.
        for key in ("pred_acc_train", "pred_acc_test"):
            arr = r.get(key)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.dtype == bool or arr.dtype.kind == "b":
                return True
        return False

    @staticmethod
    def _do_single_trial(method_name, seed, scores, source_models, target_models,
                         true_acc, coreset_size, results_base, dataset_name,
                         exp_suffix, config_dict, use_git, _tqdm_func=None):
        """Inner implementation of a single trial (no tqdm management)."""
        dir_exp = os.path.join(
            results_base,
            f"{dataset_name}{exp_suffix}",
            method_name,
            str(seed),
        )
        os.makedirs(dir_exp, exist_ok=True)

        main_result_path = os.path.join(dir_exp, "result.jbl")
        if os.path.exists(main_result_path):
            if MethodRunner._is_result_stale(main_result_path):
                # Drop the stale file so the trial re-runs below.
                try:
                    os.remove(main_result_path)
                except OSError:
                    pass
                ckpt_path = os.path.join(dir_exp, "ckpt.jbl")
                if os.path.exists(ckpt_path):
                    try:
                        os.remove(ckpt_path)
                    except OSError:
                        pass
            else:
                return None

        set_random_seed(seed)
        method = all_methods[method_name]()

        extra_fit_kwargs = {}
        base_key = getattr(method, "_base_method_key", None)
        if base_key is not None:
            base_ckpt = os.path.join(
                results_base,
                f"{dataset_name}{exp_suffix}",
                base_key,
                str(seed),
                "ckpt.jbl",
            )
            if not os.path.exists(base_ckpt):
                return None
            extra_fit_kwargs["base_ckpt_path"] = base_ckpt

        timer = Timer()
        timer.start()
        method.fit(
            source_full_scores=scores[source_models],
            coreset_size=coreset_size,
            seed=seed,
            **extra_fit_kwargs,
        )
        training_time = timer.get_last_duration()

        compressed_indices = method.get_coreset()

        timer.start()
        pred_acc_test = method.predict(
            scores[target_models][:, compressed_indices]
        )
        inference_time = timer.get_last_duration()

        pred_acc_train = method.predict(
            scores[source_models][:, compressed_indices]
        )

        test_residuals = pred_acc_test - true_acc[target_models]
        error_MAE = float(np.fabs(test_residuals).mean())
        error_MSE = float((test_residuals ** 2).mean())
        error_RMSE = np.sqrt(error_MSE)

        true_all = np.concatenate([
            true_acc[source_models], true_acc[target_models],
        ])
        pred_all = np.concatenate([pred_acc_train, pred_acc_test])
        corr_spearman = corr_kendall = corr_pearson = np.nan
        if len(true_all) >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = spearmanr(true_all, pred_all)
                if np.isfinite(rho):
                    corr_spearman = float(rho)
                tau, _ = kendalltau(true_all, pred_all)
                if np.isfinite(tau):
                    corr_kendall = float(tau)
                r, _ = pearsonr(true_all, pred_all)
                if np.isfinite(r):
                    corr_pearson = float(r)

        recorder = Recorder(
            os.path.join(dir_exp, "record"),
            config=config_dict,
            use_git=use_git,
        )
        recorder["training_time"] = training_time
        recorder["inference_time"] = inference_time
        recorder["error_MAE"] = error_MAE
        recorder["error_RMSE"] = error_RMSE
        recorder["error"] = error_MAE
        if not np.isnan(corr_spearman):
            recorder["corr_spearman"] = corr_spearman
        if not np.isnan(corr_kendall):
            recorder["corr_kendall"] = corr_kendall
        if not np.isnan(corr_pearson):
            recorder["corr_pearson"] = corr_pearson
        recorder.end_recording()

        selection_metrics = getattr(method, "selection_metrics", None)

        result_dict = {
            "seed": seed,
            "coreset_indices": np.asarray(compressed_indices).tolist(),
            "train_model_indices": [int(i) for i in source_models],
            "test_model_indices": [int(i) for i in target_models],
            "pred_acc_train": np.asarray(pred_acc_train).tolist(),
            "pred_acc_test": np.asarray(pred_acc_test).tolist(),
            "true_acc_train": true_acc[source_models].tolist(),
            "true_acc_test": true_acc[target_models].tolist(),
            "error_MAE": error_MAE,
            "error_RMSE": error_RMSE,
        }
        if selection_metrics is not None:
            result_dict["selection_metrics"] = selection_metrics

        jbl.dump(result_dict, os.path.join(dir_exp, "result.jbl"))

        method.save(os.path.join(dir_exp, "ckpt.jbl"))

        ret = {
            "method_name": method_name,
            "seed": seed,
            "source_models": source_models,
            "target_models": target_models,
            "true_acc_train": true_acc[source_models].tolist(),
            "true_acc_test": true_acc[target_models].tolist(),
            "pred_acc_train": pred_acc_train.tolist(),
            "pred_acc_test": pred_acc_test.tolist(),
            "coreset_indices": np.asarray(compressed_indices).tolist(),
            "error_MAE": error_MAE,
            "error_RMSE": error_RMSE,
            "error": error_MAE,
        }
        if selection_metrics is not None:
            ret["selection_metrics"] = selection_metrics
        return ret

    def _trial_result_path(self, method_name: str, seed: int, dataset_name: str) -> str:
        return os.path.join(
            self.results_base,
            f"{dataset_name}{self.config.exp_suffix}",
            method_name,
            str(seed),
            "result.jbl",
        )

    def run_for_dataset(self, dataset_name: str) -> Dict:
        """Run all configured methods for a single dataset."""
        print(f"\n=== Running dataset: {dataset_name} ===")

        scores, model_names, true_acc = self.loader.load(dataset_name)
        num_models, num_data = scores.shape

        coreset_size = self.resolve_coreset_size(self.config.coreset_size, num_data)
        num_train_models = getattr(self.config, "num_train_models", "default")
        n_methods = len(self.config.methods)
        n_seeds = self.config.num_run
        print(
            f"  coreset_size={self.config.coreset_size} -> {coreset_size}  "
            f"(num_data={num_data})  num_train_models={num_train_models}  "
            f"methods={n_methods}  seeds={n_seeds}  trials={n_methods * n_seeds}"
        )

        config_dict = {k: self.config[k] for k in list(self.config.keys())}

        tasks = []
        for seed in range(
            self.config.seed_start, self.config.seed_start + self.config.num_run
        ):
            set_random_seed(seed)
            source_models, target_models = self._compute_split(
                num_models, true_acc, num_train_models,
                model_names=model_names,
            )
            for method_name in self.config.methods:
                tasks.append((
                    method_name, seed, scores,
                    source_models, target_models, true_acc,
                    coreset_size, self.results_base, dataset_name,
                    self.config.exp_suffix, config_dict, self.config.use_git,
                ))

        multi_process = getattr(self.config, "multi_process", False)
        mp_progress_mode = getattr(self.config, "mp_progress_mode", "terminal")
        if mp_progress_mode not in ("terminal", "compact"):
            raise ValueError(
                f"Unknown mp_progress_mode={mp_progress_mode!r}; "
                "expected 'terminal' or 'compact'."
            )
        bar_desc = f"  {dataset_name}"
        if multi_process and len(tasks) > 1:
            n_cpus = cpu_count()
            n_workers = min(n_cpus, len(tasks))
            mode_label = (
                "compact (overall bar only)"
                if mp_progress_mode == "compact"
                else "terminal (per-worker bars)"
            )
            print(
                f"  Running {len(tasks)} trials in parallel "
                f"({n_workers} workers, {mode_label})"
            )

            mgr = Manager()
            position_queue = mgr.Queue()
            for i in range(n_workers):
                position_queue.put(i + 1)

            worker_bars = {}
            stop_event = threading.Event()
            poller = None
            progress_dict = None

            if mp_progress_mode == "terminal":
                progress_dict = mgr.dict()

                _w_digits = len(str(n_workers))
                _max_method = max(len(m) for m in self.config.methods)
                _desc_width = 3 + _w_digits + _max_method + 2 + 25

                for i in range(1, n_workers + 1):
                    worker_bars[i] = tqdm(total=1, position=i, leave=False)

                def _poll_worker_progress():
                    while not stop_event.is_set():
                        try:
                            snapshot = dict(progress_dict)
                        except Exception:
                            snapshot = {}
                        for pos, info in snapshot.items():
                            if info is not None and pos in worker_bars:
                                bar = worker_bars[pos]
                                bar.total = info.get('total') or 1
                                bar.n = min(info.get('n', 0), bar.total)
                                bar.set_description_str(
                                    info.get('desc', '').ljust(_desc_width))
                                postfix = info.get('postfix', '')
                                if postfix:
                                    bar.set_postfix_str(postfix)
                                bar.refresh()
                        stop_event.wait(0.15)

                poller = threading.Thread(
                    target=_poll_worker_progress, daemon=True,
                )
                poller.start()

            parallel_tasks = [
                t + (position_queue, progress_dict, mp_progress_mode)
                for t in tasks
            ]
            gen = Parallel(n_jobs=n_workers, return_as="generator")(
                delayed(MethodRunner._run_single_trial)(*t)
                for t in parallel_tasks
            )
            trial_results = []
            pbar_kwargs = {"desc": bar_desc, "unit": "trial"}
            if mp_progress_mode == "terminal":
                pbar_kwargs["position"] = 0
            pbar = tqdm(gen, total=len(parallel_tasks), **pbar_kwargs)
            for result in pbar:
                trial_results.append(result)
                if result is not None:
                    pbar.set_postfix_str(
                        f"last: {result['method_name']} "
                        f"(seed={result['seed']}, MAE={result['error_MAE']:.4f})"
                    )
                else:
                    pbar.set_postfix_str("last: (cached)")
            pbar.close()

            stop_event.set()
            if poller is not None:
                poller.join(timeout=1.0)
            for bar in worker_bars.values():
                bar.close()
            if mp_progress_mode == "terminal":
                print('\n' * n_workers)
            mgr.shutdown()
        else:
            total = len(tasks)
            width = len(str(total))
            trial_results = []
            for idx, task in enumerate(tasks, start=1):
                method_name, seed = task[0], task[1]
                result_path = self._trial_result_path(method_name, seed, dataset_name)
                if os.path.exists(result_path):
                    tag = "stale" if self._is_result_stale(result_path) else "cached"
                else:
                    tag = "running"
                print(
                    f"  [{idx:>{width}}/{total}] {tag:<7} {method_name} (seed={seed})",
                    flush=True,
                )
                result = MethodRunner._run_single_trial(*task)
                trial_results.append(result)
                if result is not None:
                    print(
                        f"           done    {method_name} (seed={seed})  "
                        f"MAE={result['error_MAE']:.4f}  "
                        f"RMSE={result['error_RMSE']:.4f}",
                        flush=True,
                    )

        dataset_results = {method_name: [] for method_name in self.config.methods}
        for result in trial_results:
            if result is not None:
                dataset_results[result["method_name"]].append(result)

        dataset_results = {
            method_name: {
                "results": method_results,
                "mean_error": np.mean(
                    [r.get("error_MAE", r.get("error")) for r in method_results]
                ) if method_results else float("nan"),
                "std_error": np.std(
                    [r.get("error_MAE", r.get("error")) for r in method_results]
                ) if method_results else float("nan"),
                "mean_error_RMSE": np.mean(
                    [r["error_RMSE"] for r in method_results if "error_RMSE" in r]
                ) if any("error_RMSE" in r for r in method_results) else np.nan,
                "std_error_RMSE": np.std(
                    [r["error_RMSE"] for r in method_results if "error_RMSE" in r]
                ) if any("error_RMSE" in r for r in method_results) else np.nan,
            }
            for method_name, method_results in dataset_results.items()
        }

        return dataset_results

    def run_all_datasets(self) -> Dict:
        """Run all configured datasets and persist intermediate results."""
        all_datasets_list = self.config.datasets
        os.makedirs(self.results_base, exist_ok=True)

        results = {}
        for dataset in all_datasets_list:
            dataset_results = self.run_for_dataset(dataset)
            results[dataset] = dataset_results
            jbl.dump(results, os.path.join(self.results_base, "results.jbl"))

        return results
