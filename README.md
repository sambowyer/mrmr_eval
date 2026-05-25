# mrmr_eval — minimal benchmark prediction with mRMR

A small, self-contained repo for **benchmark prediction** experiments: given an LLM-evaluation score matrix (models × questions), select a small *coreset* of questions and use that coreset to predict each model's accuracy on the full benchmark. The headline method is **mRMR** (Minimum Redundancy Maximum Relevance) coupled with Ridge / polynomial Kernel-Ridge regression as detailed in our paper: [Efficient Benchmarking Is Just Feature Selection and Multiple Regression](LINK_TO_PAPER). A handful of baselines (IRT, anchor points, random sampling/search, Lasso) are bundled for comparison and we also include the option to refit the base methods with a Ridge or Kernel Ridge regressor on top of pre-computed coresets.

The notebook [`tutorial.ipynb`](tutorial.ipynb) runs the full pipeline on one OpenLLM dataset and reproduces a 1×4 (or 2×4) combined-metric grid styled after the paper figures.

## Note on Forks

This repository is a slimmed, tutorial-first descendant of the experimentation code in https://github.com/sambowyer/mrmr_paper, which in turn started as a fork of the incredibly helpful repo https://github.com/socialfoundations/benchmark-prediction (of which this repository is also a direct fork).


## Quick start (uv)

```bash
cd mrmr_eval

# Create .venv and install the locked dependencies.
uv venv
uv sync

# Open the tutorial.
uv run jupyter lab tutorial.ipynb
```

If you do not want JupyterLab, you can execute the notebook headlessly:

```bash
uv run jupyter nbconvert --to notebook --execute tutorial.ipynb \
    --output tutorial.executed.ipynb
```

## Tutorial walkthrough

The notebook is split into eight cells you can edit in place:

1. **Imports.** Pulls in `MethodRunner`, the `benchpred.all_methods` registry,
   and `main.DEFAULT_METHODS`.
2. **Configuration knobs.** Change one of:
   ```python
   DATASET            = "ifeval"             # any name in openllm_datasets
   NUM_SEEDS          = 3
   CORESET_SIZES      = ["5%", "10%", "15%"]
   NUM_TRAIN_MODELS   = 30                    # int → 1×4 plot
   #                   = [20, 30, 40, 60]    # list → 2×4 plot
   RESULTS_DIR        = "./results"
   ```
3. **Methods.** Lists `DEFAULT_METHODS` and monkey-patches
   `random_search_and_learn` from 10 000 down to 1 000 search iterations so the
   notebook completes in a couple of minutes; bump it back up if you want the
   stronger baseline.
4. **Experiments.** Runs each `(num_train_models, coreset_size)` combination
   through `MethodRunner`. Results land in `RESULTS_DIR/<split>/coreset_<size>/
   nmodels_<n>/<dataset>/<method>/<seed>/result.jbl`; re-runs skip combinations
   that already have a `result.jbl`.
5. **Aggregation.** Walks the results tree and computes per-(method, coreset,
   nmodels) mean RMSE, MAE, Kendall τ, Spearman ρ across seeds.
6. **Plot helpers.** Defines a self-contained `method_style()` (color +
   marker + linestyle per method family) and a `pretty_label()`.
7. **Plot.** Adaptive 1×4 or 2×4 grid (saved as
   `plots/combined_grid_<dataset>_filtered.pdf`).
8. **Coreset inspection.** Shows the question indices picked by `mrmr_MIQ_y`
   on the smallest coreset / first seed, as a sanity check.

## Method registry

The full registry lives in [`benchpred/__init__.py`](benchpred/__init__.py) and
exposes (key → class):

| Family | Keys |
| --- | --- |
| mRMR (Ridge) | `mrmr_MIQ_y`, `mrmr_MID_y`, `mrmr_MI_y` (and `mrmr4_*` … `mrmr9_*` for `mi_k = 4..9`) |
| mRMR (Kernel Ridge, deg 2) | `kmrmr_MIQ_y`, `kmrmr_MID_y`, `kmrmr_MI_y`, etc. |
| IRT | `pirt`, `gpirt`, `pirt1`, `pirt5`, `gpirt1`, `gpirt5`, plus `B_*` Beta variants for continuous data |
| Random / search | `random_sampling`, `random_sampling_and_learn` (+ `k*` KRR variant), `random_search_and_learn` (+ `k*`), `small_search_and_learn` (+ `k*`), `sample_first_and_learn` (+ `k*`) |
| Anchor points | `anchor_points_weighted`, `anchor_points_predictor` |
| Other | `lasso`, `aipw`, `pca`, `double_optimize`, `sort_search_sum`, `sort_search_recursive_sum`, `metabench` |

#### Refit / backfill methods

If you have already run the base methods, you can refit them with a Ridge or Kernel Ridge regressor on top of the pre-computed coresets by adding a `+` to the end of the method key and optionally specifying the degree of the Kernel Ridge regressor, for example with gpirt or anchor_points_weighted:

| Family | Keys |
| --- | --- |
| Refit (Ridge) | `gpirt+`, `anchor_points_weighted+` |
| Refit (Kernel Ridge, deg 2) | `kgpirt+`, `kanchor_points_weighted+` |
| Refit (Kernel Ridge, deg 3) | `k3gpirt+`, `k3anchor_points_weighted+` |
| Refit (Kernel Ridge, deg 4) | `k4gpirt+`, `k4anchor_points_weighted+` |

Notes:

- **Unified MI.** `mrmr_MIQ_y` automatically uses the Ross (2014) MI estimator
  on binary score matrices and the PCA-corrected KSG / LNC estimator on
  continuous matrices, so the `MI*` family covers both regimes — there is no
  separate `PMI*` family as in the original mrmr_paper repository.
- **`metabench`** is available in the registry but is *excluded* from the
  tutorial's `DEFAULT_METHODS` because it is dramatically slower than the
  other methods even on a small dataset. Uncomment the entry in
  [`main.py`](main.py) if you want it.

## CLI usage

The notebook is the recommended entry point, but you can also run the same
pipeline from the shell:

```bash
uv run python main.py \
    --dataset_name ifeval \
    --coreset_size 10% \
    --num_run 3 \
    --num_train_models 30 \
    --methods default \
    --no-multi_process --no-use_git
```

## Repository layout

```
mrmr_eval/
├── benchpred/        # slimmed-down method library
├── data/
│   └── scores/       # cached OpenLLM / HELM / GLUE score matrices (*.jbl)
├── zarth_utils/      # tiny config + recorder + timer utilities
├── data_utils.py     # OpenLLM / HELM / GLUE benchmark downloaders
├── data_loader.py    # per-source DataLoader wrapping data_utils.get_scores
├── method_runner.py  # multi-process trial orchestrator
├── main.py           # DEFAULT_METHODS + CLI entry-point
├── tutorial.ipynb    # end-to-end walkthrough
├── pyproject.toml
├── uv.lock           
├── LICENSE.txt      
└── README.md        
```

## License

This repository is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.


# Citation

If you use this repository in your work, please cite the accompanying paper:

```bibtex
@article{
   ...
}
```