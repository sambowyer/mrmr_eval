from .anchor_points import AnchorPointsWeightedPred, AnchorPointPredictorPred
from .double_optimize import DoubleOptimizePred
from .lasso import LassoPred
from .pca import PCAPred
from .random import RandomSampling, RandomSamplingAndLearn, RandomSearchAndLearn
from .aipw import AIPWPred
from .tiny_bench import PIRTPred, GPIRTPred
from .metabench import MetaBench
from .sort_search import SortAndSearchSum, SortAndSearchRecursiveSum

all_methods = {
    "random_sampling": RandomSampling,
    "random_sampling_and_learn": RandomSamplingAndLearn,
    "random_search_and_learn": RandomSearchAndLearn,
    "aipw": AIPWPred,
    "pca": PCAPred,
    "anchor_points_weighted": AnchorPointsWeightedPred,
    "anchor_points_predictor": AnchorPointPredictorPred,
    "double_optimize": DoubleOptimizePred,
    "lasso": LassoPred,
    "pirt": PIRTPred,
    "gpirt": GPIRTPred,
    "metabench": MetaBench,
    "sort_search_sum": SortAndSearchSum,
    "sort_search_recursive_sum": SortAndSearchRecursiveSum,
}
