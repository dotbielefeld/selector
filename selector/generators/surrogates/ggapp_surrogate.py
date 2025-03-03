"""This module contains functions for the GGA++ surrogate."""
import joblib
import six
import sklearn
import sklearn.ensemble._base as base
import sys
sys.modules['sklearn.ensemble.base'] = base
sys.modules['sklearn.externals.joblib'] = joblib
sys.modules['sklearn.externals.six'] = six
sys.modules['sklearn.externals.six.moves'] = six.moves

from costcla import CostSensitiveRandomForestClassifier  # noqa: E402
from costcla.models.cost_tree import CostSensitiveDecisionTreeClassifier  # noqa: E402
from costcla.models.bagging import BaggingClassifier  # noqa: E402
from costcla.metrics import cost_loss  # noqa: E402

import numpy as np  # noqa: E402
import numbers  # noqa: E402
import copy  # noqa: E402
import itertools  # noqa: E402
import statistics as st  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from sklearn.ensemble import BaggingRegressor  # noqa: E402
from sklearn.ensemble._bagging import (  # noqa: E402
    BaseBagging,
    _parallel_build_estimators,
    _parallel_predict_regression
)
from sklearn.utils import check_random_state  # noqa: E402
from sklearn.utils.validation import (  # noqa: E402
    has_fit_parameter,
    check_is_fitted,
    _check_sample_weight
)
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators  # noqa: E402
from sklearn.base import RegressorMixin  # noqa: E402
from sklearn.utils.fixes import delayed  # noqa: E402
from joblib import Parallel  # noqa: E402
MAX_INT = np.iinfo(np.int32).max

from selector.pool import ParamType, Generator  # noqa: E402
from selector.pool import Configuration as SelConfig  # noqa: E402
from selector.point_gen import PointGen  # noqa: E402
from selector.generators.default_point_generator import (  # noqa: E402
    check_conditionals,
    check_no_goods
)
from selector.generators.random_point_generator import (  # noqa: E402
    reset_no_goods,
    reset_conditionals,
    random_point
)
from selector.generators.variable_graph_point_generator import (  # noqa: E402
    variable_graph_point,
    Mode
)
from selector.point_gen import PointGen  # noqa: E402
import uuid  # noqa: E402
import random  # noqa: E402

from threadpoolctl import ThreadpoolController
controller = ThreadpoolController()

__all__ = ['GGAppSurr']


class GGApp(CostSensitiveDecisionTreeClassifier):
    """GGA++ Decision Tree Regressor."""

    def __init__(self,
                 criterion='direct_cost',
                 criterion_weight=False,
                 num_pct=10,
                 max_features=None,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_gain=.1,
                 pruned=True,
                 q=0.1
                 ):
        """Initialize Decision Tree Regressor."""
        self.criterion = criterion
        self.criterion_weight = criterion_weight
        self.num_pct = num_pct
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_score = min_gain
        self.min_gain = min_gain
        self.pruned = pruned

        self.n_features_ = None
        self.max_features_ = None

        self.tree_ = []

        self.q = q

    def _node_value(self, y_true, X):
        """Private function to calculate the value of a node.

        Parameters
        ----------
        y_true : array indicator matrix
                Ground truth (correct) labels.
        Returns
        -------
        tuple(node prediction : float, node predicted probability : float)
        """
        # Criterion
        y_pred = st.median(y_true)
        y_prob = norm.pdf(3, loc=y_pred, scale=np.std(y_true))

        return y_pred, y_prob

    def fit(self, X, y, check_input=False):
        """Fitting function."""
        n_samples, self.n_features_ = X.shape

        # mock cost_matrix with no influence
        cost_mat = np.array([[1, 1, 0, 0] for _ in X])

        self.tree_ = self._tree_class()

        # Maximum number of features to be taken into account per split
        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(
                    1, int(self.max_features * self.n_features_))
            else:
                max_features = 1  # On sklearn is 0.
        self.max_features_ = max_features

        self.tree_.tree = self._tree_grow(y, X, cost_mat)

        if self.pruned:
            self.pruning(X, y, cost_mat)

        return self

    def _tree_grow(self, y_true, X, cost_mat, level=0):

        if len(X.shape) == 1:
            tree = dict(y_pred=y_true, y_prob=0.5, level=level,
                        split=-1, n_samples=1, gain=0)
            return tree

        # Calculate the best split of the current node
        split, score, Xl_pred, y_pred, y_prob = self._best_split(
            y_true, X, cost_mat)

        n_samples, n_features = X.shape

        # Construct the tree object as a dictionary

        tree = dict(y_pred=y_pred, y_prob=y_prob, level=level,
                    split=-1, n_samples=n_samples, gain=score)

        # Check the stopping criteria
        if score < self.min_score:
            return tree
        if self.max_depth is not None:
            if level >= self.max_depth:
                return tree
        if n_samples <= self.min_samples_split:
            return tree
        if X.shape[0] <= 10:
            return tree

        j, l = split
        filter_Xl = (X[:, j] <= l)
        filter_Xr = ~filter_Xl
        n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
        n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

        if min(n_samples_Xl, n_samples_Xr) <= self.min_samples_leaf:
            return tree

        # No stopping criteria is met
        tree['split'] = split
        tree['node'] = self.tree_.n_nodes
        self.tree_.n_nodes += 1

        tree['sl'] = self._tree_grow(
            y_true[filter_Xl], X[filter_Xl], cost_mat[filter_Xl], level + 1)
        tree['sr'] = self._tree_grow(
            y_true[filter_Xr], X[filter_Xr], cost_mat[filter_Xr], level + 1)

        return tree

    def _best_split(self, y_true, X, cost_mat):

        n_samples, n_features = X.shape
        num_pct = self.num_pct

        cost_base, _, _ = self._node_cost(y_true, cost_mat)
        y_pred, y_prob = self._node_value(y_true, X)

        # Calculate the gain of all features each split in num_pct
        scores = np.zeros((n_features, num_pct))
        pred = np.zeros((n_features, num_pct))
        splits = np.zeros((n_features, num_pct))

        # Selected features
        selected_features = np.arange(0, self.n_features_)
        # Add random state
        np.random.shuffle(selected_features)
        selected_features = selected_features[:self.max_features_]
        selected_features.sort()

        # For each feature test all possible splits
        for j in selected_features:
            splits[j, :] = np.percentile(
                X[:, j], np.arange(0, 100, 100.0 / num_pct).tolist())

            for l in range(num_pct):
                # Avoid repeated values,
                # since np.percentile may return repeated values
                if l == 0 or (l > 0 and splits[j, l] != splits[j, l - 1]):
                    split = (j, splits[j, l])
                    scores[j, l], pred[j, l] = self._calculate_score(
                        cost_base, y_true, X, cost_mat, split, self.q)

        best_split = np.unravel_index(scores.argmax(), scores.shape)

        return (best_split[0], splits[best_split]), \
            scores.max(), pred[best_split], y_pred, y_prob

    def _calculate_score(self, cost_base, y_true, X, cost_mat, split, q):
        """Calculate GGA++ score."""
        # Get qth percentile performance threshold
        j, l = split
        X_sort = copy.deepcopy(X)
        y_sort = copy.deepcopy(y_true)
        sort = np.argsort(y_sort)
        # sort = np.argsort(y_sort)[::-1]
        y_sort = y_sort[sort]
        X_sort = X_sort[sort]
        h_idx = int(len(y_sort) * q)
        v_h = y_sort[h_idx]

        # Split by split value
        filter_Xl = (X_sort[:, j] <= l)  # L
        filter_Xr = ~filter_Xl  # R

        # T: all <= h_idx, U: all > h_idx

        # Compute score
        lls = 0
        rls = 0
        lts = 0
        rts = 0
        ltn = 0  # |L ∩ T|
        rtn = 0  # |R ∩ T|

        for l_idx, part_l in enumerate(filter_Xl):
            if l_idx > h_idx and part_l:
                lls += (v_h - y_true[l_idx])**2

            if l_idx <= h_idx and part_l:
                lts += (y_true[l_idx] - v_h)**2
                # ltn = |L ∩ T|
                ltn += 1

            if l_idx > h_idx and not part_l:
                rls += (v_h - y_true[l_idx])**2

            if l_idx <= h_idx and not part_l:
                rts += (y_true[l_idx] - v_h)**2
                # rtn = |R ∩ T|
                rtn += 1

        al = (ltn + lts) / (1 + lls)
        ar = (rtn + rts) / (1 + rls)

        if ltn > rtn:
            score = al
        elif ltn == rtn:
            score = min([al, ar])
        elif ltn < rtn:
            score = ar

        # Reduce Iterations avoiding low gain
        if score < self.min_gain:
            return 0.0, int(np.sign(y_true.mean() - 0.5) == 1)

        n_samples, n_features = X_sort.shape

        # Check if one of the leafs is empty
        # TODO: This must be check in _best_split
        if np.nonzero(filter_Xl)[0].shape[0] in [0, n_samples]:  # 1 leaf empty
            return 0.0, 0.0

        Xl_pred, _ = self._node_value(y_true[filter_Xr],
                                      cost_mat[filter_Xr, :])

        return score, Xl_pred


class GGAppRegressorMixin(RegressorMixin):
    """Mixin class for all regression estimators in scikit-learn."""

    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Compute score."""
        from .metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        return {"requires_y": True}

    def _fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        sample_weight=None,
        check_input=True,
    ):
        random_state = check_random_state(self.random_state)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # Remap output
        n_samples = X.shape[0]
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if max_samples > X.shape[0]:
            raise ValueError("max_samples must be <= n_samples")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * self.n_features_in_)

        if max_features > self.n_features_in_:
            raise ValueError("max_features must be <= n_features")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only",
                             "available if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only",
                             "available if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]: starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
                check_input=check_input,
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )

        if self.oob_score:
            self._set_oob_score(X, y)

        return self


class BaggingRegressor(GGAppRegressorMixin, BaseBagging):
    """sklearn Bagging Regressor Redefinition."""

    def __init__(
        self,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        base_estimator="deprecated",
    ):
        """Initialize BaggingRegressor."""
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            base_estimator=base_estimator,
        )

    def _fit(
        self,
        X,
        y,
        max_samples=None,
        max_depth=None,
        sample_weight=None,
        check_input=True,
    ):
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        max_samples : int or float, default=None
            Argument to use instead of self.max_samples.
        max_depth : int, default=None
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.
        check_input : bool, default=True
            Override value used when fitting base estimator. Only supported
            if the base estimator has a check_input parameter for fit function.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)
        sample_weight = None

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # Remap output
        n_samples = X.shape[0]
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if max_samples > X.shape[0]:
            raise ValueError("max_samples must be <= n_samples")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * self.n_features_in_)

        if max_features > self.n_features_in_:
            raise ValueError("max_features must be <= n_features")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only",
                             "available if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only",
                             " available if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError(
                "n_estimators=%d must be larger or equal to "
                "len(estimators_)=%d when warm_start==True"
                % (self.n_estimators, len(self.estimators_))
            )

        elif n_more_estimators == 0:
            warn(
                "Warm-start fitting without increasing n_estimators does not "
                "fit new trees."
            )
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(
            n_more_estimators, self.n_jobs
        )
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = Parallel(
            n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args()
        )(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]: starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
                # check_input=check_input,
            )
            for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(
            itertools.chain.from_iterable(t[0] for t in all_results)
        )
        self.estimators_features_ += list(
            itertools.chain.from_iterable(t[1] for t in all_results)
        )

        if self.oob_score:
            self._set_oob_score(X, y)

        return self


class GGAppBaggingRegressor(BaggingRegressor):
    """GGA++ Bagging Regressor."""

    def __init__(
        self,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        base_estimator="deprecated",
    ):
        """Initialize GGA++ Bagging Regressor."""
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            base_estimator=base_estimator,
        )

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]

        predictions = np.zeros((n_samples,))
        n_predictions = np.zeros((n_samples,))

        for estimator, samples, features in zip(
            self.estimators_, self.estimators_samples_,
            self.estimators_features_
        ):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            predictions[mask] += \
                estimator.predict((X[mask, :])[:, features])
            n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warn(
                "Some inputs do not have OOB scores. "
                "This probably means too few estimators were used "
                "to compute any reliable oob estimates."
            )
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions

        self.oob_prediction_ = predictions
        self.oob_score_ = r2_score(y, predictions)

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            dtype=None,
            force_all_finite=False,
            reset=False,
        )

        # Parallel loop
        n_jobs, _, starts = _partition_estimators(self.n_estimators,
                                                  self.n_jobs)

        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_regression)(
                self.estimators_[starts[i]: starts[i + 1]],
                self.estimators_features_[starts[i]: starts[i + 1]],
                X,
            )
            for i in range(n_jobs)
        )

        # Reduce
        y_hat = sum(all_y_hat) / self.n_estimators

        return y_hat


class GGAppRandomForestRegressor(GGAppBaggingRegressor):
    """GGA++ Random Forest."""

    def __init__(self,
                 n_estimators=10,
                 combination='majority_voting',
                 max_features='auto',
                 n_jobs=1, # will be scenario.tournament_size
                 verbose=False,
                 pruned=False,
                 q=0.1):
        """Initialize Random Forest."""
        super(GGAppBaggingRegressor, self).__init__(
            base_estimator=GGApp(max_features=max_features,
                                 pruned=pruned, q=q),
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            n_jobs=n_jobs,
            random_state=None,
            verbose=verbose)
        self.pruned = pruned


class GGAppSurr():
    """Surrogate for GGA++.

    Note
    ----
    Implementation based on the paper "Model-Based Genetic Algorithms for
    Algorithm Configuration, C. Ans{\'o}tegui et al. and using source code of
    the package costcla.

    Parameters
    ----------
    scenario : selector.scenario.Scenario
        AC scenario.
    seed : int
        Random seed.
    cost : list
        Cost matrix for cost-sensitive classification. Per default set to neutral to take no effect.
    logger : logging.Logger
        Logger from main loop. Default is None, so no Debug infos.
    """

    def __init__(self, scenario, seed=False, cost=[1, 1, 0, 0], logger=None):
        if not seed:
            self.seed = False
        else:
            self.seed = seed

        # Control How many threads/cores numpy and scipy use
        @controller.wrap(limits=scenario.tournament_size,
                         user_api='openmp')
        @controller.wrap(limits=scenario.tournament_size,
                         user_api='blas')
        def threaded_init(scenario, seed, cost):
            self.scenario = scenario
            self.logger = logger
            self.regressor = \
                GGAppRandomForestRegressor(n_jobs=scenario.tournament_size)
            self.sc = StandardScaler()
            self.x_stash = np.array([])
            self.y_stash = np.array([])
            self.cost = cost
            self.best_q = None

            self.transfom_selector_scenario_for_ggapp(scenario)
            self.random_generator = PointGen(self.scenario, random_point,
                                             seed=self.seed)
            self.variable_graph_generator = PointGen(self.scenario,
                                                     variable_graph_point,
                                                     seed=42)

        threaded_init(scenario, seed, cost)

    def transfom_selector_scenario_for_ggapp(self, scenario):
        """Transform scenario from Selector to suit GGAppSurr.

        Parameters
        ----------
        scenario : selector.scenario.Scenario
            AC scenario.
        """
        paramsecenario = self.scenario.parameter
        self.types = {}
        for ps in paramsecenario:
            self.types[ps.name] = ps.type

    def set_cat(self, c):
        """
        Set Cat value to binary.

        Parameters
        ----------
        c : bool
            Any bool value.
        Returns
        -------
        int
            True will be 0, False will be 1.

        """
        if c is True:
            c = 1
        elif c is False:
            c = 0

        return c

    def transform_values(self, conf):
        """Transform configuration values in GGA++ format.

        Parameters
        ----------
        conf : selector.pool.Configurator or list of selector.pool.Configuration
            Configuration(s) to transform for GGAppSurr.

        Returns
        -------
        selector.pool.Configurator or list of selector.pool.Configuration
            Transformed configuration(s).
        """
        config = []
        count = 0
        if type(conf) is list:
            for c in conf:
                for param in self.scenario.parameter:
                    if param.name not in c.conf:
                        c.conf[param.name] = param.bound[0]
            for c in conf:
                count += 1
                single_conf = []
                for t in self.types.keys():
                    if self.types[t] == ParamType.categorical and t in c.conf:
                        if c.conf[t] is None:
                            for sp in self.scenario.parameter:
                                if t == sp.name:
                                    single_conf.append(sp.bound[0])
                        else:
                            if isinstance(c.conf[t], (str, np.str_)):
                                for sp in self.scenario.parameter:
                                    if t == sp.name:
                                        single_conf.append(
                                            sp.bound.index(c.conf[t]))
                            else:
                                single_conf.append(
                                    self.set_cat(float(c.conf[t])))
                    elif t in c.conf:
                        if c.conf[t] is None:
                            for sp in self.scenario.parameter:
                                if t == sp.name:
                                    single_conf.append(sp.bound[0])
                        else:
                            single_conf.append(float(c.conf[t]))
                config.append(np.array(single_conf))
        else:
            for param in self.scenario.parameter:
                if param.name not in conf.conf:
                    conf.conf[param.name] = param.bound[0]
            for t in self.types.keys():
                if self.types[t] == ParamType.categorical and t in conf.conf:
                    if conf.conf[t] is None:
                        for sp in self.scenario.parameter:
                            if t == sp.name:
                                config.append(sp.bound[0])
                    else:
                        if isinstance(conf.conf[t], (str, np.str_)):
                            for sp in self.scenario.parameter:
                                if t == sp.name:
                                    config.append(
                                        sp.bound.index(conf.conf[t]))
                        else:
                            config.append(self.set_cat(float(conf.conf[t])))
                elif t in conf.conf:
                    if conf.conf[t] is None:
                        for sp in self.scenario.parameter:
                            if t == sp.name:
                                config.append(sp.bound[0])
                    else:
                        config.append(float(conf.conf[t]))

        config = np.array(config, dtype=object)

        return config

    def get_costs(self, y):
        """
        Generate cost matrix.

        Parameters
        ----------
        y : ndarray
            Configuration performances.

        Returns
        -------
        ndarray
            **costs**: ndarray of mock cost matrices.
        """
        costs = np.array([self.cost for _ in y])

        return costs

    def update(self, history, configs, results, terminations, ac_runtime=None):
        """Update GGA++ epm.

        Parameters
        ----------
        history : list of selector.pool.Tournament
            Tournament history.
        configs : list of selector.pool.Configuration
            Configurations that participated in the tournament.
        results : dict
            Results of the tournament.
        terminations : dict
            Information about terminations of runs that occurred.
        ac_runtime : int
            Total AC runtime in seconds so far.
        """
        conf = []
        result = []
        config_dict = {}
        for c in configs:
            config_dict[c.id] = c

        # instances in tournament
        instances = history.instance_set
        for cid in config_dict.keys():
            use_results = False
            perf_sum = 0
            perf_count = 0
            # config in results
            for ins in instances:
                # OMIT every censored date in update
                if cid in terminations:
                    if ins in terminations[cid]:
                        break
                else:
                    r = results[cid][ins]
                    perf_count += 1
                    use_results = True
                    if r is not None and not np.isnan(r):
                        perf_sum += results[cid][ins]
                        # result.append(results[cid][ins])
                    else:
                        # This cid/ins pair was a time limit reach
                        perf_sum += self.scenario.cutoff_time
                        # result.append(self.scenario.cutoff_time)
            if use_results:
                result.append(perf_sum / perf_count)
                conf.append(self.transform_values(config_dict[cid]))

        if ac_runtime >= self.scenario.wallclock_limit * 0.15 and \
                len(self.y_stash) > len(result) * 2:
            self.y_stash = self.y_stash[len(result):]
            self.y_stash = np.append(self.y_stash, np.array(result))
        else:
            self.y_stash = np.append(self.y_stash, np.array(result))

        if self.best_q is None:
            if len(self.y_stash) == 0:
                import sys
                self.best_q = sys.maxsize
            else:
                self.best_q = np.min(self.y_stash)
        elif np.min(self.y_stash) < self.best_q:
            self.best_q = np.min(self.y_stash)

        if len(conf) != 0:

            if len(self.x_stash) > 0:
                if ac_runtime >= self.scenario.wallclock_limit * 0.15 and \
                        len(self.x_stash) > len(conf) * 2:
                    self.x_stash = self.x_stash[len(conf):]
                    self.x_stash = np.vstack([self.x_stash, np.array(conf)])
                else:
                    self.x_stash = np.vstack([self.x_stash, np.array(conf)])
            else:
                self.x_stash = np.array(conf)

            self.x_stash = self.sc.fit_transform(self.x_stash)
            self.regressor.fit(self.x_stash, self.y_stash,
                               self.get_costs(self.y_stash))

        if self.logger is not None:
            self.logger.info(f"Length x_stash: {len(self.x_stash)}")

    def get_suggestions(self, scenario, n_samples, data, results, _,
                        oversampling=10):
        """
        Suggest configurations to run next based on the next instance set to run on.

        Parameters
        ----------
        scenario : selector.scenario.Scenario
            AC scenario.
        n_samples : int
            Number of configurations to return.
        data : list of selector.pool.Tournament
            Tournament history.
        results : dict
            Performances of the configuration on the instance set of the tournament.
        _ : list of str
            Instance set to run on in the next tournament.
        oversampling : int
            Multiplier for generation via GGA graph crossover before filtering with GGApp model.
        Returns
        -------
        list of selector.pool.Configuration
            Suggested configurations.
        """
        suggestions = []
        for i in range(oversampling * n_samples):
            suggestions.append(self.variable_graph_generator.point_generator(
                results=results, mode=Mode.best_and_random,
                alldata=data, lookback=i + 1, seed=(42 + i)))

        if len(self.x_stash) > 1:
            predicted_quality = \
                self.predict(suggestions)[0]
            sugg_sorted = np.argsort(predicted_quality)
        else:
            sugg_sorted = [i for i in range(n_samples)]

        best_idx = sugg_sorted[:n_samples]

        best_suggs = list(np.array(suggestions)[best_idx])[:n_samples]

        suggestions = []

        for idx, bs in enumerate(best_suggs):
            config_setting = {}

            if not self.seed:
                identity = uuid.uuid4()
            else:
                identity = uuid.UUID(int=random.getrandbits(self.seed))

            for t in self.types.keys():
                if t in bs.conf:
                    config_setting[t] = bs.conf[t]

            suggestions.append(SelConfig(identity,
                                         config_setting,
                                         Generator.ggapp))

            # Check conditionals and reset parameters if violated
            cond_vio = check_conditionals(scenario,
                                          suggestions[idx].conf)
            if cond_vio:
                suggestions[idx].conf = \
                    reset_conditionals(scenario,
                                       suggestions[idx].conf,
                                       cond_vio)

            # Check no goods and reset values if violated
            ng_vio = check_no_goods(scenario, suggestions[idx].conf)
            while ng_vio:
                suggestions[idx].conf = \
                    reset_no_goods(scenario, suggestions[idx].conf)
                ng_vio = check_no_goods(scenario,
                                        suggestions[idx].conf)

        return suggestions

    def predict(self, confs, _=None):
        """
        Predict performance/quality of configurations with GGA++ EPM.

        Parameters
        ----------
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        _ : list
            List of next instances to run the tournament on.
        Returns
        -------
        tuple
            - ndarray,
              Mean of predicted performance/quality.
            - ndarray,
              Variance of predicted performance/quality.
        """
        if type(confs) is not np.ndarray:
            confs = self.transform_values(confs)

        confs = self.sc.transform(confs)
        estimators_predicts = \
            np.array(([x.predict(confs) for x in self.regressor.estimators_]))

        variances = []
        for idx, _ in enumerate(estimators_predicts[0]):
            variances.append(np.var(estimators_predicts[:, idx]))

        return np.array(self.regressor.predict(confs)), np.array(variances)

    def expected_improvement(self, suggestions, _):
        """
        Compute expected improvement via CPPL model.

        Parameters
        ----------
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        _ : list of str
            List of next instances to be run.
        Returns
        -------
        ndarray
            **ei**: Expected improvements.
        """
        mean, var = self.predict(suggestions)
        std = np.sqrt(var)

        def calculate_ei():
            z = (self.best_q - mean) / std
            return (self.best_q - mean) * norm.cdf(z) + std * norm.pdf(z)

        if np.any(std == 0.0):
            stdc = np.copy(std)
            std[stdc == 0.0] = 1.0
            ei = calculate_ei()
            ei[stdc == 0.0] = 0.0
            return ei
        else:
            return calculate_ei()

    def probability_improvement(self, suggestions, r, ni):
        """
        Compute probability of improvement.

        Parameters
        ----------
        suggestions : list of selector.pool.Configuration
            Suggested configurations.
        r : dict
            Performances of the configuration on the instance set of the tournament.
        ni : list
            List of next instances to run the tournament on.

        Returns
        -------
        ndarray
            **pi_output**: Probabilities of improvement.
        """
        mean, var = self.predict(suggestions)
        std = np.sqrt(var)

        pi = norm.cdf((self.best_q - mean) / std)
        pi_output = []
        for p in pi:
            if np.isnan(p):
                pi_output.append([0])
            else:
                pi_output.append([p])

        return pi_output
