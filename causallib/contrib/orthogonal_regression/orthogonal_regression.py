from typing import Dict, Hashable, Protocol, Union
from copy import deepcopy
from collections import defaultdict

import pandas as pd

from causallib.survival.regression_curve_fitter import RegressionCurveFitter as TimePooledModel


class Classifier(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict_proba(self, X): ...


class Regressor(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...


class OrthogonalRegression:
    def __init__(
        self,
        covariate_models: Dict[Hashable, Union[Regressor, Classifier]],
        id_col: Hashable,
        time_col: Hashable,
        lag: int = 1,
        pool_time=False,
    ):
        """Orthogonal regression for time-varying effect estimation.
        Orthogonal regression is a method to transform time-varying data
        in order to make valid effect estimation in the presence of
        treatment-confounder feedback.

        The method works by modeling the time-varying covariates at each
         time point using the covariates and treatment assignment from previous
         time-points and then residualizing them.

        This model implements a generic variation of the algorithm presented in
        Causal Inference with Orthogonalized Regression Adjustment: Taming the Phantom
        by Bates et al. 2022, https://arxiv.org/abs/2201.13451

        Args:
            covariate_models: A mapping between covariate names and an estimator.
                It is up to the user to match binary covariates with "classification
                models" (implementing `predict_proba()`) and match continuous covariates
                with regressors (implementing `predict()`).
                Covariates existing in the data and not here will not be modeled and
                residualized (tranformed), i.e., will be kept as is. This fits covariates
                that are either baseline covariates or may not be part of a
                treatment-confounder feedback loop.
            id_col: The name of the column specifying an observation's ID.
            time_col: The name of the column specifying the time
                      (data in assumed to be in a person-time "long" format).
            lag: Number of time-steps history to use when modeling a covariate at time `t`.
                 Uses all time-points possible until `t-lag`, e.g., if lag=2, at `t=2` it
                 will only use `t=1` but at `t=3` it will use both `t=1` and `t=2` and at
                 `t=4` it will again use `t=2` and `t=3`.
            pool_time: Whether to pool together different time-points into a single model.
                       NotImplemented, currently IGNORED.
        """
        self.covariate_models = covariate_models
        self.id_col = id_col
        self.time_col = time_col
        self.lag = lag
        # self.pool_time = pool_time  # TODO: add a time-pooled model?

    def fit(
        self,
        X: pd.DataFrame,
        a: pd.DataFrame,
        y=None,
    ):
        """Fits an Orthogonal Regression transformer.

        Args:
            X: Time-varying covariates in person-time ("long") format,
               with ID column name specified in `self.id_col` and
               time column name specified in `self.time_col`.
               Baseline covariates can be specified by first being duplicated
               at every time point, and second having a time_point preceding
               the time-varying time-points. For example, if time-varying covariates
               start at `t=1`, then baseline covairates should start at `t=0`.
               Column names should match the keys in `self.covariate_models`.
               Time-varying covariates that shouldn't be modelled can exist as
               columns in `X` but not be specified as keys in `self.covariate_models`.
            a: Time-varying treatment assignment in person-time ("long") format,
               with ID column name specified in `self.id_col` and
               time column name specified in `self.time_col`.
            y:

        Returns:
            OrthogonalRegression
        """
        # TODO: check for match between self.covariate_models.keys() and X.columns?
        models = defaultdict(dict)  # TODO: should index by either time/cov first or even by tuple?
        Xa, Y = self._prepare_data(X, a, y)
        # a-priori merging saves some duplicated wrangling, but creates edge cases (e.g., all-null predictors/targets)
        time_points = Xa.index.unique(level=self.time_col).sort_values()
        for covariate in self.covariate_models.keys():
            for time_point in time_points:
                cur_y = Y.xs(time_point, level=self.time_col)[covariate]
                if cur_y.isna().all():
                    # No data for `covariate` at this time-point.
                    # Was probably added in the "outer" merge of `X` and `a`.
                    break
                cur_Xa = Xa.xs(time_point, level=self.time_col)
                cur_Xa = cur_Xa.dropna(axis="columns", how="all")
                cur_model = deepcopy(self.covariate_models[covariate])
                if self._is_intercept_data(cur_Xa):
                    try:
                        cur_model.fit_intercept = False
                    except AttributeError:
                        pass

                cur_model.fit(cur_Xa, cur_y)
                models[covariate][time_point] = cur_model

        self.covariate_models_ = models
        return self

    # TODO: maybe do "create data" which will build a multivariate (i.e. multioutput) time-pooled data X, y.
    #       Then decide whether to model x_j,t separately (as now), pool time together for each covariate,
    #       do multivariate model (across x_j) at each timepoint separately, or both.

    def _prepare_data(self, X, a, y=None):
        Xa = X.merge(a, on=["id", "t"], how="outer")
        # a-priori merging saves some duplicated wrangling, but creates edge cases (e.g., all-null predictors/targets)
        time_points = sorted(Xa[self.time_col].unique())
        design_X, design_Y = {}, {}
        for time_point in time_points:
            cur_Y = self._get_targets(Xa, time_point)
            if cur_Y.isna().all().all():
                continue
            cur_Xa = self._get_predictors(Xa, time_point, cur_Y.index)
            design_Y[time_point] = cur_Y
            design_X[time_point] = cur_Xa
        design_X = pd.concat(design_X, names=[self.time_col])
        design_Y = pd.concat(design_Y, names=[self.time_col])
        return design_X, design_Y

    def _get_predictors(self, Xa, time_point, index, covariate=None):
        """Extracts the predictors matrix for a given time.
        `index` is required in case an empty set of predictors is defined and
        an intercept is required to be generated manually."""
        cur_time_span = max((Xa[self.time_col]).min(), time_point - self.lag)
        cur_Xa = Xa.loc[Xa[self.time_col].between(cur_time_span, time_point - 1)]
        # cur_Xa = cur_Xa[covariate]  # TODO: model by just the covariate? predictor set per covariate?
        cur_Xa = self._long_to_wide(cur_Xa)

        if cur_Xa.empty:
            cur_Xa = self._create_intercept_data(index)
        return cur_Xa

    def _get_targets(self, Xa, time_point):
        """Extracts the prediction outcome at a given time."""
        # cur_y = Xa.loc[Xa[self.time_col] == time_point, [self.id_col, covariate]]
        # cur_y = cur_y.set_index(self.id_col)[covariate]  # Remove column dimension
        modeled_covariates = list(self.covariate_models.keys())
        cur_y = (
            Xa.set_index([self.time_col, self.id_col])
            .xs(time_point, level=self.time_col)
        )[modeled_covariates]
        return cur_y

    def _long_to_wide(self, cur_Xa, dropna=True, abst2deltat=True):
        """Transforms long person-time format to wide format indexed by person.
        `dropna` is used to drop entirely empty columns
        (no information for some covariate at specific time).
        `abst2deltat` converts absolute time to delta time,
        this is required for proper column naming during `fit`,
        but can be redundant when transforming."""
        cur_Xa = cur_Xa.pivot(index=self.id_col, columns=self.time_col)
        if abst2deltat:
            # Convert absolute time to delta time from current time-point:
            # Example: if absolute times are [2, 3] map to delta time [t-2, t-1]
            level_values = cur_Xa.columns.unique(level=self.time_col)
            level_values = level_values - level_values.min() + 1
            level_values = level_values[::-1]
            if not level_values.empty:
                # `set_levels` crashes for empty Index
                cur_Xa.columns = cur_Xa.columns.set_levels(
                    level_values, level=self.time_col
                )
        cur_Xa.columns = [f"{c[0]}__{c[1]}" for c in cur_Xa.columns.to_flat_index()]
        if dropna:
            # No data for covariate, probably due to outer merge:
            cur_Xa = cur_Xa.dropna(axis="columns", how="all")
        # Remove repeated baseline columns; "last" will always take t-1.
        cur_Xa = cur_Xa.T.drop_duplicates(keep="last").T
        # TODO: also remove time-point suffix in col name?
        # TODO: alternatively, pass another optional X_baseline instead of duplicating in person-time format
        return cur_Xa

    def _orthogonalize(self, X, a, y=None):
        """Main logic for orthogonalizing the time-varying covariates.
        Returns `X` and `a` combined in a long-format."""
        Xa, Y = self._prepare_data(X, a, y)
        res = X.merge(a, on=["id", "t"], how="outer")
        time_points = Xa.index.unique(level=self.time_col).sort_values()
        for covariate in self.covariate_models.keys():
            for time in time_points:
                # TODO: then maybe change the structure to dict of tuple: estimator, rather than nested dict?
                cur_y = Y.xs(time, level=self.time_col)[covariate]
                if cur_y.isna().all():
                    break
                cur_Xa = Xa.xs(time, level=self.time_col)
                cur_Xa = cur_Xa.dropna(axis="columns", how="all")
                cur_model = self.covariate_models_.get(covariate, {}).get(time)
                if cur_model is None:
                    raise ValueError(
                        f"Covariate `{covariate}` at time `{time}` does not seem to be modelled. "
                        f"Probably was not seen during `fit`. "
                        f"Available times: {list(self.covariate_models_.keys())}. "
                        f"Available covariates: f{list(self.covariate_models_.get(time, {}).keys())}"
                    )
                cur_pred = self.__estimator_predict(cur_model, cur_Xa)
                cur_resid = cur_y.values - cur_pred  # avoid assignment by Index
                res.loc[res[self.time_col] == time, covariate] = cur_resid
        return res

    def transform(
        self,
        X,
        a,
        y=None
    ):
        """Orthogonalizes the data.
        Data is returned in wide-format, indexed by person ID,
        ready to be modelled against a non-time-varying outcome `y`.

        Args:
            X: Time-varying covariates in person-time ("long") format,
               with ID column name specified in `self.id_col` and
               time column name specified in `self.time_col`,
               matching the input specification in `fit()`.
            a: Time-varying treatment assignment in person-time ("long") format,
               with ID column name specified in `self.id_col` and
               time column name specified in `self.time_col`.
            y:

        Returns:
            pd.DataFrame: wide-format orthogonalized data.
        """
        res = self._orthogonalize(X, a, y)
        res = self._long_to_wide(res, abst2deltat=False)
        return res

    @staticmethod
    def _create_intercept_data(index):
        intercept = pd.DataFrame(data=1, columns=["intercept"], index=index)
        return intercept

    @staticmethod
    def _is_intercept_data(X):
        is_intercept = (
                X.columns.to_list() == ["intercept"]
                and all(X["intercept"] == 1)
        )
        return is_intercept

    @staticmethod
    def __estimator_predict(estimator, X):
        try:
            return estimator.predict_proba(X)[:, -1]
        except AttributeError:
            return estimator.predict(X)



