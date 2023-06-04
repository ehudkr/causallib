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
        lag=1,
        pool_time=False,
    ):
        self.covariate_models = covariate_models
        self.id_col = id_col
        self.time_col = time_col
        self.lag = lag
        self.pool_time = pool_time

    def fit(self, X, a, y=None):
        # TODO: check for match between self.covariate_models.keys() and X.columns?
        models = defaultdict(dict)  # TODO: should index by either time/cov first or even by tuple?
        Xa = X.merge(a, on=["id", "t"], how="outer")
        # a-priori merging saves some duplicated wrangling, but creates edge cases (e.g., all-null predictors/targets)
        time_points = sorted(Xa[self.time_col].unique())
        for time_point in time_points:
            # TODO: add a time-pooled model?
            for covariate in self.covariate_models.keys():
                cur_y = self._get_target(Xa, covariate, time_point)
                if cur_y.isna().all():
                    # No data for `covariate` at this time-point.
                    # Was probably added in the "outer" merge of `X` and `a`.
                    break
                cur_Xa = self._get_predictors(Xa, time_point, cur_y.index, covariate)
                cur_model = deepcopy(self.covariate_models[covariate])
                if self._is_intercept_data(cur_Xa):
                    try:
                        cur_model.fit_intercept = False
                    except AttributeError:
                        pass

                cur_model.fit(cur_Xa, cur_y)
                models[time_point][covariate] = cur_model

        self.covariate_models_ = models
        return self

    def _get_predictors(self, Xa, time_point, index, covariate=None):
        cur_time_span = max((Xa[self.time_col]).min(), time_point - self.lag)
        cur_Xa = Xa.loc[Xa[self.time_col].between(cur_time_span, time_point - 1)]
        # cur_Xa = cur_Xa[covariate]  # TODO: model by just the covariate? predictor set per covariate?
        cur_Xa = self._long_to_wide(cur_Xa)

        if cur_Xa.empty:
            cur_Xa = self._create_intercept_data(index)
        return cur_Xa

    def _get_target(self, Xa, covariate, time_point):
        # cur_y = Xa.loc[Xa[self.time_col] == time_point, [self.id_col, covariate]]
        # cur_y = cur_y.set_index(self.id_col)[covariate]  # Remove column dimension
        cur_y = (
            Xa.set_index([self.time_col, self.id_col])
            .xs(time_point, level=self.time_col)[covariate]
        )
        return cur_y

    def _long_to_wide(self, cur_Xa, dropna=True):
        cur_Xa = cur_Xa.pivot(index=self.id_col, columns=self.time_col)
        cur_Xa.columns = [f"{c[0]}__{c[1]}" for c in cur_Xa.columns.to_flat_index()]
        if dropna:
            # No data for covariate, probably due to outer merge:
            cur_Xa = cur_Xa.dropna(axis="columns", how="all")
        cur_Xa = cur_Xa.T.drop_duplicates(keep="first").T  # Repeated baseline columns
        # TODO: also remove time-point suffix in col name?
        # TODO: alternatively, pass another optional X_baseline instead of duplicating in person-time format
        return cur_Xa

    def _orthogonalize(self, X, a, y=None):
        Xa = X.merge(a, on=["id", "t"], how="outer")
        res = Xa.copy()
        time_points = sorted(Xa[self.time_col].unique())
        for time in time_points:
            for covariate in self.covariate_models.keys():
                # TODO: then maybe change the structure to dict of tuple: estimator, rather than nested dict?
                cur_y = self._get_target(Xa, covariate, time)
                if cur_y.isna().all():
                    break
                cur_Xa = self._get_predictors(Xa, time, cur_y.index)
                cur_model = self.covariate_models_.get(time, {}).get(covariate)
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

    def transform(self, X, a, y=None):
        res = self._orthogonalize(X, a, y)
        res = self._long_to_wide(res)
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



