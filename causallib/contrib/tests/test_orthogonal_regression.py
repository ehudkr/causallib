import unittest
import numpy as np
import pandas as pd
from scipy.special import expit

from sklearn.linear_model import LinearRegression, LogisticRegression

from causallib.contrib.orthogonal_regression import OrthogonalRegression


def generate_two_step_sim(
        n=1000,
        rho_ax=0.75,
        rho_xa=0.5,
        rho_ux=0.5,
        rho_xbx=0.25,
        rho_xba=0.25,
        with_step1=False,
        with_baseline=False,
        seed=0,
):
    rng = np.random.default_rng(seed)
    mu = 0
    sigma = 1

    U = rng.normal(mu, sigma, n)
    Xb = rng.normal(mu, sigma, n) if with_baseline else np.zeros(n)
    if with_step1:
        X_1 = rho_ux * U + rho_xbx * Xb + np.sqrt(1 - rho_ux**2 - rho_xbx**2) * rng.normal(size=n)
        # A_1 = 2*((X_1 > 0) - .5)
        A_1 = rho_xa * X_1 + rho_xba * Xb + np.sqrt(1 - rho_xa ** 2 - rho_xbx**2) * rng.normal(size=n)
    else:
        A_1 = np.random.normal(mu, sigma, n)

    X_2 = (
            rho_ax * A_1
            + rho_ux * U
            + rho_xbx * Xb
            + np.sqrt(1 - rho_ax ** 2 - rho_ux ** 2 - rho_xbx ** 2) * rng.normal(size=n)
    )
    A_2 = rho_xa * X_2 + rho_xba * Xb + np.sqrt(1 - rho_xa ** 2 - rho_xbx ** 2) * rng.normal(size=n)

    A = pd.concat({
        1: pd.Series(A_1),
        2: pd.Series(A_2),
    }, names=["t", "id"]
    )
    A = A.rename("a")
    if with_step1:
        X = pd.concat({
            0: pd.DataFrame({0: Xb}),
            1: pd.DataFrame({0: Xb, 1: X_1}),
            2: pd.DataFrame({0: Xb, 1: X_2}),
        }, names=["t", "id"])
    else:
        X = pd.concat({
            0: pd.DataFrame({0: Xb}),
            1: pd.DataFrame({0: Xb}),
            2: pd.DataFrame({0: Xb, 1: X_2}),
        }, names=["t", "id"])
    if not with_baseline:
        X = X.drop(columns=0)
        X = X.dropna(how="all")
    X = X.add_prefix("x_")
    Y = pd.Series(U).rename("y")
    Y.index.set_names("id", inplace=True)
    # pd.pivot_table(A.to_frame("a"), columns="t", index="id")  .columns.to_flat_index()  # [f'{x}_{y}' for x,y in df.columns]
    return X, A, Y


def generate_multi_step_sim(
    n=1000,
    steps=5,
    rho_ax=0.75,
    rho_xa=0.5,
    rho_ux=0.5,
    rho_xbx=0.25,
    rho_xba=0.25,
    with_step1=False,
    with_baseline=False,
    bin_x=False,
    seed=0,
):
    rng = np.random.default_rng(seed)
    mu = 0
    sigma = 1

    U = rng.normal(mu, sigma, n)
    Xb = rng.normal(mu, sigma, n) if with_baseline else np.zeros(n)
    X = {}
    A = {}
    if with_step1:
        X_t = rho_ux * U + rho_xbx * Xb + np.sqrt(1 - rho_ux**2 - rho_xbx**2) * rng.normal(size=n)
        X[1] = rng.binomial(n=1, p=expit(X_t)) if bin_x else X_t
        # A_1 = 2*((X_1 > 0) - .5)
        A[1] = rho_xa * X[1] + rho_xba * Xb + np.sqrt(1 - rho_xa**2 - rho_xba**2) * rng.normal(size=n)
    else:
        A[1] = rho_xba * Xb + np.random.normal(size=n)  # np.random.normal(mu, sigma, n)

    for t in range(2, steps):
        X_t = (
            rho_ax * A[t-1]  # A_1
            + rho_ux * U
            + rho_xbx * Xb
            + np.sqrt(1 - rho_ax ** 2 - rho_ux ** 2 - rho_xbx ** 2) * rng.normal(size=n)
        )
        X_t = rng.binomial(n=1, p=expit(X_t)) if bin_x else X_t
        A_t = rho_xa * X_t + rho_xba * Xb + np.sqrt(1 - rho_xa ** 2 - rho_xba**2) * rng.normal(size=n)
        X[t] = X_t
        A[t] = A_t

    A = pd.concat(
        {k: pd.Series(v) for k, v in A.items()},
        names=["t", "id"]
    ).rename("a")
    X = pd.concat(
        {0: pd.DataFrame({0: Xb})} |
        ({1: pd.DataFrame({0: Xb})} if min(X.keys()) > 1 else {}) |
        {k: pd.DataFrame({0: Xb, 1: v}) for k, v in X.items()},
        names=["t", "id"]
    )
    if not with_baseline:
        X = X.drop(columns=0).dropna(how="all")
    X = X.add_prefix("x_")
    Y = pd.Series(U).rename("y")
    Y.index.set_names("id", inplace=True)
    # pd.pivot_table(A.to_frame("a"), columns="t", index="id")  .columns.to_flat_index()  # [f'{x}_{y}' for x,y in df.columns]
    return X, A, Y


class MyTestCase(unittest.TestCase):

    def test_simple_data_generation(self):
        X, a, y = generate_two_step_sim(with_baseline=False)
        Xb, ab, yb = generate_two_step_sim(with_step1=True, with_baseline=False)
        self.assertTupleEqual(ab.shape, a.shape)
        self.assertEqual(Xb.shape[0], X.shape[0] * 2)
        self.assertListEqual(Xb.index.get_level_values("t").unique().to_list(), [1, 2])
        self.assertListEqual(X.index.get_level_values("t").unique().to_list(), [2])
        self.assertListEqual(a.index.get_level_values("t").unique().to_list(), [1, 2])

    def test_fit_with_baseline_x(self):
        X, a, y = generate_multi_step_sim(with_baseline=True)
        X, a = X.reset_index(), a.reset_index()
        model = OrthogonalRegression(
            covariate_models={"x_1": LinearRegression()},
            id_col="id", time_col="t",
        )
        model.fit(X, a)

        self.assertSetEqual(set(model.covariate_models_.keys()), {2, 3, 4})
        self.assertSetEqual(set(model.covariate_models_[2].keys()), {"x_1"})  # `x_0` not modeled)
        self.assertEqual(model.covariate_models_[2]["x_1"].coef_.size, 2)  # baseline + treatment
        self.assertEqual(model.covariate_models_[3]["x_1"].coef_.size, 3)  # baseline + treatment + t-1
        self.assertEqual(model.covariate_models_[4]["x_1"].coef_.size, 3)  # baseline + treatment + t-1

        Xt = model.transform(X, a)
        self.assertTupleEqual(Xt.shape, (1000, 1 + 3 + 4))  # baseline + X_t + a_t

    def test_fit_without_baseline_x(self):
        X, a, y = generate_multi_step_sim(with_baseline=False)
        X, a = X.reset_index(), a.reset_index()
        model = OrthogonalRegression(
            covariate_models={"x_1": LinearRegression()},
            id_col="id", time_col="t",
        )
        model.fit(X, a)

        self.assertSetEqual(set(model.covariate_models_.keys()), {2, 3, 4})
        self.assertSetEqual(set(model.covariate_models_[2].keys()), {"x_1"})
        self.assertEqual(model.covariate_models_[2]["x_1"].coef_.size, 1)  # treatment
        self.assertEqual(model.covariate_models_[3]["x_1"].coef_.size, 2)  # treatment + t-1
        self.assertEqual(model.covariate_models_[4]["x_1"].coef_.size, 2)  # treatment + t-1

        Xt = model.transform(X, a)
        self.assertTupleEqual(Xt.shape, (1000, 3 + 4))  # X_t + a_t

    def test_intercept_only_model(self):
        X, a, y = generate_multi_step_sim(with_baseline=False, with_step1=True)
        X, a = X.reset_index(), a.reset_index()
        model = OrthogonalRegression(
            covariate_models={"x_1": LinearRegression()},
            id_col="id", time_col="t",
        )
        model.fit(X, a)

        self.assertSetEqual(set(model.covariate_models_.keys()), {1, 2, 3, 4})
        self.assertSetEqual(set(model.covariate_models_[1].keys()), {"x_1"})
        self.assertEqual(model.covariate_models_[1]["x_1"].coef_.size, 1)  # intercept
        self.assertEqual(model.covariate_models_[1]["x_1"].fit_intercept, False)
        self.assertListEqual(
            model.covariate_models_[1]["x_1"].feature_names_in_.tolist(),
            ["intercept"]
        )
        self.assertEqual(model.covariate_models_[2]["x_1"].coef_.size, 2)  # treatment + t-1
        self.assertEqual(model.covariate_models_[2]["x_1"].fit_intercept, True)
        self.assertEqual(model.covariate_models_[3]["x_1"].coef_.size, 2)  # treatment + t-1
        self.assertEqual(model.covariate_models_[4]["x_1"].coef_.size, 2)  # treatment + t-1

        Xt = model.transform(X, a)
        self.assertTupleEqual(Xt.shape, (1000, 4 + 4))  # X_t (with step1) + a_t

    def test_intercept_data(self):
        X, a, y = generate_multi_step_sim(with_baseline=False, with_step1=True)
        Xa = X.join(a).reset_index()
        model = OrthogonalRegression(
            covariate_models={"x_1": LinearRegression()},
            id_col="id", time_col="t",
        )
        cur_y = model._get_target(Xa, "x_1", 1)
        pd.testing.assert_series_equal(
            cur_y, X.xs(1, level="t")["x_1"]
        )
        intercept = model._get_predictors(Xa, 1, [0, 1, 2])
        pd.testing.assert_frame_equal(
            intercept,
            pd.Series(data=1, index=[0, 1, 2], name="intercept").to_frame()
        )

    def test_lag(self):
        X, a, y = generate_multi_step_sim(with_baseline=True, with_step1=True)
        X, a = X.reset_index(), a.reset_index()
        model = OrthogonalRegression(
            covariate_models={"x_1": LinearRegression()},
            id_col="id", time_col="t",
            lag=2,
        )
        model.fit(X, a)

        self.assertSetEqual(set(model.covariate_models_.keys()), {1, 2, 3, 4})
        self.assertSetEqual(set(model.covariate_models_[1].keys()), {"x_1"})
        self.assertEqual(model.covariate_models_[1]["x_1"].coef_.size, 1)  # baseline
        self.assertEqual(model.covariate_models_[2]["x_1"].coef_.size, 3)  # baseline + A_t-1 + X_t-1
        self.assertEqual(model.covariate_models_[3]["x_1"].coef_.size, 5)  # baseline + A_t-1 + A_t-2 + X_t-1 + X_t-2
        self.assertEqual(model.covariate_models_[4]["x_1"].coef_.size, 5)  # baseline + A_t-1 + A_t-2 + X_t-1 + X_t-2

        Xt = model.transform(X, a)
        self.assertTupleEqual(Xt.shape, (1000, 1 + 4 + 4))  # baseline + X_t (with step1) + a_t

    def test_compare_to_naive_regression(self):
        import statsmodels.api as sm

        def generate_simple_data(n=1000, seed=0):
            rng = np.random.default_rng(seed)
            U = rng.normal(size=n)
            A_1 = rng.normal(size=n)
            X_2 = 0.8 * A_1 + 0.5 * U + rng.normal(scale=np.sqrt(1 - 0.8**2 - 0.5**2), size=n)
            A_2 = 0.5 * X_2 + rng.normal(scale=np.sqrt(1 - 0.5**2), size=n)
            X = pd.concat({2: pd.Series(X_2)}, names=["t", "id"]).to_frame().add_prefix("x_")
            A = pd.concat({1: pd.Series(A_1), 2: pd.Series(A_2)}, names=["t", "id"]).rename("a")
            Y = pd.Series(U).rename("y")
            return X, A, Y

        # X, a, y = generate_multi_step_sim(
        #     steps=5,
        #     rho_ax=0.5,
        #     rho_ux=0.5,
        #     rho_xa=0.5,
        #     rho_xbx=0, rho_xba=0,
        #     with_baseline=False, with_step1=False
        # )
        X, a, y = generate_simple_data(10000)
        X, a = X.reset_index(), a.reset_index()
        model = OrthogonalRegression(
            covariate_models={"x_0": LinearRegression()},
            id_col="id", time_col="t",
        )
        model.fit(X, a)
        X_ortho = model.transform(X, a)
        X_naive = model._long_to_wide(
            pd.merge(X, a, on=["id", "t"], how="outer")
        )
        assert X_ortho.shape == X_naive.shape

        m_ortho = sm.OLS(y, sm.add_constant(X_ortho)).fit()
        m_naive = sm.OLS(y, sm.add_constant(X_naive)).fit()

        worst_ortho = m_ortho.params.filter(like="a", axis="index").abs().max()
        worst_naive = m_naive.params.filter(like="a", axis="index").abs().max()
        self.assertGreater(worst_naive, worst_ortho * 10)
        self.assertAlmostEquals(worst_ortho, 0, places=1)
        with self.assertRaises(AssertionError):
            self.assertAlmostEquals(worst_naive, 0, places=1)
        # TODO: why does last `A` has the same coefficient?

    def test_unseen_time_steps(self):
        X, a, y = generate_multi_step_sim(with_baseline=True, with_step1=True)
        X, a = X.reset_index(), a.reset_index()
        model = OrthogonalRegression(
            covariate_models={"x_1": LinearRegression()},
            id_col="id", time_col="t",
            lag=2,
        )
        model.fit(X, a)
        X["t"] += 10
        a["t"] += 10
        with self.assertRaises(ValueError):
            model.transform(X, a)

    def test_with_fancy_models(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import GridSearchCV

        estimator = GridSearchCV(
            GradientBoostingRegressor(),
            param_grid={
                "n_estimators": [5, 10, 20]
            },
            cv=3,  # Save some redundant CPU cycles
            refit=True,
        )

        with self.subTest("With baseline"):
            X, a, y = generate_multi_step_sim(with_baseline=True, with_step1=True)
            X, a = X.reset_index(), a.reset_index()
            model = OrthogonalRegression(
                covariate_models={"x_1": estimator},
                id_col="id", time_col="t",
            )
            model.fit(X, a)
            Xt = model.transform(X, a)
            self.assertTupleEqual(Xt.shape, (1000, 1 + 4 + 4))  # baseline + X_t (with step1) + a_t

        with self.subTest("No baseline, intercept"):
            X, a, y = generate_multi_step_sim(with_baseline=False, with_step1=True)
            X, a = X.reset_index(), a.reset_index()
            model = OrthogonalRegression(
                covariate_models={"x_1": estimator},
                id_col="id", time_col="t",
            )
            model.fit(X, a)
            Xt = model.transform(X, a)
            self.assertTupleEqual(Xt.shape, (1000, 4 + 4))  # X_t (with step1) + a_t

    def test_binary_covariate(self):
        with self.subTest("With baseline:"):
            X, a, y = generate_multi_step_sim(
                with_baseline=True, with_step1=False, bin_x=True
            )
            X, a = X.reset_index(), a.reset_index()
            model = OrthogonalRegression(
                covariate_models={"x_1": LogisticRegression()},
                id_col="id", time_col="t",
            )
            model.fit(X, a)
            self.assertSetEqual(set(model.covariate_models_.keys()), {2, 3, 4})

            Xt = model.transform(X, a)
            for col in Xt.filter(regex="x_1", axis=1):
                min_resid = Xt[col].min()
                max_resid = Xt[col].max()
                self.assertGreater(min_resid, -1)
                self.assertGreater(1, max_resid)

        with self.subTest("Intercept data"):
            X, a, y = generate_multi_step_sim(
                with_baseline=False, with_step1=True, bin_x=True
            )
            X, a = X.reset_index(), a.reset_index()
            model = OrthogonalRegression(
                covariate_models={"x_1": LogisticRegression()},
                id_col="id", time_col="t",
            )
            model.fit(X, a)

            self.assertSetEqual(set(model.covariate_models_.keys()), {1, 2, 3, 4})
            self.assertSetEqual(set(model.covariate_models_[1].keys()), {"x_1"})
            self.assertEqual(model.covariate_models_[1]["x_1"].coef_.size, 1)  # intercept
            self.assertEqual(model.covariate_models_[1]["x_1"].fit_intercept, False)
            self.assertListEqual(
                model.covariate_models_[1]["x_1"].feature_names_in_.tolist(),
                ["intercept"]
            )
            self.assertEqual(model.covariate_models_[2]["x_1"].coef_.size, 2)  # treatment + t-1
            self.assertEqual(model.covariate_models_[2]["x_1"].fit_intercept, True)
            Xt = model.transform(X, a)
            self.assertTupleEqual(Xt.shape, (1000, 4 + 4))  # X_t (with step1) + a_t

    def test_estimator_predict_binary(self):
        X, a, y = generate_multi_step_sim(
            with_baseline=False, with_step1=True, bin_x=True,
            steps=1,
        )
        X, a = X.reset_index(), a.reset_index()
        model = OrthogonalRegression(
            covariate_models={"x_1": LogisticRegression(random_state=0)},
            id_col="id", time_col="t",
        )
        model.fit(X, a)
        test_X = model._create_intercept_data(index=X["id"])
        x1_model = model.covariate_models_[1]["x_1"]
        y_pred = model._OrthogonalRegression__estimator_predict(x1_model, test_X)
        expected_y_pred = LogisticRegression(
            random_state=0,
        ).fit(test_X, X["x_1"]).predict_proba(test_X)[:, 1]  # binary
        np.testing.assert_array_almost_equal(y_pred, expected_y_pred, decimal=4)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(y_pred, 1 - expected_y_pred, decimal=4)

#
# if __name__ == '__main__':
#     unittest.main()
