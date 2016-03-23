import unittest
import numpy as np
import numpy.testing as npt
import scipy.stats as stat

from problem_8 import log_multivariate_normal_density, \
    vec_log_multivariate_normal_density, fun_log_multivariate_normal_density


class TestMyFunctions(unittest.TestCase):

    def test_small_data(self):
        d, n = 30, 50
        np.random.seed(d)
        mean_v = np.random.rand(d)
        A = np.random.rand(d, d)
        cov = A.dot(A.T) + np.eye(d)
        x = np.random.multivariate_normal(mean_v, cov, n)
        true_ans = stat.multivariate_normal(mean_v, cov).logpdf(x)
        n_ans = log_multivariate_normal_density(x, mean_v, cov)
        v_ans = vec_log_multivariate_normal_density(x, mean_v, cov)
        f_ans = fun_log_multivariate_normal_density(x, mean_v, cov)
        npt.assert_allclose(true_ans, n_ans)
        npt.assert_allclose(true_ans, v_ans)
        npt.assert_allclose(true_ans, f_ans)

    def test_mid_data(self):
        d, n = 300, 100
        np.random.seed(d)
        mean_v = np.random.rand(d)
        A = np.random.rand(d, d)
        cov = A.dot(A.T) + np.eye(d)
        x = np.random.multivariate_normal(mean_v, cov, n)
        true_ans = stat.multivariate_normal(mean_v, cov).logpdf(x)
        n_ans = log_multivariate_normal_density(x, mean_v, cov)
        v_ans = vec_log_multivariate_normal_density(x, mean_v, cov)
        f_ans = fun_log_multivariate_normal_density(x, mean_v, cov)
        npt.assert_allclose(true_ans, n_ans)
        npt.assert_allclose(true_ans, v_ans)
        npt.assert_allclose(true_ans, f_ans)

    def test_big_data(self):
        d, n = 1000, 1000
        np.random.seed(d)
        mean_v = np.random.rand(d)
        A = np.random.rand(d, d)
        cov = A.dot(A.T) + np.eye(d)
        x = np.random.multivariate_normal(mean_v, cov, n)
        true_ans = stat.multivariate_normal(mean_v, cov).logpdf(x)
        v_ans = vec_log_multivariate_normal_density(x, mean_v, cov)
        f_ans = fun_log_multivariate_normal_density(x, mean_v, cov)
        npt.assert_allclose(true_ans, v_ans)
        npt.assert_allclose(true_ans, f_ans)

if __name__ == "__main__":
    unittest.main()