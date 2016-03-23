import unittest
import numpy as np
import numpy.testing as npt
import scipy.spatial.distance as spd

from problem_7 import find_pairwise_distances, vec_find_pairwise_distances, \
    fun_find_pairwise_distances


class TestMyFunctions(unittest.TestCase):

    def test_small_data(self):
        d, m, n = 2, 5, 10
        np.random.seed(d)
        x = np.random.rand(m, d)
        y = np.random.rand(n, d)
        true_ans = spd.cdist(x, y)
        n_ans = find_pairwise_distances(x, y)
        v_ans = vec_find_pairwise_distances(x, y)
        f_ans = fun_find_pairwise_distances(x, y)
        npt.assert_allclose(n_ans, true_ans)
        npt.assert_allclose(v_ans, true_ans)
        npt.assert_allclose(f_ans, true_ans)

    def test_min_data(self):
        d, m, n = 30, 100, 50
        np.random.seed(d)
        x = np.random.rand(m, d)
        y = np.random.rand(n, d)
        true_ans = spd.cdist(x, y)
        n_ans = find_pairwise_distances(x, y)
        v_ans = vec_find_pairwise_distances(x, y)
        f_ans = fun_find_pairwise_distances(x, y)
        npt.assert_allclose(n_ans, true_ans)
        npt.assert_allclose(v_ans, true_ans)
        npt.assert_allclose(f_ans, true_ans)

    def test_big_data(self):
        d, m, n = 100, 500, 1000
        np.random.seed(d)
        x = np.random.rand(m, d)
        y = np.random.rand(n, d)
        true_ans = spd.cdist(x, y)
        n_ans = find_pairwise_distances(x, y)
        v_ans = vec_find_pairwise_distances(x, y)
        f_ans = fun_find_pairwise_distances(x, y)
        npt.assert_allclose(n_ans, true_ans)
        npt.assert_allclose(v_ans, true_ans)
        npt.assert_allclose(f_ans, true_ans)

if __name__ == "__main__":
    unittest.main()
