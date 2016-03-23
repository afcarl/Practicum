import unittest
import numpy as np
import numpy.testing as npt

from problem_2 import get_vector, vec_get_vector, \
    fun_get_vector


class TestMyFunctions(unittest.TestCase):

    def test_small_data(self):
        m, n = 100, 30
        len_i = 15
        np.random.seed(12)
        x = np.random.rand(m, n)*5
        i, j = (np.random.randint(m, size=len_i),
                np.random.randint(n, size=len_i))
        npt.assert_allclose(get_vector(x, i, j), vec_get_vector(x, i, j))
        npt.assert_allclose(get_vector(x, i, j), fun_get_vector(x, i, j))

    def test_mid_data(self):
        m, n = 1000, 1000
        len_i = 700
        np.random.seed(12)
        x = np.random.rand(m, n)*5
        i, j = (np.random.randint(m, size=len_i),
                np.random.randint(n, size=len_i))
        npt.assert_allclose(get_vector(x, i, j), vec_get_vector(x, i, j))
        npt.assert_allclose(get_vector(x, i, j), fun_get_vector(x, i, j))

    def test_big_data(self):
        m, n = 5000, 7000
        len_i = 2000
        np.random.seed(12)
        x = np.random.rand(m, n)*5
        i, j = (np.random.randint(m, size=len_i),
                np.random.randint(n, size=len_i))
        npt.assert_allclose(get_vector(x, i, j), vec_get_vector(x, i, j))
        npt.assert_allclose(get_vector(x, i, j), fun_get_vector(x, i, j))


if __name__ == "__main__":
    unittest.main()
