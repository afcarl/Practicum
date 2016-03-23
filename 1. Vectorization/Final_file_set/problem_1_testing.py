import unittest
import numpy as np

from problem_1 import non_zero_prod, vec_non_zero_prod, \
    fun_non_zero_prod


class TestMyFunctions(unittest.TestCase):

    def test_one_element_matrix(self):
        m, n = 1, 1
        np.random.seed(5)
        x = np.random.rand(m, n)*5
        self.assertEqual(non_zero_prod(x), vec_non_zero_prod(x))
        self.assertEqual(non_zero_prod(x), fun_non_zero_prod(x))

    def test_small_data(self):
        m, n = 100, 30
        np.random.seed(12)
        x = np.random.rand(m, n)*5
        zeros = np.random.randint(min(m, n), size=int(min(m, n)/4))
        x[zeros, zeros] = 0
        self.assertEqual(non_zero_prod(x), vec_non_zero_prod(x))
        self.assertEqual(non_zero_prod(x), fun_non_zero_prod(x))

    def test_mid_data(self):
        m, n = 1000, 1000
        np.random.seed(3)
        x = np.random.rand(m, n)*2.9
        zeros = np.random.randint(min(m, n), size=int(min(m, n)/4))
        x[zeros, zeros] = 0
        self.assertEqual(non_zero_prod(x), vec_non_zero_prod(x))
        self.assertEqual(non_zero_prod(x), fun_non_zero_prod(x))

    def test_big_data(self):
        m, n = 5000, 7000
        np.random.seed(8)
        x = np.random.rand(m, n)*2.8
        zeros = np.random.randint(min(m, n), size=int(min(m, n)/4))
        x[zeros, zeros] = 0
        self.assertEqual(non_zero_prod(x), vec_non_zero_prod(x))
        self.assertEqual(non_zero_prod(x), fun_non_zero_prod(x))

if __name__ == "__main__":
    unittest.main()
