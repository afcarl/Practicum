import unittest
import numpy as np

from problem_3 import equal_multiset, vec_equal_multiset, \
    fun_equal_multiset


class TestMyFunctions(unittest.TestCase):

    def test_small_data_eq_ms(self):
        vlen = 30
        np.random.seed(20)
        mset = np.random.rand(vlen)
        np.random.seed(10)
        np.random.shuffle(mset)
        i = np.copy(mset)
        np.random.seed(15)
        np.random.shuffle(mset)
        j = mset
        self.assertTrue(equal_multiset(i, j))
        self.assertTrue(vec_equal_multiset(i, j))
        self.assertTrue(fun_equal_multiset(i, j))

    def test_mid_data_eq_ms(self):
        vlen = 1000
        np.random.seed(20)
        mset = np.random.rand(vlen)
        np.random.seed(10)
        np.random.shuffle(mset)
        i = np.copy(mset)
        np.random.seed(15)
        np.random.shuffle(mset)
        j = mset
        self.assertTrue(equal_multiset(i, j))
        self.assertTrue(vec_equal_multiset(i, j))
        self.assertTrue(fun_equal_multiset(i, j))

    def test_big_data_eq_ms(self):
        vlen = 10000
        np.random.seed(20)
        mset = np.random.rand(vlen)
        np.random.seed(10)
        np.random.shuffle(mset)
        i = np.copy(mset)
        np.random.seed(15)
        np.random.shuffle(mset)
        j = mset
        self.assertTrue(equal_multiset(i, j))
        self.assertTrue(vec_equal_multiset(i, j))
        self.assertTrue(fun_equal_multiset(i, j))

    def test_small_data_neq_ms(self):
        vlen = 30
        np.random.seed(12)
        i = np.random.rand(vlen, 1)
        np.random.seed(6)
        j = np.random.rand(vlen, 1)
        self.assertFalse(equal_multiset(i, j))
        self.assertFalse(vec_equal_multiset(i, j))
        self.assertFalse(fun_equal_multiset(i, j))

    def test_mid_data_neq_ms(self):
        vlen = 1000
        np.random.seed(12)
        i = np.random.rand(vlen)
        np.random.seed(6)
        j = np.random.rand(vlen)
        self.assertFalse(equal_multiset(i, j))
        self.assertFalse(vec_equal_multiset(i, j))
        self.assertFalse(fun_equal_multiset(i, j))

    def test_big_data_neq_ms(self):
        vlen = 10000
        np.random.seed(12)
        i = np.random.rand(vlen)
        np.random.seed(6)
        j = np.random.rand(vlen)
        self.assertFalse(equal_multiset(i, j))
        self.assertFalse(vec_equal_multiset(i, j))
        self.assertFalse(fun_equal_multiset(i, j))


if __name__ == "__main__":
    unittest.main()
