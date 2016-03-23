import unittest
import numpy as np

from problem_4 import maximum_in_front_of_zero, fun_maximum_in_front_of_zero, \
    vec_maximum_in_front_of_zero


class TestMyFunctions(unittest.TestCase):

    def test_on_not_obvious_data(self):
        v_lst = []
        v_lst += [np.array([0, 1, 6, 2, 0, 3, 0, 0, 5, 7, 0, 9])]
        v_lst += [np.array([0, 9, 6, 2, 0, 3, 0, 0, 5, 7, 0, 9])]
        v_lst += [np.array([0, 1])]
        v_lst += [np.array([0, 0])]
        for v in v_lst:
            self.assertEqual(maximum_in_front_of_zero(v),
                             vec_maximum_in_front_of_zero(v))
            self.assertEqual(maximum_in_front_of_zero(v),
                             fun_maximum_in_front_of_zero(v))

    def test_small_data(self):
        vlen = 30
        num_zeros = 10
        np.random.seed(vlen)
        v = np.random.rand(vlen)
        zeros = np.random.randint(vlen, size=num_zeros)
        v[zeros] = 0
        self.assertEqual(maximum_in_front_of_zero(v),
                         vec_maximum_in_front_of_zero(v))
        self.assertEqual(maximum_in_front_of_zero(v),
                         fun_maximum_in_front_of_zero(v))

    def test_mid_data(self):
        vlen = 1000
        num_zeros = 10
        np.random.seed(vlen)
        v = np.random.rand(vlen)
        zeros = np.random.randint(vlen, size=num_zeros)
        v[zeros] = 0
        self.assertEqual(maximum_in_front_of_zero(v),
                         vec_maximum_in_front_of_zero(v))
        self.assertEqual(maximum_in_front_of_zero(v),
                         fun_maximum_in_front_of_zero(v))

    def test_big_data(self):
        vlen = 10000
        num_zeros = 10
        np.random.seed(vlen)
        v = np.random.rand(vlen)
        zeros = np.random.randint(vlen, size=num_zeros)
        v[zeros] = 0
        self.assertEqual(maximum_in_front_of_zero(v),
                         vec_maximum_in_front_of_zero(v))
        self.assertEqual(maximum_in_front_of_zero(v),
                         fun_maximum_in_front_of_zero(v))


if __name__ == "__main__":
    unittest.main()
