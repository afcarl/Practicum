import unittest
import numpy as np
import numpy.testing as npt

from problem_6 import run_length_encoding, vec_run_length_encoding, \
    fun_run_length_encoding


class TestMyFunctions(unittest.TestCase):

    def test_not_obvious_data(self):
        v = np.array([0])
        nv = run_length_encoding(v)
        vv = vec_run_length_encoding(v)
        fv = fun_run_length_encoding(v)
        npt.assert_allclose(nv[0], vv[0])
        npt.assert_allclose(nv[1], vv[1])
        npt.assert_allclose(nv[0], fv[0])
        npt.assert_allclose(nv[1], fv[1])

    def test_small_data(self):
        vlen = 30
        max_num = 8
        np.random.seed(vlen)
        v = np.random.randint(max_num, size=vlen)
        nv = run_length_encoding(v)
        vv = vec_run_length_encoding(v)
        fv = fun_run_length_encoding(v)
        npt.assert_allclose(nv[0], vv[0])
        npt.assert_allclose(nv[1], vv[1])
        npt.assert_allclose(nv[0], fv[0])
        npt.assert_allclose(nv[1], fv[1])

    def test_mid_data(self):
        vlen = 1000
        max_num = 20
        np.random.seed(vlen)
        v = np.random.randint(max_num, size=vlen)
        nv = run_length_encoding(v)
        vv = vec_run_length_encoding(v)
        fv = fun_run_length_encoding(v)
        npt.assert_allclose(nv[0], vv[0])
        npt.assert_allclose(nv[1], vv[1])
        npt.assert_allclose(nv[0], fv[0])
        npt.assert_allclose(nv[1], fv[1])

    def test_big_data(self):
        vlen = 10000
        max_num = 500
        np.random.seed(vlen)
        v = np.random.randint(max_num, size=vlen)
        nv = run_length_encoding(v)
        vv = vec_run_length_encoding(v)
        fv = fun_run_length_encoding(v)
        npt.assert_allclose(nv[0], vv[0])
        npt.assert_allclose(nv[1], vv[1])
        npt.assert_allclose(nv[0], fv[0])
        npt.assert_allclose(nv[1], fv[1])


if __name__ == "__main__":
    unittest.main()
