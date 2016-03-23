import unittest
import numpy as np
import numpy.testing as npt
from scipy.misc import imread, imsave

from problem_5 import add_channels, fun_add_channels, vec_add_channels


class TestMyFunctions(unittest.TestCase):

    def setUp(self):
        self.pic_1 = imread("test_img.png")[:, :, :3]
        self.pic_2 = imread("test_img_2.png")[:, :, :3]
        self.pic_3 = imread("test_img_3.png")[:, :, :3]
        self.weights = np.array([0.299, 0.587, 0.114])

    def test_mid_data(self):
        npt.assert_allclose(add_channels(self.pic_1, self.weights),
                            vec_add_channels(self.pic_1, self.weights))
        npt.assert_allclose(add_channels(self.pic_1, self.weights),
                            fun_add_channels(self.pic_1, self.weights))

    def test_small_data(self):
        npt.assert_allclose(add_channels(self.pic_2, self.weights),
                            vec_add_channels(self.pic_2, self.weights))
        npt.assert_allclose(add_channels(self.pic_2, self.weights),
                            fun_add_channels(self.pic_2, self.weights))

    def test_big_data(self):
        npt.assert_allclose(add_channels(self.pic_3, self.weights),
                            vec_add_channels(self.pic_3, self.weights))
        npt.assert_allclose(add_channels(self.pic_3, self.weights),
                            fun_add_channels(self.pic_3, self.weights))


if __name__ == "__main__":
    unittest.main()
