import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import time
import scipy.stats as stat
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans


def visualize_data(points, indices, color_list=None):
    """
    Draw data points with different colours
    :param points: points
    :param indices: indices
    :param color_list: list of colors
    :return:
    """
    if points.shape[1] != 2:
        raise ValueError('Invalid data shape')
    num_gaussians = len(set(indices))
    if not color_list is None:
        if num_gaussians != len(color_list):
            Warning('Length of color list and the number of indices are not equal')

    if color_list is None:
        color_list = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo']
        for i in range(num_gaussians):
            plt.plot(points[indices == i, 0], points[indices == i, 1], color_list[i % len(color_list)])
        # plt.contour()


def generate_data(num, dim, num_gauss, random_state):
    """
    Generate data from a mixture of gaussians
    :param num: number of points
    :param dim: dimension
    :param num_gauss: number of gaussians
    :param random_state: random seed
    :return: (num, dim)-shaped numpy array
    """
    if num <= 0 or dim <= 0 or num_gauss <= 0:
        raise ValueError('Invalid value for one of the parameters')
    np.random.seed(random_state)
    gauss_index = np.random.randint(num_gauss, size=num)
    x = np.zeros((num, dim))
    for i in range(num_gauss):
        mu = np.random.rand(dim,)
        cov = np.random.rand(dim, dim)
        cov = cov.dot(cov.T) + np.eye(dim)
        cov /= 400
        num_i = np.sum(gauss_index == i)
        x[gauss_index == i] = np.random.multivariate_normal(mean=mu, cov=cov, size=num_i)
    return x, gauss_index


class GM:

    def __init__(self, number_of_gaussians, verbose=False, diagonal=False, min_var=1e-8):
        self.num_gauss = number_of_gaussians
        self.means, self.covs, self.mixprobs = [], [], None
        self.dim = None
        self.isfitted = False
        self.likelihood = -np.inf
        self.verbose = verbose
        self.diagonal = diagonal
        self.min_var = min_var

    def fit(self, x, num_runs=5, max_iter=100, eps=1e-1, random_state=241):

        if self.isfitted:
            raise Exception("This EM object is already fitted")

        n, d = x.shape
        self.dim = d
        K = self.num_gauss
        if K == 1:
            self.means = [np.sum(x, axis = 0) / n]
            mu = self.means[0]
            anc_mat = x - mu.reshape(1, d)
            if self.diagonal:
                diff = (x.T - mu[:, None])**2
                self.covs = [np.diag(np.sum(diff, axis=1)) / n + np.eye(d)*(self.min_var + 1e-4)]
            else:
                self.covs = [anc_mat.T.dot(anc_mat) / n + np.eye(d)*(self.min_var + 1e-4)]
            self.mixprobs = np.array([1])
            return
        scale = np.max(x) - np.min(x)
        means_scale = np.max(x)
        # scale, means_scale = 1., 1.
        # np.random.seed(random_state)
        for run in range(num_runs):
            if self.verbose:
                print('Run ', run)
            likelihood_lst = []
            w = np.ones((K, 1)) / K
            gamma = np.zeros((K, n))
            np.random.seed(random_state + run)
            mean_lst = [np.random.rand(d, 1)*means_scale for k in range(K)]
            covariance_lst = [np.eye(d)*scale/200] * K
            densities = np.zeros((K, n))
            old_likelihood = -np.inf
            for iteration in range(max_iter):
                for k in range(K):
                    mu = mean_lst[k]
                    cov = covariance_lst[k]
                    densities[k] = self.log_multivariate_normal_density(x, mu.T, cov)
                likelihood = np.sum(np.log((np.exp(densities.T)).dot(w)))
                if likelihood - old_likelihood < eps:
                    if not np.isinf(likelihood):
                        break
                max_density = np.max(densities, axis=0)
                densities -= max_density

                for k in range(K):
                    gamma[k] = np.exp(densities[k])
                gamma *= w
                gamma /= np.sum(gamma, axis=0)
                # gamma += 1e-8
                w = (np.sum(gamma, axis=1)/n)[:, None]
                if np.any(np.max(gamma, axis=1) == 0):
                    break
                # print(np.max(gamma, axis=1))
                gamma /= np.max(gamma, axis=1)[:, None]
                # print(np.max(gamma, axis=1))
                if self.diagonal:
                    for k in range(K):
                        mean_lst[k] = gamma[k].dot(x)[:, None] / np.sum(gamma[k])
                        diff = (x.T - mean_lst[k])**2
                        covariance_lst[k] = np.diag(diff.dot(gamma[k][:, None])[:, 0])
                        covariance_lst[k] /= np.sum(gamma[k])
                        covariance_lst[k] += np.eye(d) * 1e-4
                else:
                    for k in range(K):
                        mean_lst[k] = gamma[k].dot(x)[:, None] / np.sum(gamma[k])
                        diff = (x.T - mean_lst[k])
                        covariance_lst[k] = (gamma[k]*diff).dot(diff.T)
                        covariance_lst[k] /= np.sum(gamma[k])
                        covariance_lst[k] += np.eye(d) * 1e-4
                if self.verbose:
                    print('\tIteration', iteration, ': likelihood =', likelihood)
                likelihood_lst.append(likelihood)
                old_likelihood = likelihood
                # print(iteration)

            if likelihood > self.likelihood:
                self.likelihood = likelihood
                self.means = mean_lst
                self.covs = covariance_lst
                self.mixprobs = w
                best_likelihood_lst = likelihood_lst
        for cov in self.covs:
            cov += np.eye(d)*self.min_var
        return best_likelihood_lst

    def log_pdf(self, point):
        densities = np.zeros((self.num_gauss,))
        if self.num_gauss == 1:
            return self.log_multivariate_normal_density(point, self.means[0].T, self.covs[0])
        for k in range(self.num_gauss):
            densities[k] = self.log_multivariate_normal_density(point, self.means[k].T, self.covs[k])
        return np.sum(np.log((np.exp(densities.T)).dot(self.mixprobs)))

    def visualize(self, nstd=3):
        if self.dim != 2:
            raise ValueError("The dimensionality has to be equal to 2, to perform visualization")
        ax = plt.gca()
        means = self.means
        covariances = self.covs
        for i in range(len(means)):
            mean = means[i]
            cov = covariances[i]
            eigvals, eigvecs = np.linalg.eig(cov)
            width = 2 * nstd * np.sqrt(eigvals[0])
            height = 2 * nstd * np.sqrt(eigvals[1])
            angle = np.arccos(np.dot(eigvecs[0], np.array([1, 0])))
            # if (eigvecs[0])[0] > 0:
            #     angle = 2 * np.pi - angle
            ell = Ellipse(xy=mean, width=width, height=height, angle=np.degrees(angle))
            ell.set_alpha(0.2)
            ax.add_artist(ell)
        plt.show()

    def log_multivariate_normal_density(self, points, mean, covariance):
        """
        Logarithm of the Normal pdf
        :param points: points of evaluation of the pdf
        :param mean: mean of the distribution
        :param covariance: covariance matrix of the distribution
        :return: a vector of densities
        """
        d = points.shape[1]
        diff = (points - mean)
        if d == 1:
            diff = diff[:, 0]
            sigma = covariance[0, 0]
            return - np.log(2 * math.pi * sigma)/2 - np.square(diff)/(2 * sigma)
        if self.diagonal:
            l = np.diag(np.sqrt(np.diagonal(covariance)))
        else:
            covariance += np.eye(covariance.shape[0]) * 1e-4
            l = np.linalg.cholesky(covariance)
        anc_mat = sp.linalg.solve_triangular(l, diff.T)
        log_det = np.sum(np.log(np.diag(l)))
        return -np.square(np.linalg.norm(anc_mat, axis=0))/2 - d*np.log(2*math.pi)/2 - log_det

if __name__ == '__main__':
    num, dim, num_gauss = 500, 2, 3
    x, indices = generate_data(num, dim, num_gauss, 20)
    em = GM(num_gauss, diagonal=False, verbose=False)
    em.fit(x, num_runs=5, max_iter=1000, eps=1e-4, random_state=1)
    # print(em.log_pdf(np.array([[0]])))
    visualize_data(x, indices)
    em.visualize()
    plt.show()