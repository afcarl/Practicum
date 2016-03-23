import cvxopt
import numpy as np
import matplotlib.pyplot as plt
import time
from math import ceil
from sklearn.svm import LinearSVC, SVC


class SVM:
    def __init__(self, C=1., gamma=0., solver='dual_qp', tol=1e-6, max_iter=100, verbose=False):
        """
        :param C: the penalty for violating the gap
        :param gamma: RBF kernel width. If 0, linear kernel is used
        :param solver: The svm optimization solver being used: 'primal_qp', 'dual_qp', 'subgradient', 'liblinear' or
        'libsvm'. If gamma is nonzero (so that RBF kernel is used). The default value is 'dual_qp'
        :return: SVM object
        """
        if not (solver in {'primal_qp', 'dual_qp', 'subgradient', 'liblinear', 'libsvm'}):
            raise ValueError("Unknown solver")
        if gamma > 0 and (solver in {'primal_qp', 'subgradient', 'liblinear'}):
            raise ValueError("Solver " + solver + " is incompatible with RBF kernel")
        if C < 0 or gamma < 0:
            raise ValueError("C and gamma must be non-negative")

        self.C = C
        self.gamma = gamma
        self.solver = solver
        self.dual = solver in {'dual_gp', 'liblinear', 'libsvm'}
        self.sklearn_predictor = None
        self.primal_weights = None
        self.dual_weights = None
        self.data = None
        self.labels = None
        self.support_vectors = None
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, X, y, *args, **kwargs):
        """
        Fit SVM to the data.
        :param X: (N, D)-shaped array, training set
        :param y: (N, 1)-shaped array, labels for training set
        :param args: additional arguments passed to the solver
        :param kwargs: additional keyword-arguments passed to the solver
        :return: a dictionary {'w': primal weights, 'A': dual weights (for dual problem only),
        'status': 0 if method achieved the required accuracy, 1 otherwise,
        'objective_curve': list of function values iteration-wise, 'time': the time it took the method to converge}
        """
        self.check_data(X, y)
        self.data, self.labels = X, y
        return{
            'primal_qp': self.svm_qp_primal_solver,
            'dual_qp': self.svm_qp_dual_solver,
            'subgradient': self.svm_subgradient_solver,
            'liblinear': self.svm_liblinear_solver,
            'libsvm': self.svm_libsvm_solver
        }[self.solver](*args, **kwargs)

    def predict(self, X, sign=True):
        """
        Predict class labels for new points
        :param X: (N, D)-shaped array, data set
        :return: (N, 1)-shaped array, labels for the data set
        """
        if not self.sklearn_predictor is None:
            return self.sklearn_predictor.predict(X)

        # if not self.primal_weights is None:
        if self.gamma == 0.:
            return np.sign(np.hstack((X, np.ones((X.shape[0], 1)))).dot(self.primal_weights))[:, 0]

        if self.dual_weights is None:
            raise AttributeError("It looks like the SVM is not fitted")

        A = self.dual_weights
        K = self.K_mat(self.data, X)
        if not sign:
            return (np.sum(K * self.labels * A, axis=0) + self.primal_weights[-1, 0])
        return np.sign(np.sum(K * self.labels * A, axis=0) + self.primal_weights[-1, 0])

    @staticmethod
    def plot_data(points, labels, pos_color='bx', neg_color='ro'):
        """Visualization"""
        if not isinstance(points, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError("The first two arguments must be numpy arrays")
        if points.shape[1] != 2:
            raise ValueError("Points array must be 2-dimensional")

        loc_labels = labels.reshape((labels.shape[0],))
        plt.plot(points[loc_labels == 1, 0], points[loc_labels == 1, 1], pos_color)
        plt.plot(points[loc_labels == -1, 0], points[loc_labels == -1, 1], neg_color)

    def visualize(self):#, *args, **kwargs):
        """
        Draw data points and the separating surface
        :return: None
        """
        X, y = self.data, self.labels
        self.check_data(X, y)
        if (X.shape[1] != 2):
            raise ValueError("Only 2-d data is supported")

        test_density = 500
        left, right = np.min(X[:, 0]), np.max(X[:, 0])
        diff = (right - left)
        left -= diff/20
        right += diff/20
        down, top = np.min(X[:, 1]), np.max(X[:, 1])
        diff = (top - down)
        down -= diff/20
        top +=diff/20

        plt.axis("off")
        if self.solver != 'liblinear':
            if self.support_vectors is None:
                self.compute_support_vectors()
            plt.plot(self.support_vectors[:, 0], self.support_vectors[:, 1], 'wo', markersize=15)
        self.plot_data(X, y)
        x_1 = np.linspace(left, right, test_density)
        x_2 = np.linspace(down, top, test_density)
        x_1, x_2 = np.meshgrid(x_1, x_2)
        x_test = np.array(list(zip(x_1.reshape(-1).tolist(), x_2.reshape(-1).tolist())))
        predicted = self.predict(x_test, sign=False)
        plt.contour(x_1, x_2, predicted.reshape((test_density, test_density)), levels=[0], colors='k')

    def compute_primal_objective(self, X=None, y=None, w=None):
        """
        Computes SVM's loss function
        :param X: (N, D)-shaped array, training set
        :param y: (N, 1)-shaped array, labels for training set
        :param w: (D+1, 1)-shaped array, SVM's weights
        :return: loss at point w
        """
        if X is None:
            X = self.data
        if y is None:
            y = self.labels
        # if self.gamma != 0:
        #     raise ValueError("This function is implemented only for the linear kernel")
        if w is None:
            w = self.primal_weights
        if w is None:
            raise ValueError("No weights provided")
        self.check_w(X, y, w)
        errors = np.max(np.hstack((1 - y * (X.dot(w[:-1]) + w[-1]), np.zeros(y.shape))), axis=1)[:, None]
        return (np.dot(w[:-1].T, w[:-1]) / 2 + self.C * np.sum(errors, axis=0))[0, 0]

    def compute_dual_objective(self, X=None, y=None, A=None):
        """
        Computes SVM's dual loss function
        :param X: (N, D)-shaped array, training set
        :param y: (N, 1)-shaped array, labels for training set
        :param A: (N, 1)-shaped array, SVM's dual weights
        :return: loss at point w
        """
        if X is None:
            X = self.data
        if y is None:
            y = self.labels
        if A is None:
            A = self.dual_weights
        if A is None:
            raise ValueError("No dual weights provided")
        self.check_a(X, y, A)
        num, dim = X.shape
        K = self.K_mat(X, X)
        Q = K * (y.dot(y.T))
        p = -np.ones((num,))
        return (-A.T.dot(Q.dot(A))/2 + p.dot(A))[0, 0]

    def compute_primal_batch_subgradient(self, X, y, w, low, high):
        """
        :param X: (N, D)-shaped array, training set
        :param y: (N, 1)-shaped array, labels for training set
        :param w: (D+1, 1)-shaped array, SVM's weights
        :param low: lower bound of the batch
        :param high: upper bound of the batch
        :return: subgradient (one of them) at point w
        """

        loc_x = X[low:high]
        loc_y = y[low:high]
        errors = np.sign(np.max(np.hstack((1 - loc_y * (loc_x.dot(w[:-1]) + w[-1]), np.zeros(loc_y.shape))),
                                axis=1)[:, None])
        w_derivative = w[:-1] - self.C * np.sum(errors * loc_y * loc_x, axis=0)[:, None]
        w0_derivative = -self.C * np.sum(errors * loc_y)
        return np.vstack((w_derivative, w0_derivative))

    @classmethod
    def check_data(self, X, y):
        """
        Checking the inputs for optimization methods
        :param X: (N, D)-shaped array, training set
        :param y: (N, 1)-shaped array, labels for training set
        :return: None
        """
        if not(isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            raise TypeError("X and y must be numpy arrays")
        if not(X.shape[0] == y.shape[0]):
            raise ValueError("X, y must have equal shapes along the dimension 0")
        if y.shape[1] != 1:
            raise ValueError("y and must have shape like (*, 1))")

    @classmethod
    def check_w(self, X, y, w):
        """
        Checking the inputs for objective functions and function, computing gradients
        :param X: (N, D)-shaped array, training set
        :param y: (N, 1)-shaped array, labels for training set
        :param w: (D+1, 1)-shaped array, SVM's weights
        :return: None
        """
        self.check_data(X, y)
        if not isinstance(w, np.ndarray):
            raise TypeError("w must be a numpy array")
        if y.shape[1] != 1 or w.shape[1] != 1:
            raise ValueError("y and w must  have shapes like (*, 1))")
        if X.shape[1] != w.shape[0] - 1:
            raise ValueError("X.shape[1] and w.shape[0]-1 must be equal")

    def check_a(self, X, y, A):
        """
        :param X: (N, D)-shaped array, training set
        :param y: (N, 1)-shaped array, labels for training set
        :param A: (N, 1)-shaped array, dual weights
        :return: None
        """
        self.check_data(X, y)
        if not isinstance(A, np.ndarray):
            raise TypeError("A must be a numpy.array")
        if A.shape != y.shape:
            raise ValueError("A is of the wrong shape")

    def K_mat(self, x, y):
        """
        :param x: (N, D)-shaped array
        :param y: (N, D)-shaped array
        :return: matrix K, such that K_ij = k(x_i, x_j), where k is the kernel function
        """
        if self.gamma == 0:
            return x.dot(y.T)
        x_norm = np.linalg.norm(x, axis=1)[:, None]
        y_norm = np.linalg.norm(y, axis=1)[None, :]
        d = np.square(x_norm) + np.square(y_norm) - 2 * x.dot(y.T)
        return np.exp(-self.gamma * d)

    def compute_w(self, X=None, y=None, A=None):
        """
        Compute primal weights given dual weights for linear kernel.
        :param X: (N, D)-shaped array, training set
        :param y: (N, 1)-shaped array, labels for training set
        :param A: (N, 1)-shaped array, dual weights
        :return: primal weights
        """
        if X is None:
            X = self.data
        if y is None:
            y = self.labels
        if A is None:
            A = self.dual_weights
        if A is None:
            raise ValueError("No dual weights provided")
        self.check_a(X, y, A)
        eps = 1e-5
        w = np.sum(A * y * X, axis=0)[:, None]
        return w

    def compute_support_vectors(self):
        """
        Compute support vectors in the training data set
        :return: None
        """
        # if self.solver in {'libsvm'}:
        self.support_vectors = self.data[np.abs(self.dual_weights[:, 0]) > self.tol]
        # else:
        #     self.support_vectors = self.data[self.dual_weights[:, 0] > self.tol]

    def svm_qp_primal_solver(self, obj_curve=False):
        """
        Solve the primal optimization problem of the SVM using interior point method
        """
        X, y = self.data, self.labels
        if self.gamma != 0:
            raise ValueError("This function is implemented only for the linear kernel")
        num, dim = X.shape
        Q = np.hstack((np.eye(dim), np.zeros((dim, num + 1))))
        Q = np.vstack((Q, np.zeros((num + 1, Q.shape[1]))))
        p = np.vstack((np.zeros((dim + 1, 1)), self.C * np.ones((num, 1))))
        G = -np.vstack((np.hstack((X * y, y, np.eye(num))),
                       np.hstack((np.zeros((num, dim + 1)),  np.eye(num)))))
        h = -np.vstack((np.ones(y.shape), np.zeros((num, 1))))
        Q = cvxopt.matrix(Q)
        p = cvxopt.matrix(p)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)

        cvxopt.solvers.options['show_progress'] = self.verbose
        cvxopt.solvers.options['maxiters'] = self.max_iter
        cvxopt.solvers.options['abstol'] = self.tol
        # cvxopt.solvers.options['feastol'] = self.tol

        start = time.time()
        solution = cvxopt.solvers.qp(Q, p, G, h)
        finish = time.time()
        self.dual_weights = np.array(solution['z'][:num, 0])
        self.primal_weights = np.array(solution['x'])[:dim+1, :]

        fun_lst = None
        if obj_curve:
            fun_lst = []
            cvxopt.solvers.options['show_progress'] = False
            cvxopt.solvers.options['abstol'] = 1e-15
            cvxopt.solvers.options['reltol'] = 1e-15
            cvxopt.solvers.options['feastol'] = 1e-15
            for i in range(self.max_iter):
                cvxopt.solvers.options['maxiters'] = i + 1
                solution = cvxopt.solvers.qp(Q, p, G, h)
                fun_lst.append(self.compute_primal_objective(w=np.array(solution['x'])[:dim+1, :]))
        return {'w': self.primal_weights, 'status': int(solution['status'] != 'optimal'), 'objective_curve': fun_lst,
                'time': finish - start}

    def svm_qp_dual_solver(self, obj_curve=False):
        """
        Solve the dual optimization problem of the SVM using interior point method
        """
        X, y = self.data, self.labels
        num, dim = X.shape
        K = self.K_mat(X, X)
        Q = cvxopt.matrix(K * (y.dot(y.T)))
        p = cvxopt.matrix(-np.ones((num,)))
        G = cvxopt.matrix(np.vstack((-np.eye(y.size), np.eye(y.size))))
        h = cvxopt.matrix(np.vstack((np.zeros(y.shape), np.ones(y.shape) * self.C)))
        A = cvxopt.matrix(np.ones((1, num)) * y.T)
        b = cvxopt.matrix(0.)
        cvxopt.solvers.options['show_progress'] = self.verbose
        cvxopt.solvers.options['maxiters'] = self.max_iter
        cvxopt.solvers.options['abstol'] = self.tol
        start = time.time()
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)
        finish = time.time()
        # if self.gamma > 0:

        a = np.array(solution['x'])
        self.primal_weights = np.vstack((self.compute_w(X, y, a), np.array(solution['y'])))
        self.dual_weights = a

        fun_lst = None
        if obj_curve:
            fun_lst = []
            cvxopt.solvers.options['show_progress'] = False
            cvxopt.solvers.options['abstol'] = 1e-15
            cvxopt.solvers.options['reltol'] = 1e-15
            cvxopt.solvers.options['feastol'] = 1e-15
            for i in range(self.max_iter):
                cvxopt.solvers.options['maxiters'] = i + 1
                solution = cvxopt.solvers.qp(Q, p, G, h, A, b)
                w = np.vstack((self.compute_w(X, y, np.array(solution['x'])), np.array(solution['y'])))
                fun_lst.append(self.compute_primal_objective(w=w))

        return {'w': self.primal_weights, 'A': self.dual_weights, 'status': int(solution['status'] != 'optimal'),
                'objective_curve': fun_lst, 'time': finish - start}

    def svm_subgradient_solver(self, batch_num=1, alpha=None, beta=1):
        """
        Solve the primal unconstrained non-smooth optimization problem of the SVM using subgradient descent
        :param batch_num: number of batches for stochastic subgradient descent; batch_num == 1 is equivalent to using
           full subgradient descent
        :param alpha: used in step size rule
        :param beta: used in step size rule
        """
        X, y = self.data, self.labels
        if alpha is None:
            alpha = 1 / self.C
        def step_size(alpha, beta):
            """
            Coefficient sequence generator
            :param step0: first step size
            :param gamma: beta_n = step0 / n^gamma
            :return: beta_n
            """
            i = 1
            while True:
                yield alpha / batch_num * np.power((i + 1), beta)
                i += 1

        batch_size = ceil(y.size / batch_num)
        num, dim = X.shape
        w = np.ones((dim + 1, 1))
        fun_lst = []
        steps = step_size(alpha=alpha, beta=beta)
        optim = 1

        loc_data = np.hstack((X, y))

        start = time.time()
        fun = self.compute_primal_objective(X, y, w)
        fun_lst.append(fun)
        grad = self.compute_primal_batch_subgradient(X, y, w, 0, batch_size)
        for iteration_counter in range(self.max_iter * batch_num):
            i = iteration_counter % batch_num
            if i == 0:
                np.random.shuffle(loc_data)
                loc_y, loc_x = (loc_data[:, -1])[:, None], loc_data[:, :-1]
                step = next(steps)
            w -= step * grad / np.linalg.norm(grad)
            new_fun = self.compute_primal_objective(loc_x, loc_y, w)
            new_grad = self.compute_primal_batch_subgradient(loc_x, loc_y, w, i * batch_size, (i+1) * batch_size)
            # new_grad = self.compute_primal_batch_subgradient(X, y, w, 0, y.size)

            if self.verbose:
                print("Iteration ", iteration_counter, ":", "loss = ", fun)
            if np.abs(new_fun - fun) < self.tol or step < self.tol:
                optim = 0
                break
            fun, grad = new_fun, new_grad
            if i == 0:
                fun_lst.append(fun)
        finish = time.time()
        self.primal_weights = w
        errors = np.sign(np.max(np.hstack((1 - y * (X.dot(w[:-1]) + w[-1]), np.zeros(loc_y.shape))),
                                axis=1))
        self.support_vectors = X[errors > 0]
        return {'w': w, 'status': optim, 'objective_curve': fun_lst, 'time': finish - start}

    def svm_liblinear_solver(self, obj_curve=False):
        """
        Solve the dual optimization problem of the SVM using liblinear solver
        """
        X, y = self.data, self.labels
        svm = LinearSVC(C=self.C, tol=self.tol, loss='hinge', max_iter=self.max_iter, verbose=self.verbose,
                        random_state=241, intercept_scaling=100)
        start = time.time()
        svm.fit(X, y[:, 0])
        finish = time.time()

        self.sklearn_predictor = svm
        w = np.vstack((svm.coef_.T, np.array(svm.intercept_)))
        self.primal_weights = w

        fun_lst = None
        if obj_curve:
            fun_lst = []
            for i in range(self.max_iter):
                svm = LinearSVC(C=self.C, tol=1e-20, loss='hinge', max_iter=i + 1, verbose=False, random_state=241,
                        intercept_scaling=100)
                svm.fit(X, y[:, 0])
                w = np.vstack((svm.coef_.T, np.array(svm.intercept_)))
                fun_lst.append(self.compute_primal_objective(w=w))
        return {'w': w, 'A': None, 'status': None, 'objective_curve': fun_lst, 'time': finish - start}

    def svm_libsvm_solver(self, obj_curve=False):
        """
        Solve the dual optimization problem of the SVM using libsvm method
        """
        X, y = self.data, self.labels
        kernel = 'rbf'
        if self.gamma == 0.:
            kernel = 'linear'
        svm = SVC(C=self.C, kernel=kernel, gamma=self.gamma, tol=self.tol, verbose=self.verbose, max_iter=self.max_iter,
                  random_state=241)
        start = time.time()
        svm.fit(X, y[:, 0])
        finish = time.time()
        self.sklearn_predictor = svm
        self.support_vectors = svm.support_vectors_
        indices = svm.support_
        self.dual_weights = np.zeros((y.size, 1))
        self.dual_weights[indices, :] = svm.dual_coef_.T
        self.dual_weights *= y
        if self.gamma == 0:
            self.primal_weights = np.vstack((svm.coef_.T, np.array(svm.intercept_)))
        else:
            self.primal_weights = None

        fun_lst = None
        if obj_curve and self.gamma != 0:
            fun_lst = []
            for i in range(self.max_iter):
                svm = SVC(C=self.C, kernel=kernel, gamma=self.gamma, tol=1e-20, verbose=False,
                          max_iter=i+1, random_state=241)
                svm.fit(X, y[:, 0])
                w = np.vstack((svm.coef_.T, np.array(svm.intercept_)))
                fun_lst.append(self.compute_primal_objective(w=w))
        if self.gamma == 0 and obj_curve:
            print("Objective curve is only implemented for linear kernel by now")
        return {'w': self.primal_weights, 'A': self.dual_weights, 'status': None, 'objective_curve': fun_lst,
                'time': finish - start}