import numpy

# NOTE: This template makes use of Python classes. If
# you are not yet familiar with this concept, you can
# find a short introduction here:
# http://introtopython.org/classes.html


class LinearRegression:
    """
    Linear regression implementation.
    """

    def __init__(self):

        pass

    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # converts 1d vector to matrix

        ones = numpy.ones((X.shape[0], 1)) 
        new_X = numpy.hstack([ones, X])  # make new matrix with 1s

        # bulidng the normal equation - Aw = b, solving for w
        A = new_X.T @ new_X
        b = new_X.T @ t
        self.w = numpy.linalg.solve(A, b)

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """
        if X.ndim == 1: 
            X = X.reshape(-1,1)
        ones = numpy.ones((X.shape[0], 1))
        new_X = numpy.hstack([ones, X])

        return new_X @self.w

