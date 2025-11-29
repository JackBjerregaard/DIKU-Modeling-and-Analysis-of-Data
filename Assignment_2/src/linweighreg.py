import numpy

# NOTE: This template makes use of Python classes. If
# you are not yet familiar with this concept, you can
# find a short introduction here:
# http://introtopython.org/classes.html


class WeightedLinearRegression:
    """
    Weighted linear regression implementation.
    """

    def __init__(self):

        pass

    def fit(self, X, t, alpha):
        """
        Fits the weighted linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        alpha : Array of shape [n_samples, 1] - weights for each sample
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # converts 1d vector to matrix

        if alpha.ndim == 1:
            alpha = alpha.reshape(-1, 1)

        ones = numpy.ones((X.shape[0], 1))
        new_X = numpy.hstack([ones, X])

        # Create diagonal matrix A
        A = numpy.diag(alpha.flatten())

        XtA = new_X.T @ A        # X^T times A
        XtAX = XtA @ new_X       # (X^T A) times X
        XtAt = XtA @ t           # (X^T A) times t

        self.w = numpy.linalg.solve(XtAX, XtAt)

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

