import numpy as np
from abc import ABC, abstractmethod
from typing import Callable

from chapter2 import invert
from chapter4 import det


class LRKernel(ABC):
    """ Abstract class for feature kernels in linear regression.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ Compute the feature matrix for the given input data.

        x: (N, D) - The input data

        Returns: (N, K) - The feature matrix
        """

    @abstractmethod
    def predict(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """ Predict the output for the given input data.

        x: (N, D) - The input data
        coeffs: (K) - The coefficients

        Returns: (N) - The predicted output
        """

    @abstractmethod
    def format(self, coeffs: np.ndarray) -> str:
        """ Format the coefficients of the kernel into a human-readable string.

        coeffs: (K) - The coefficients

        Returns: str - The formatted string
        """


class PolynomialKernel(LRKernel):
    """ Polynomial feature kernel for linear regression.
    """

    def __init__(self, degree=1):
        self.degree = degree

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.hstack([x**i for i in range(self.degree+1)])

    def predict(self, x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        return self(x) @ coeffs

    def format(self, coeffs: np.ndarray) -> str:
        return ' + '.join([f'{c:.2f}x^{i}' for i, c in enumerate(coeffs[::-1])])


def plot_estimate(x: np.ndarray, y: np.ndarray, estimator: Callable[[np.ndarray, np.ndarray, LRKernel, ...], np.ndarray], kernel: LRKernel, **estimator_kwargs):
    """ Plot the data and the regression curve.

    x: (N, D) - The input data
    y: (N) - The output data
    estimator: Callable[[np.ndarray, np.ndarray, LRKernel, ...], np.ndarray] - The estimator function
    kernel: LRKernel - The feature kernel to use
    estimator_kwargs: ... - Additional arguments to pass to the estimator
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("You need to install the matplotlib package to plot the data.")
    coeffs = estimator(x, y, kernel, **estimator_kwargs)
    y_pred = kernel.predict(x, coeffs)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    sigma = np.sqrt(np.mean((y - y_pred)**2))
    plt.fill_between(x[:, 0], y_pred - sigma, y_pred + sigma, color='gray', alpha=0.3, label="±1 std noise")
    plt.fill_between(x[:, 0], y_pred - 2*sigma, y_pred + 2*sigma, color='gray', alpha=0.1, label="±2 std noise")
    plt.scatter(x[:, 0], y, label="Data points")
    plt.plot(x[:, 0], y_pred, 'r', label=f"Regression Curve: {kernel.format(coeffs)}")

    # Plot title and labels
    plt.title(f"{estimator.__name__}\nRMSE: {rmse:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # Show the plot
    plt.show()


def estimate_mle(x: np.ndarray, y: np.ndarray, kernel: LRKernel) -> np.ndarray:
    """ Compute the Maximum Likelihood Estimate of the data using the given feature kernel.

    x: (N, D) - The input data
    y: (N) - The output data
    kernel: LRKernel - The feature kernel to use

    Returns: (K) - The estimated coefficients
    """
    x = x
    y = y[:, np.newaxis]
    A = kernel(x)
    w = invert(A.T @ A) @ A.T @ y
    return w[:, 0]


def estimate_map(x: np.ndarray, y: np.ndarray, kernel: LRKernel, beta2: np.ndarray) -> np.ndarray:
    """ Compute the Maximum A Posteriori Estimate of the data using the given feature kernel.

    x: (N, D) - The input data
    y: (N) - The output data
    beta2: (1) - The prior's variance
    kernel: LRKernel - The feature kernel to use

    Returns: (K) - The estimated coefficients
    """
    x = x
    y = y[:, np.newaxis]
    A = kernel(x)
    AA = A.T @ A
    w = invert(AA + 1/beta2 * np.eye(AA.shape[0])) @ A.T @ y
    return w[:, 0]


# BAYESIAN INFERENCE


class Distribution(ABC):

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class Gaussian(Distribution):

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = cov

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.logpdf(x))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        d = x - self.mean
        return -0.5 * np.log(2*np.pi) - 0.5 * np.log(det(self.cov)) - 0.5 * np.sum(d @ invert(self.cov) * d, axis=1)

    def __repr__(self) -> str:
        return f"Gaussian(mean={self.mean}, cov={self.cov})"


def compute_posterior_distribution(x: np.ndarray, y: np.ndarray, kernel: LRKernel, prior: Gaussian, sigma2: float) -> Gaussian:
    """ Compute the posterior distribution p(w | x, y) using the given feature kernel and prior distribution.

    x: (N, D) - The input data
    y: (N) - The output data
    kernel: LRKernel - The feature kernel to use
    prior: Gaussian - The prior distribution over the coefficients
    sigma2: float - The noise variance

    Returns: Gaussian - The posterior distribution over the coefficients
    """
    x = kernel(x) # (N, K)
    prior_cov_inv = invert(prior.cov) # (K, K)
    cov = invert(prior_cov_inv + 1/sigma2 * (x.T @ x)) # (K, K)
    mean = cov @ (prior_cov_inv @ prior.mean + 1/sigma2 * (x.T @ y)) # (K)
    return Gaussian(mean, cov)


def compute_predictive_dist(x: np.ndarray, y: np.ndarray, kernel: LRKernel, params: Gaussian, sigma2: float) -> list[Gaussian]:
    """ Compute the predictive distribution p(y* | x*, x, y) using the given feature kernel and prior distribution.

    x: (N, D) - The input data
    y: (N) - The output data
    kernel: LRKernel - The feature kernel to use
    params: Gaussian - The prior (or posterior) distribution over the coefficients
    sigma2: float - The noise variance

    Returns: list[Gaussian] - The predictive distribution for each input point
    """
    results = []
    x = kernel(x) # (N, K)
    for x_, y_ in zip(x, y):
        # x_: (K), y_: (1)
        mean = np.dot(x_, params.mean) # ()
        cov = (x_ @ params.cov @ x_).item() + sigma2 # ()
        results.append(Gaussian(mean, cov))
    return results


def plot_predictive_dist(x: np.ndarray, y: np.ndarray, kernel: LRKernel, prior: Gaussian, sigma2: float):
    """ Plot the data and the predictive distribution.

    x: (N, D) - The input data
    y: (N) - The output data
    kernel: LRKernel - The feature kernel to use
    prior: Gaussian - The prior distribution over the coefficients
    sigma2: float - The noise variance
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("You need to install the matplotlib package to plot the data.")
    predictive_dists = compute_predictive_dist(x, y, kernel, prior, sigma2)
    means = np.array([dist.mean for dist in predictive_dists])
    stds = np.array([np.sqrt(dist.cov) for dist in predictive_dists])
    plt.fill_between(x[:, 0], means - stds, means + stds, color='gray', alpha=0.3, label="±1 std confidence")
    plt.fill_between(x[:, 0], means - 2*stds, means + 2*stds, color='gray', alpha=0.1, label="±2 std confidence")
    plt.scatter(x[:, 0], y, label="Data points")
    plt.plot(x[:, 0], means, 'r', label=f"Mean Curve")
    print(means)

    # Plot title and labels
    plt.title(f"Bayesian Linear Regression\nNoise Variance: {sigma2}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # Show the plot
    plt.show()