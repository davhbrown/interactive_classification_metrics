import numpy as np
from bokeh.models import ColumnDataSource
from scipy.stats.kde import gaussian_kde
from scipy.stats import skewnorm


class NormalDistData:
    """Handles the normal dists for each set of class "predictions" for the example.

    Instances get fed into Metrics to calculate binary classification performance metrics.
    """

    # Values must be (from) ColumnDataSource for interactivity to work
    def __init__(self, n: int, mean: float, sd: float, skew: float):
        self._n = n
        self._mean = mean
        self._sd = sd
        self._skew = skew
        raw_data, kde_curve = self._create_distribution()
        self.raw_data = ColumnDataSource(data=raw_data)
        self.kde_curve = ColumnDataSource(data=kde_curve)

    def _norm_dist(self) -> np.ndarray:
        """Generate normally distributed data with specified mean and standard deviation."""
        #return np.random.normal(self._mean, self._sd, self._n)
        return skewnorm.rvs(self._skew, self._mean, self._sd, size=self._n)

    def _create_distribution(self):
        """Return normally distributed data and a KDE curve from distribution parameters.

        Returns
        -------
        distribution_data : dict[numpy.ndarray] of normally distributed data according to params

        kde_source : dict of (x, y) coordinates of the KDE curve, keys ['x', 'y']

        """
        distribution_data = self._norm_dist()
        # Generate binned data
        BINS = 20
        _, bin_edges = np.histogram(distribution_data, bins=BINS, density=True)
        # Set KDE bounds
        kde_min = np.floor(bin_edges.min())
        kde_max = np.ceil(bin_edges.max())
        # x datapoints for plotting
        N_POINTS = 200
        x = np.linspace(kde_min, kde_max, N_POINTS)
        # Make KDE obj from norm dist data
        kde_obj_y = gaussian_kde(distribution_data)
        # Get y values of KDE & convert to raw counts, not density
        y = kde_obj_y(x) * self._n
        # Return (x, y) coords as dict
        kde_curve = dict(x=x, y=y)
        distribution_data = dict(data=distribution_data)
        return distribution_data, kde_curve

    # Callback handling
    def n_handler(self, attr, old, new):
        """Changes data & kde_curve in response to N Slider."""
        self._n = new
        self.raw_data.data, self.kde_curve.data = self._create_distribution()

    def mean_handler(self, attr, old, new):
        """Changes data & kde_curve in response to mean Slider."""
        self._mean = new
        self.raw_data.data, self.kde_curve.data = self._create_distribution()

    def sd_handler(self, attr, old, new):
        """Changes data & kde_curve in response to sd Slider."""
        self._sd = new
        self.raw_data.data, self.kde_curve.data = self._create_distribution()

    def skew_handler(self, attr, old, new):
        """Changes data & kde_curve in response to skew Slider."""
        self._skew = new
        self.raw_data.data, self.kde_curve.data = self._create_distribution()
