"""Define distribution classes for fitting and comparing distribution functions."""
import numpy as np
from scipy.stats import norm, maxwell, gamma, poisson, rayleigh, erlang, lognorm, betaprime, burr, invweibull


# #############################################################################
# Super class
# #############################################################################
class Distributions:
    def __init__(self):
        # General settings
        self.dist_name = None  # For saving (paths)
        self.dist_label = None  # For output
        self.dist_label_flat = None
        # Parameter settings
        self.epsilon = 1e-10  # Small value to ensure parameter bounds larger than zero
        self.par_names_calc = []  # Parameters that are calculated after fitting (only for weights)
        # # Density
        self.par_names_pdf = None
        self.par_labels_pdf = None
        self.par_bounds_pdf = None
        self.par_n_pdf = None
        # # Kernel
        self.par_names_kernel = None
        self.par_labels_kernel = None
        self.par_bounds_kernel = None
        self.par_lims = None
        self.par_n_kernel = None

    def kernel(self, *args):
        raise NotImplementedError

    def pdf(self, *args):
        raise NotImplementedError

    def cdf(self, *args):
        raise NotImplementedError

    def logpdf(self, *args):
        pdf = self.pdf(*args)
        return np.log(pdf.xlip(min=self.epsilon))

    @staticmethod
    def guess_pdf(bin_centers, hist, std, fish_age):
        # raise NotImplementedError
        return None

    @staticmethod
    def guess_kernel(bin_centers, hist, std):
        raise NotImplementedError

    @staticmethod
    def guess_mode(bin_centers, hist):
        return bin_centers[np.argmax(hist)]

    @staticmethod
    def sort_popt_kernel(popts):
        return popts

    @staticmethod
    def sort_popt_pdf(popts):
        return popts

    def sort_popt(self, popts):
        """Ensures that output corresponds to self.par_names_pdf"""
        return self.sort_popt_pdf(popts)

    def get_bounds(self, fish_age, dist_type='pdf'):
        if dist_type == 'pdf':
            return self.par_bounds_pdf
        elif dist_type == 'kernel':
            return self.par_bounds_kernel

    def set_bounds(self, bounds_dict):
        pass

    def get_par_lims(self, par_lims_dict):
        pass


# #############################################################################
# Single distributions
# #############################################################################
class Normal(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "normal"
        self.dist_label = "Normal"
        self.dist_label_flat = "Normal"
        # Parameter settings
        # # Kernel
        self.par_names_kernel = ["mu", "sigma", "weight_Normal"]
        self.par_labels_kernel = [r"$\mu$", r"$\sigma$", r"$w$"]
        self.par_bounds = [(-np.inf, np.inf), (self.epsilon, np.inf), (0, +np.inf)]
        self.par_lims = [(-30, 30), (0, 60), (0, 0.01)]  # OrientationChange
        self.par_n_kernel = 3
        # # Density
        self.par_names_pdf = ["mu", "sigma"]
        self.par_labels_pdf = [r"$\mu$", r"$\sigma$"]
        self.par_bounds_pdf = [(-np.inf, np.inf), (self.epsilon, np.inf)]
        self.par_n_pdf = 2

    def kernel(self, x, mu, sigma, weight):
        return weight * norm.pdf(x, mu, sigma)

    def pdf(self, x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    def logpdf(self, x, mu, sigma):
        return norm.logpdf(x, mu, sigma)

    @staticmethod
    def guess_pdf(bin_centers, hist, std, *args):
        mu = bin_centers[np.argmax(hist)]
        return mu, std

    def guess_kernel(self, bin_centers, hist, std):
        mu, sigma = self.guess_pdf(bin_centers, hist, std)
        w = np.max(hist) * (std * np.sqrt(2 * np.pi))
        return mu, std, w

    def set_bounds(self, bounds_dict):
        mu = bounds_dict["mu"]
        sigma = bounds_dict["sigma"]
        weight = bounds_dict["weight"]
        self.par_bounds_pdf = [mu, sigma]
        self.par_bounds_kernel = [mu, sigma, weight]

    def get_par_lims(self, par_lims_dict):
        mu = par_lims_dict["mu"]
        sigma = par_lims_dict["sigma"]
        weight = par_lims_dict["weight"]
        return [mu, sigma, weight]


class Maxwell(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "maxwell"
        self.dist_label = "Maxwell"
        self.dist_label_flat = "Maxwell"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["mode"]
        self.par_labels_pdf = ["mode"]
        self.par_bounds_pdf = [(self.epsilon, np.inf)]
        self.par_n_pdf = 1
        # # Kernel
        self.par_names_kernel = ["mode", "weight_Maxwell"]
        self.par_labels_kernel = ["mode", "weight"]
        self.par_bounds_kernel = [(self.epsilon, np.inf), (0, +np.inf)]
        self.par_n_kernel = 2

    def kernel(self, x, mode, weight):
        scale = mode / np.sqrt(2)
        return weight * maxwell.pdf(x, scale=scale)

    def pdf(self, x, mode):
        scale = mode / np.sqrt(2)
        return maxwell.pdf(x, scale=scale)

    def logpdf(self, x, mode):
        scale = mode / np.sqrt(2)
        return maxwell.logpdf(x, scale=scale)

    @staticmethod
    def guess_pdf(bin_centers, hist, std, *args):
        return abs(bin_centers[np.argmax(hist)])  # return mode

    def guess_kernel(self, bin_centers, hist, std):
        mode = self.guess_pdf(bin_centers, hist, std)
        weight = np.max(hist) * (std * np.sqrt(2 * np.pi))
        return mode, weight

    def set_bounds(self, bounds_dict):
        mode = bounds_dict["mode"]
        weight = bounds_dict["weight"]
        self.par_bounds_pdf = [mode]
        self.par_bounds_kernel = [mode, weight]

    def get_par_lims(self, par_lims_dict):
        mode = par_lims_dict["mode"]
        weight = par_lims_dict["weight"]
        return [mode, weight]


class Poisson(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "poisson"
        self.dist_label = "Poisson"
        self.dist_label_flat = "Poisson"
        # Parameter settings
        # # Kernel
        self.par_names_kernel = ["rate", "weight_Poisson"]
        self.par_labels_kernel = [r"$\lambda$", "weight"]
        self.par_bounds_kernel = [(self.epsilon, np.inf), (0, +np.inf)]
        self.par_lims = [(0, 60), (0, 0.01)]
        self.par_n_kernel = 2
        # # Density
        self.par_names_pdf = ["rate"]
        self.par_labels_pdf = [r"$\lambda$"]
        self.par_bounds_pdf = [(self.epsilon, np.inf)]
        self.par_n_pdf = 1

    def kernel(self, x, rate, weight):
        return weight * poisson.pmf(x, rate)

    def pdf(self, x, rate):
        return poisson.pmf(x, rate)

    def logpdf(self, x, rate):
        return poisson.logpmf(x, rate)

    @staticmethod
    def guess_pdf(bin_centers, hist, std, *args):
        rate = np.average(bin_centers, weights=hist)
        return rate

    def guess_kernel(self, bin_centers, hist, std):
        rate = self.guess_pdf(bin_centers, hist, std)
        w = np.max(hist) * (std * np.sqrt(2 * np.pi))
        return rate, w


class Rayleigh(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "rayleigh"
        self.dist_label = "Rayleigh"
        self.dist_label_flat = "Rayleigh"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["mode"]
        self.par_labels_pdf = ["mode"]
        self.par_bounds_pdf = [(self.epsilon, np.inf)]

    def pdf(self, x, mode):
        return rayleigh.pdf(x, scale=mode)

    def guess_pdf(self, bin_centers, hist, std, *args):
        return self.guess_mode(bin_centers, hist)


# 2 Parameter distributions ###################################################
class Gamma(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "gamma"
        self.dist_label = "Gamma"
        self.dist_label_flat = "Gamma"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["k", "theta", ]
        self.par_bounds_pdf = [(1, np.inf), (self.epsilon, np.inf)]
        self.par_n_pdf = 2
        # # Kernel
        self.par_names_kernel = ["k", "theta", "weight_Gamma"]
        self.par_bounds_kernel = [(1, np.inf), (self.epsilon, np.inf), (0, +np.inf)]
        self.par_lims = [(0, 5), (0, 60), (0, 0.01)]
        self.par_n_kernel = 3

    def pdf(self, x, shape, scale):
        return gamma.pdf(x, a=shape, scale=scale)

    def kernel(self, x, shape, scale, weight):
        return weight * gamma.pdf(x, a=shape, scale=scale)

    def logpdf(self, x, shape, scale):
        return gamma.logpdf(x, a=shape, scale=scale)

    # def sort_popt_pdf(self, popt):
    #     """Ensures that output corresponds to self.par_names_calc + self.par_names_pdf"""
    #     shape, scale = popt
    #     mode = self.get_mode(shape, scale)
    #     var = self.get_var(shape, scale)
    #     return mode, var, shape, scale

    @staticmethod
    def get_mode(shape, scale):
        return (shape - 1) * scale  # Assuming shape > 1

    @staticmethod
    def get_var(shape, scale):
        return gamma.var(a=shape, scale=scale)
        # return shape * scale ** 2

    @staticmethod
    def get_shape_and_scale(mode, var):
        # Calculate the shape parameter k
        discriminant = mode ** 2 * (mode ** 2 + 4 * var)
        k_plus = ((mode**2 + 2*var) + np.sqrt(discriminant)) / (2 * var)
        k_min = ((mode**2 + 2*var) - np.sqrt(discriminant)) / (2 * var)
        k = np.max([k_plus, k_min, 0])  # Take maximum k to ensure k > 0
        # Calculate the scale parameter θ
        if k > 1:
            theta = mode / (k - 1)
        else:
            theta = 0
        return k, theta

    def guess_pdf(self, bin_centers, hist, std, *args):
        mode = bin_centers[np.argmax(hist)]
        mean = np.average(bin_centers, weights=hist)
        var = np.average((bin_centers - mean) ** 2, weights=hist)
        return self.get_shape_and_scale(mode, var)

    def guess_kernel(self, bin_centers, hist, std):
        shape, scale = self.guess_pdf(bin_centers, hist, std)
        w = np.max(hist) * (std * np.sqrt(2 * np.pi))
        return shape, scale, w


class MyGamma(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "my_gamma"
        self.dist_label = "Gamma"
        self.dist_label_flat = "Gamma"
        # Parameter settings
        self.par_names_pdf = ['mode', 'var']
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf)]
        self.par_n_pdf = 2

    def pdf(self, x, mode, var):
        shape, scale = self.get_shape_and_scale(mode, var)
        return gamma.pdf(x, shape, scale=scale)

    def logpdf(self, x, mode, var):
        shape, scale = self.get_shape_and_scale(mode, var)
        return gamma.logpdf(x, shape, scale=scale)

    @staticmethod
    def get_shape_and_scale(mode, var):
        # Calculate the shape parameter k
        discriminant = mode ** 2 * (mode ** 2 + 4 * var)
        k_plus = ((mode**2 + 2*var) + np.sqrt(discriminant)) / (2 * var)
        k_min = ((mode**2 + 2*var) - np.sqrt(discriminant)) / (2 * var)
        k = np.max([k_plus, k_min, 0])  # Take maximum k to ensure k > 0
        # Calculate the scale parameter θ
        if k > 1:
            theta = mode / (k - 1)
        else:
            theta = 0
        return k, theta

    def guess_pdf(self, bin_centers, hist, std, *args):
        mode = bin_centers[np.argmax(hist)]
        mean = np.average(bin_centers, weights=hist)
        var = np.average((bin_centers - mean) ** 2, weights=hist)
        return mode, var


class Erlang(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "erlang"
        self.dist_label = "Erlang"
        self.dist_label_flat = "Erlang"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["k", "beta"]
        self.par_labels_pdf = ["k", "beta"]
        self.par_bounds_pdf = [(1, np.inf), (self.epsilon, np.inf)]
        self.par_n_pdf = 2

    def pdf(self, x, k, scale):
        return erlang.pdf(x, k, scale=scale)

    def guess_pdf(self, bin_centers, hist, std, *args):
        return 5, 1


class Weibull(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "weibull"
        self.dist_label = "Weibull"
        self.dist_label_flat = "Weibull"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["l", "k"]
        self.par_labels_pdf = ["l", "k"]
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf)]
        self.par_n_pdf = 2

    def pdf(self, x, l, k):
        return (k / l) * (x / l)**(k - 1) * np.exp(-(x / l)**k)

    def guess_pdf(self, bin_centers, hist, std, *args):
        return 1, 5


class Lognormal(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "lognormal"
        self.dist_label = "Lognormal"
        self.dist_label_flat = "Lognormal"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["loc", "scale"]
        self.par_labels_pdf = ["loc", "scale"]
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf)]
        self.par_n_pdf = 2

    def pdf(self, x, loc, scale):
        return lognorm.pdf(x, loc, scale)

    def guess_pdf(self, bin_centers, hist, std, *args):
        return 0, 0.5


class BetaPrime(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "betaprime"
        self.dist_label = "Beta Prime"
        self.dist_label_flat = "Beta Prime"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["a", "b"]
        self.par_labels_pdf = ["a", "b"]
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf)]
        self.par_n_pdf = 2

    def pdf(self, x, a, b):
        return betaprime.pdf(x, a, b)

    def guess_pdf(self, bin_centers, hist, std, *args):
        return 2, 2


class Burr(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "burr"
        self.dist_label = "Burr"
        self.dist_label_flat = "Burr"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["c", "k"]
        self.par_labels_pdf = ["c", "k"]
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf)]
        self.par_n_pdf = 2

    def pdf(self, x, c, k):
        return burr.pdf(x, c, k)

    def guess_pdf(self, bin_centers, hist, std, *args):
        return 3, 1


class Frechet(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "frechet"
        self.dist_label = "Frechet"
        self.dist_label_flat = "Frechet"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["loc", "scale"]
        self.par_labels_pdf = ["loc", "scale"]
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf)]
        self.par_n_pdf = 2

    def pdf(self, x, loc, scale):
        return invweibull.pdf(x, loc, scale)

    def guess_pdf(self, bin_centers, hist, std, *args):
        return 3, 1


# #############################################################################
# Double distributions
# #############################################################################
class DoubleNormal(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "double_normal"
        self.dist_label = "Double Normal"
        self.dist_label_flat = "Double Normal"
        # Parameter settings
        # # Kernel
        self.par_names_kernel = ["mu1", "mu2", "sigma1", "sigma2", "weight1_DoubleNormal", "weight2_DoubleNormal"]
        self.par_labels_kernel = [r"$\mu_1$", r"$\mu_2$", r"$\sigma_1$", r"$\sigma_2$", r"$w_1$", r"$w_2$"]
        self.par_bounds_kernel = [(-np.inf, np.inf), (-np.inf, np.inf), (self.epsilon, np.inf), (self.epsilon, np.inf),
                                  (0, +np.inf), (0, +np.inf)]
        self.par_lims = [(0, 60), (-60, 0), (0, 60), (0, 60), (0, 0.005), (0, 0.005)]  # OrientationChange
        self.par_n_kernel = 6
        # # Density
        self.par_names_pdf = ["mu1", "mu2", "sigma1", "sigma2", "ratio1"]
        self.par_labels_pdf = [r"$\mu_1$", r"$\mu_2$", r"$\sigma_1$", r"$\sigma_2$", r"$r_1$"]
        self.par_bounds_pdf = [(-np.inf, np.inf), (-np.inf, np.inf), (self.epsilon, np.inf), (self.epsilon, np.inf),
                               (0, 1)]
        self.par_n_pdf = 5

    def kernel(self, x, mu1, mu2, sigma1, sigma2, weight1, weight2):
        return weight1 * norm.pdf(x, mu1, sigma1) + weight2 * norm.pdf(x, mu2, sigma2)

    def pdf(self, x, mu1, mu2, sigma1, sigma2, ratio1):
        return ratio1 * norm.pdf(x, mu1, sigma1) + (1 - ratio1) * norm.pdf(x, mu2, sigma2)

    @staticmethod
    def sort_popt_kernel(popt):
        # Juveniles: Sort parameters such that mu1 > mu2
        mu1, mu2, sigma1, sigma2, weight1, weight2 = popt
        if mu1 < mu2:
            popt = [mu2, mu1, sigma2, sigma1, weight2, weight1]

        # Larvae: Sort parameters such that (abs mu1) < (abs mu2)
        mu1, mu2, sigma1, sigma2, weight1, weight2 = popt
        if abs(mu1) > abs(mu2):
            popt = [mu2, mu1, sigma2, sigma1, weight2, weight1]
        return popt

    @staticmethod
    def guess_pdf(bin_centers, hist, std, *args):
        # Optimised for OrientationChange
        # mu1 = bin_centers[np.argmax(hist)]
        # mu2 = -1 * mu1
        mu1, mu2 = 45, -45  # For OrientationChange
        sigma1 = std / 2
        sigma2 = std / 2
        ratio1 = 0.5
        return mu1, mu2, sigma1, sigma2, ratio1

    def guess_kernel(self, bin_centers, hist, std):
        mu1, mu2, sigma1, sigma2, ratio1 = self.guess_pdf(bin_centers, hist, std)
        w = np.max(hist) * (std * np.sqrt(2 * np.pi))
        w1 = w / 2
        w2 = w / 2
        return mu1, mu2, sigma1, sigma2, w1, w2

    def set_bounds(self, bounds_dict):
        mu = bounds_dict["mu"]
        sigma = bounds_dict["sigma"]
        weight = bounds_dict["weight"]
        self.par_bounds_pdf = [mu, mu, sigma, sigma, (0, 1)]
        self.par_bounds_kernel = [mu, mu, sigma, sigma, weight, weight]

    def get_par_lims(self, par_lims_dict):
        mu = par_lims_dict["mu"]
        sigma = par_lims_dict["sigma"]
        weight = par_lims_dict["weight"]
        return [mu, mu, sigma, sigma, weight, weight]


class DoubleMaxwell(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "double_maxwell"
        self.dist_label = "Double Maxwell"
        self.dist_label_flat = "Double Maxwell"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["scale1", "scale2", "ratio1"]
        self.par_labels_pdf = ["scale1", "scale2", "ratio1"]
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf), (0, 1)]
        self.par_lims = [(0, 30), (0, 30), (0, 0.005), (0, 0.005)]  # OrientationChange
        self.par_n_pdf = 3
        # # Kernel
        self.par_names_kernel = ["a1", "a2", "weight1_DoubleMaxwell", "weight2_DoubleMaxwell"]
        self.par_labels_kernel = ["scale1", "scale2", "weight1", "weight2"]
        self.par_bounds_kernel = [(self.epsilon, np.inf), (self.epsilon, np.inf), (0, +np.inf), (0, +np.inf)]
        self.par_n_kernel = 4

    def kernel(self, x, scale1, scale2, weight1, weight2):
        return weight1 * maxwell.pdf(x, scale=scale1) + weight2 * maxwell.pdf(-1 * x, scale=scale2)

    def pdf(self, x, scale1, scale2, ratio1):
        return ratio1 * maxwell.pdf(x, scale=scale1) + (1 - ratio1) * maxwell.pdf(-1 * x, scale=scale2)

    @staticmethod
    def guess_pdf(bin_centers, hist, std, *args):
        # mode = abs(bin_centers[np.argmax(hist)])
        # scale1 = mode / np.sqrt(2)
        # scale2 = mode / np.sqrt(2)
        # Optimised for OrientationChange
        scale1, scale2 = 25, 25
        ratio1 = 0.5
        return scale1, scale2, ratio1

    def guess_kernel(self, bin_centers, hist, std):
        scale1, scale2, ratio1 = self.guess_pdf(bin_centers, hist, std)
        w = np.max(hist) * (std * np.sqrt(2 * np.pi))
        w1 = w / 2
        w2 = w / 2
        return scale1, scale2, w1, w2

    def set_bounds(self, bounds_dict):
        scale = bounds_dict["scale"]
        weight = bounds_dict["weight"]
        self.par_bounds_pdf = [scale, scale, (0, 1)]
        self.par_bounds_kernel = [scale, scale, weight, weight]

    def get_par_lims(self, par_lims_dict):
        scale = par_lims_dict["scale"]
        weight = par_lims_dict["weight"]
        return [scale, scale, weight, weight]


class DoubleGamma(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "double_gamma"
        self.dist_label = "Double Gamma"
        self.dist_label_flat = "Double Gamma"
        # Parameter settings
        # # Kernel
        self.par_names_kernel = ["k1", "k2", "theta1", "theta2", "weight1_DoubleGamma", "weight2_DoubleGamma"]
        self.par_labels_kernel = ["shape1", "shape2", "scale1", "scale2", "weight1", "weight2"]
        self.par_bounds_kernel = [(self.epsilon, np.inf), (self.epsilon, np.inf), (self.epsilon, np.inf),
                                  (self.epsilon, np.inf), (0, +np.inf), (0, +np.inf)]
        self.par_lims = [(0, 5), (0, 5), (0, 60), (0, 60), (0, 0.02), (0, 0.02)]  # OrientationChange
        self.par_n_kernel = 6
        # # Density
        self.par_names_pdf = ["shape1", "shape2", "scale1", "scale2", "ratio1"]
        self.par_labels_pdf = ["shape1", "shape2", "scale1", "scale2", "ratio1"]
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf), (self.epsilon, np.inf),
                               (self.epsilon, np.inf), (0, 1)]
        self.par_n_pdf = 5

    def kernel(self, x, shape1, shape2, scale1, scale2, weight1, weight2):
        return weight1 * gamma.pdf(x, shape1, scale=scale1) + weight2 * gamma.pdf(-1*x, shape2, scale=scale2)

    def pdf(self, x, shape1, shape2, scale1, scale2, ratio1):
        return ratio1 * gamma.pdf(x, shape1, scale=scale1) + (1 - ratio1) * gamma.pdf(-1*x, shape2, scale=scale2)

    @staticmethod
    def guess_pdf(bin_centers, hist, std, *args):
        """Shape and scale must be positive."""
        mode = bin_centers[np.argmax(hist)]
        scale = abs(std / mode)
        shape = abs(mode / scale)
        ratio1 = 0.5
        return shape, shape, scale, scale, ratio1

    def guess_kernel(self, bin_centers, hist, std):
        shape1, shape2, scale1, scale2, ratio1 = self.guess_pdf(bin_centers, hist, std)
        w = np.max(hist) * (std * np.sqrt(2 * np.pi))
        w1 = w / 2
        w2 = w / 2
        return shape1, shape2, scale1, scale2, w1, w2

    def set_bounds(self, bounds_dict):
        shape = bounds_dict["shape"]
        scale = bounds_dict["scale"]
        weight = bounds_dict["weight"]
        self.par_bounds_pdf = [shape, shape, scale, scale, (0, 1)]
        self.par_bounds_kernel = [shape, shape, scale, scale, weight, weight]

    def get_par_lims(self, par_lims_dict):
        shape = par_lims_dict["shape"]
        scale = par_lims_dict["scale"]
        weight = par_lims_dict["weight"]
        return [shape, shape, scale, scale, weight, weight]


# #############################################################################
# Triple distributions
# #############################################################################
class TripleNormal(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "triple_normal"
        self.dist_label = "Triple Normal"
        self.dist_label_flat = "Triple Normal"
        # Parameter settings
        # # Kernel
        self.par_names_kernel = ["mu1", "mu2", "sigma1", "sigma2", "sigma3", "weight1", "weight2", "weight3"]
        self.par_labels_kernel = [r"$\mu_1$", r"$\mu_2$", r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$",
                                  r"$w_1$", r"$w_2$", r"$w_3$"]
        self.par_bounds_kernel = [(-np.inf, np.inf), (-np.inf, np.inf), (self.epsilon, np.inf),
                                  (self.epsilon, np.inf), (self.epsilon, np.inf), (0, +np.inf), (0, +np.inf), (0, +np.inf)]
        self.par_lims = [(0, 60), (-60, 0), (0, 60), (0, 60), (0, 60), (0, 30), (0, 30), (0, 30)]  # OrientationChange
        self.par_n_kernel = 8
        # # Density
        self.par_names_pdf = ["mu1", "mu2", "sigma1", "sigma2", "sigma3", "ratio1", "ratio3"]
        self.par_labels_pdf = [r"$\mu_1$",  r"$\mu_2$", r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$",
                               r"$r_1$", r"$r_2$"]
        self.par_bounds_pdf = [(-np.inf, np.inf), (-np.inf, np.inf), (self.epsilon, np.inf),
                               (self.epsilon, np.inf), (self.epsilon, np.inf), (0, 1), (0, 1)]
        self.par_n_pdf = 7

    def kernel(self, x, mu1, mu2, sigma1, sigma2, sigma3, weight1, weight2, weight3):
        return weight1 * norm.pdf(x, mu1, sigma1) + weight2 * norm.pdf(x, mu2, sigma2) + weight3 * norm.pdf(x, 0, sigma3)

    def pdf(self, x, mu1, mu2, sigma1, sigma2, sigma3, ratio1, ratio3):
        """ratio2 = (1 - ratio1 - ratio3) can be negative: between -2 and 1"""
        return ratio1 * norm.pdf(x, mu1, sigma1) + (1 - ratio1 - ratio3) * norm.pdf(x, 0, sigma3) + ratio3 * norm.pdf(x, mu2, sigma2)

    @staticmethod
    def sort_popt_kernel(popts):
        # Sort parameters such that mu1 > mu3
        mu1, mu3, sigma1, sigma2, sigma3, weight1, weight2, weight3 = popts
        if mu1 < mu3:
            popts = [mu3, mu1, sigma3, sigma2, sigma1, weight3, weight2, weight1]
        return popts

    @staticmethod
    def guess_pdf(bin_centers, hist, std, *args):
        # mu1 = bin_centers[np.argmax(hist)]
        # mu3 = -1 * mu1
        # Optimised for OrientationChange
        mu1, mu3 = 45, -45
        sigma1 = std / 2
        sigma2 = std / 10
        sigma3 = std / 2
        ratio1 = 1/3
        ratio2 = 1/3
        return mu1, mu3, sigma1, sigma2, sigma3, ratio1, ratio2

    def guess_kernel(self, bin_centers, hist, std):
        mu1, mu3, sigma1, sigma2, sigma3, ratio1, ratio2 = self.guess_pdf(bin_centers, hist, std)
        w = np.max(hist) * (std * np.sqrt(2 * np.pi))
        w1 = w / 3
        w2 = w / 3
        w3 = w / 3
        return mu1, mu3, sigma1, sigma2, sigma3, w1, w2, w3

    def set_bounds(self, bounds_dict):
        mu = bounds_dict["mu"]
        sigma = bounds_dict["sigma"]
        sigma_center = bounds_dict["sigma_center"]
        weight = bounds_dict["weight"]
        self.par_bounds_pdf = [mu, mu, sigma, sigma, sigma_center, (0, 1), (0, 1)]
        self.par_bounds_kernel = [mu, mu, sigma, sigma, sigma_center, weight, weight, weight]

    def get_par_lims(self, par_lims_dict):
        mu = par_lims_dict["mu"]
        sigma = par_lims_dict["sigma"]
        sigma_center = par_lims_dict["sigma"]
        weight = par_lims_dict["weight"]
        return [mu, mu, sigma, sigma, sigma_center, weight, weight, weight]


class DoubleMaxwellCenterNormal(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "double_maxwell_center_normal"
        self.dist_label = "Double Maxwell\nCenter Normal"
        self.dist_label_flat = "Double Maxwell Center Normal"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["mode1", "mode2", "ratio1", "ratio2", "sigma3"]
        self.par_names_calc = ["ratio3"]  # These parameters are calculated after fitting
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf),
                               (0, +np.inf), (0, +np.inf),
                               (self.epsilon, np.inf), ]
        self.par_n_pdf = 5
        # # Kernel
        self.par_names_kernel = ["mode1", "mode2",
                                 "weight1_DoubleMaxwellCenterNormal", "weight2_DoubleMaxwellCenterNormal", "weight3_DoubleMaxwellCenterNormal",
                                 "sigma3"]
        self.par_bounds_kernel = [(self.epsilon, np.inf), (self.epsilon, np.inf),
                                  (0, +np.inf), (0, +np.inf), (0, +np.inf),
                                  (self.epsilon, np.inf), ]
        self.par_n_kernel = 6

    def pdf(self, x, mode1, mode2, ratio1, ratio2, sigma3):
        scale1 = mode1 / np.sqrt(2)
        scale2 = mode2 / np.sqrt(2)
        return ratio1 * maxwell.pdf(x, scale=scale1) + ratio2 * maxwell.pdf(-1 * x, scale=scale2) + (1 - ratio1 - ratio2) * norm.pdf(x, 0, sigma3)

    def kernel(self, x, mode1, mode2, weight1, weight2, weight3, sigma3):
        scale1 = mode1 / np.sqrt(2)
        scale2 = mode2 / np.sqrt(2)
        return weight1 * maxwell.pdf(x, scale=scale1) + weight2 * maxwell.pdf(-1 * x, scale=scale2) + weight3 * norm.pdf(x, 0, sigma3)

    def sort_popt_pdf(self, popt):
        """Ensures that output corresponds to self.par_names_pdf + self.par_names_calc"""
        mode1, mode2, ratio1, ratio2, sigma3 = popt
        ratio3 = 1 - ratio1 - ratio2
        return mode1, mode2, ratio1, ratio2, sigma3, ratio3

    @staticmethod
    def guess_pdf(bin_centers, hist, std, fish_age):
        return 30, 30, 1/3, 1/3, 5

    def guess_kernel(self, bin_centers, hist, std):
        mode1, mode2, ratio1, ratio2, sigma3 = self.guess_pdf(bin_centers, hist, std)
        w = np.max(hist) * (std * np.sqrt(2 * np.pi))
        w1 = w / 3
        w2 = w / 3
        w3 = w / 3
        return mode1, mode2, w1, w2, w3, sigma3


class DoubleMaxwellCenterNormalConstrained(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "double_maxwell_center_normal_constrained"
        self.dist_label = "Double Maxwell\nCenter Normal\nConstrained"
        self.dist_label_flat = "Double Maxwell Center Normal Constrained"
        # Parameter settings
        # # Density
        self.mode = 25  # deg
        self.sigma = 5  # deg
        self.par_names_pdf = ["ratio1", "ratio2"]
        self.par_names_calc = ["ratio3"]  # These parameters are calculated after fitting
        self.par_bounds_pdf = [(0, +np.inf), (0, +np.inf),]
        self.par_n_pdf = 2

    def pdf(self, x, ratio1, ratio2):
        scale1 = self.mode / np.sqrt(2)
        scale2 = self.mode / np.sqrt(2)
        sigma3 = self.sigma
        return ratio1 * maxwell.pdf(x, scale=scale1) + ratio2 * maxwell.pdf(-1 * x, scale=scale2) + (1 - ratio1 - ratio2) * norm.pdf(x, 0, sigma3)

    def sort_popt(self, popts):
        ratio1, ratio2 = popts
        ratio3 = 1 - ratio1 - ratio2
        return ratio1, ratio2, ratio3

    @staticmethod
    def guess_pdf(bin_centers, hist, std, fish_age):
        return 1/3, 1/3


class DoubleGammaCenterNormal(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "double_gamma_center_normal"
        self.dist_label = "Double Gamma\nCenter Normal"
        self.dist_label_flat = "Double Gamma Center Normal"
        # Parameter settings
        # # Kernel
        self.par_names_kernel = ["k1", "k2", "theta1", "theta2", "sigma3", "weight1_DoubleGammaCenterNormal", "weight2_DoubleGammaCenterNormal", "weight3_DoubleGammaCenterNormal"]
        self.par_labels_kernel = ["shape1", "shape2", "scale1", "scale2", r"$\sigma_3$", "weight1", "weight2", "weight3"]
        self.par_bounds_kernel = [(self.epsilon, np.inf), (self.epsilon, np.inf), (self.epsilon, np.inf),
                                  (self.epsilon, np.inf), (self.epsilon, np.inf),
                                  (0, +np.inf), (0, +np.inf), (0, +np.inf)]
        self.par_lims = [(0, 200), (0, 200), (0, 60), (0, 60), (0, 100), (0, 1), (0, 1), (0, 0.005)]  # OrientationChange
        self.par_n_kernel = 8
        # # Density
        self.par_names_pdf = ["shape1", "shape2", "scale1", "scale2", "sigma3", "ratio1", "ratio2"]
        self.par_labels_pdf = ["shape1", "shape2", "scale1", "scale2", r"$\sigma_3$", "ratio1", "ratio2"]
        self.par_bounds_pdf = [(self.epsilon, np.inf), (self.epsilon, np.inf), (self.epsilon, np.inf),
                               (self.epsilon, np.inf), (self.epsilon, np.inf), (0, 1), (0, 1)]
        self.par_n_pdf = 7

    def kernel(self, x, shape1, shape2, scale1, scale2, sigma3, weight1, weight2, weight3):
        return weight1 * gamma.pdf(x, shape1, scale=scale1) + weight2 * gamma.pdf(-1*x, shape2, scale=scale2) + weight3 * norm.pdf(x, 0, sigma3)

    def pdf(self, x, shape1, shape2, scale1, scale2, sigma3, ratio1, ratio2):
        return self.kernel(x, shape1, shape2, scale1, scale2, sigma3, ratio1, ratio2, 1 - ratio1 - ratio2)

    @staticmethod
    def guess_pdf(bin_centers, hist, std, *args):
        # Optimised for OrientationChange
        mode = bin_centers[np.argmax(hist)]
        scale = abs(std / mode)
        shape = abs(mode / scale)
        sigma3 = std / 10
        ratio1 = 1/3
        ratio2 = 1/3
        return shape, shape, scale, scale, sigma3, ratio1, ratio2

    def guess_kernel(self, bin_centers, hist, std):
        shape1, shape2, scale1, scale2, sigma3, ratio1, ratio2 = self.guess_pdf(bin_centers, hist, std)
        w = np.max(hist) * (std * np.sqrt(2 * np.pi))
        w1 = w / 3
        w2 = w / 3
        w3 = w / 3
        return shape1, shape2, scale1, scale2, sigma3, w1, w2, w3

    def set_bounds(self, bounds_dict):
        shape = bounds_dict["shape"]
        sigma = bounds_dict["sigma"]
        scale = bounds_dict["scale"]
        weight = bounds_dict["weight"]
        self.par_bounds_pdf = [shape, shape, scale, scale, sigma, (0, 1), (0, 1)]
        self.par_bounds_kernel = [shape, shape, scale, scale, sigma, weight, weight, weight]

    def get_par_lims(self, par_lims_dict):
        shape = par_lims_dict["shape"]
        sigma = par_lims_dict["sigma"]
        scale = par_lims_dict["scale"]
        weight = par_lims_dict["weight"]
        return [shape, shape, scale, scale, sigma, weight, weight, weight]


# #############################################################################
# Not really a distribution but we want to keep the same structure
# #############################################################################
class PercentageRightDist(Distributions):
    def __init__(self):
        super().__init__()
        self.dist_name = "percentage_right"
        self.dist_label = "Percentage Right"
        self.dist_label_flat = "Percentage Right"
        # Parameter settings
        # # Density
        self.par_names_pdf = ["p_right"]
        self.par_labels_pdf = [r"$P_{Right}$"]
        self.par_bounds_pdf = [(0, 1)]
        self.par_n_pdf = 1

