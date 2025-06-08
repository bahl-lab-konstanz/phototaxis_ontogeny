"""Defines property classes"""
import numpy as np

from .dist_settings import *


# #############################################################################
# Super class
# #############################################################################
class PropClass(Distributions):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = None  # str
        self.label = None  # str
        self.unit = None  # str
        self.prop_min, self.prop_max = None, None
        self.prop_resolution = None

        # Distribution properties
        self.bins = None
        self.bin_centers = None
        self.bin_centers_plot = None
        # Fitted parameters (median, distribution parameters)
        self.par_names = []
        self.par_labels = []
        self.par_bounds = []

        # Plot settings
        self.prop_lim = None  # list of floats
        self.prop_ticks = None
        self.prop_ticklabels = None
        self.prop_axlines = None
        # # Fitted parameters (median, distribution parameters)
        self.par_lims = []
        self.par_ticks = None
        self.par_ticklabels = None
        self.par_axlines = [None]
        self.par_lim_dict = {}
        self.par_tick_dict = {}

        # Slopes
        self.par_slope_lims = None

    # Helper functions ########################################################
    def set_bins(self, include_zero=False):
        if include_zero:
            prop_min = self.prop_min - self.prop_resolution / 2
            prop_max = self.prop_max + self.prop_resolution / 2
        else:
            prop_min = self.prop_min
            prop_max = self.prop_max

        self.bins = np.arange(prop_min, prop_max + self.prop_resolution, self.prop_resolution)
        self.bin_centers = (self.bins[1:] + self.bins[:-1]) / 2
        self.bins_plot = np.linspace(self.prop_min, self.prop_max, 1000)  # for smooth plot
        self.bin_centers_plot = (self.bins_plot[1:] + self.bins_plot[:-1]) / 2

    def get_prop_data(self, df):
        # Replace inf values with NaN
        # Drop rows containing NaN values
        return df[self.prop_name].replace([np.inf, -np.inf], np.nan).dropna()


# #############################################################################
# Property Super classes
# #############################################################################
class Distance(PropClass, MyGamma):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'total_distance'
        self.label = 'Displacement'  # str
        self.unit = 'mm'
        self.prop_min, self.prop_max = 0, 4  # cm (1 for larvae but 4 for juveniles)
        self.prop_resolution = 2 * 0.05  # cm
        # Distribution properties
        self.set_bins()

        # Spatial-temporal model
        # self.model = InterswimModel()

        # Fitted parameters (MyGamma)
        self.par_names = ['median', 'var']
        self.par_labels = ['median (mm)', r'variance (mm$^2$)']
        self.par_bounds_pdf = [(self.epsilon, 2), (self.epsilon, 0.4)]  # mode, var
        self.par_names_pdf = ['median', 'var']

        # Plot settings
        self.prop_lim = [0, 2]              # cm
        self.prop_ticks = [0, 1, 2]         # cm
        self.prop_ticklabels = [0, 10, 20]  # mm
        # Median, fitted parameters (MyGamma)
        self.par_lims = [[0, 1.5], [0, 0.3]]                 # cm
        self.par_ticks = [[0, 0.5, 1, 1.5], [0, 0.15, 0.3]]  # cm
        self.par_ticklabels = [[0, 5, 10, 15], [0, 15, 30]]   # mm
        self.par_lim_dict = {
            'p1': [-0.4, 0.4], 'p0': [0, 2],
            'a_pos': [-0.002, 0.002], 'a_neg': [-0.002, 0.002], 'b': [0, 2],
            'a_log': [-2, 2], 'b_log': [-0.2, 0.2],  # cm
        }
        self.par_tick_dict = {
            'a_log': np.linspace(-2, 2, 5), 'b_log': np.linspace(-0.2, 0.2, 5),
            'a_log_str': [-20, -10, 0, 10, 20], 'b_log_str': [-2, -1, 0, 1, 2],  # Converted to cm
        }
        # RMSE and MSE
        self.rmse_label = f'RMSE {self.label}\n({self.unit})'
        self.mse_label = f'MSE {self.label}\n({self.unit}2)'
        self.rmse_ticks_dict = {'larva': [0, 0.05, 0.1],  'juvie': [0, 0.05, 0.1]}  # cm
        self.mse_ticks_dict = {'larva': [0, 0.005, 0.01], 'juvie': [0, 0.005, 0.01]}    # cm2
        self.rmse_ticklabels_dict = {'larva': [0, 0.5, 1],  'juvie': [0, 0.5, 1]}   # mm
        self.mse_ticklabels_dict = {'larva': [0, 0.5, 1], 'juvie': [0, 0.5, 1]}     # mm2


class DistancePerBodyLength(PropClass, MyGamma):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'total_distance_per_body_length'
        self.label = 'Displacement / body length'  # str
        self.unit = '-'
        self.prop_min, self.prop_max = 0, 0.2
        self.prop_resolution = 0.01
        # Distribution properties
        self.set_bins()

        # Spatial-temporal model
        # self.model = SpaceModel()

        # Fitted parameters (MyGamma)
        self.par_names = ['mode', 'var']
        self.par_labels = ['mode (mm)', r'variance (mm$^2$)']
        self.par_bounds = [(self.epsilon, 0.2), (self.epsilon, 0.4)]  # mode, var

        # Plot settings
        self.prop_lim = [0, 1.5]  # [-]
        self.prop_ticks = [0, 0.5, 1, 1.5]  # [-]
        self.prop_ticklabels = self.prop_ticks  # [-]
        # Median, fitted parameters (MyGamma)
        self.par_lims = [[0, 1.5], [0, 0.3]]
        self.par_ticks = [[0, 0.5, 1, 1.5], [0, 0.15, 0.3]]
        self.par_ticklabels = self.par_ticks
        self.par_lim_dict = {
            'p1': [-0.2, 0.2],  'p0': [0, 1.5],
            'a_pos': [-0.02, 0.02], 'a_neg': [-0.02, 0.02], 'b': [0, 1.5],
            'a_log': [-4, 4], 'b_log': [-0.3, 0.3],
        }


class PercentageTurns(PropClass):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'percentage_turns'
        self.dist_name = 'none'  # One value per individual
        self.label = 'Turns'  # str
        self.unit = '%'
        self.prop_min, self.prop_max = 0, 1

        # Distribution properties
        self.par_names = ['percentage_turns']
        self.par_labels = ['percentage turns']

        # Plot settings
        self.prop_lim = [0, 100]
        self.prop_ticks = [0, 25, 50, 75, 100]
        self.prop_ticklabels = [0, 25, 50, 75, 100]
        self.prop_axlines = [50]
        self.par_lims = [[0, 100]]
        self.par_ticks = [[0, 25, 50, 75, 100]]
        self.par_ticklabels = [[0, 25, 50, 75, 100]]
        self.par_axlines = [[50]]

        self.par_lim_dict = {
            'p1': [-20, 20],  'p0': [0, 100],
            'a_pos': [-0.2, 0.2], 'a_neg': [-0.2, 0.2], 'b': [0, 100],
            'a_log': [-150, 150], 'b_log': [-20, 20],
        }

        # RMSE and MSE
        self.rmse_label = f'RMSE\n{self.label}\n({self.unit})'
        self.mse_label = f'MSE\n{self.label}\n({self.unit}2)'
        self.rmse_ticks_dict = {'larva': [3, 5, 7],  'juvie': [0, 2, 4]}     # %
        self.mse_ticks_dict = {'larva': [10, 25, 40], 'juvie': [0, 10, 20]}  # %2
        self.rmse_ticklabels_dict = self.rmse_ticks_dict    # %
        self.mse_ticklabels_dict = self.mse_ticks_dict      # %2


class OrientationChange(PropClass, DoubleMaxwellCenterNormal):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'estimated_orientation_change'
        self.label = 'Orientation change'  # str
        self.unit = 'deg'
        self.prop_min, self.prop_max = -100, 100  # deg
        self.prop_resolution = 2 * 5  # deg

        # Distribution properties
        self.set_bins(include_zero=True)
        self.par_names = [
            "ratio1", "ratio2", "ratio3",
            "mode1", "mode2",
            "sigma3",
        ]
        self.par_labels = [
            'left weight', 'right weight', 'center weight',
            'left mode (deg)', 'right mode (deg)',
            'forward shape (deg)',
        ]

        # Plot settings
        self.prop_lim = [-100, 100]
        self.prop_ticks = [-100, 0, 100]
        self.prop_ticklabels = [-100, 0, 100]
        self.prop_axlines = [0]
        self.par_lims = [
            [0, 1], [0, 1], [0, 1],
            [0, 50], [0, 50],
            [0, 50],
        ]
        self.par_ticks = [
            [0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1],
            [0, 25, 50], [0, 25, 50],
            [0, 25, 50],
        ]
        self.par_ticklabels = self.par_ticks

    # Overwrite guess function from distribution
    @staticmethod
    def guess_pdf(bin_centers, hist, std, fish_age):
        # Optimised for OrientationChange
        if fish_age <= 5:
            # Larva
            mode = 25
            ratio = 1 / 5
            sigma = std / 10
        else:
            # Juvenile
            mode = 40
            ratio = 1 / 5
            sigma = 45

        return mode, mode, ratio, ratio, sigma


class TurnAngle(PropClass):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'turn_angle'
        self.dist_name = 'none'  # One value per individual
        self.label = 'Abs. angle of turns'  # str
        self.unit = 'deg'
        self.prop_min, self.prop_max = 0, 40  # deg
        self.prop_resolution = 2 * 5  # deg

        # Spatial-temporal model
        # self.model = TurnAngleModel()

        # Fitted parameters
        self.par_names = ['mode', 'var']
        self.par_labels = ['mode (deg)', r'variance (deg$^2$)']

        # Plot settings
        self.prop_lim = [0, 100]
        self.prop_ticks = [0, 50, 100]
        self.prop_ticklabels = [0, 50, 100]
        # # Fitted parameters (median, distribution parameters)
        self.par_lims = [[15, 65], [0, 40]]
        self.par_ticks = [[15, 25, 35, 45, 55, 65], [0, 20, 40]]
        self.par_ticklabels = self.par_ticks

        self.par_lim_dict = {
            'p1': [-30, 30], 'p0': [0, 100],
            'a_pos': [-0.4, 0.4], 'a_neg': [-0.4, 0.4], 'b': [0, 100],
            'a_log': [-100, 100],  'b_log': [-10, 10],
        }

        # RMSE and MSE
        self.rmse_label = f'RMSE\n{self.label}\n({self.unit})'
        self.mse_label = f'MSE\n{self.label}\n({self.unit}2)'
        self.rmse_ticks_dict = {'larva': [0, 5, 10],  'juvie': [0, 5, 10]}     # deg
        self.mse_ticks_dict = {'larva': [0, 30, 60], 'juvie': [0, 10, 20]}  # deg2
        self.rmse_ticklabels_dict = self.rmse_ticks_dict    # deg
        self.mse_ticklabels_dict = self.mse_ticks_dict      # deg2


class PercentageLeft(PropClass, PercentageRightDist):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'percentage_left'
        self.label = 'Left turns'  # st
        self.unit = '% of turns'
        self.threshold = 10  # threshold to detect turns
        self.prop_min, self.prop_max = -100, 100  # deg

        # Spatial-temporal model
        # self.model = PercentageLeftModel()

        # Distribution properties
        # # Define bins for left, straight and right turns
        self.bins = np.asarray([self.prop_min, -self.threshold, self.threshold, self.prop_max])
        self.bin_centers = (self.bins[1:] + self.bins[:-1]) / 2
        # Fitted parameters (median, distribution parameters)
        self.par_names = ['p_left']
        self.par_labels = [r'P$_{left}$']

        # Plot settings
        self.prop_lim = [0, 100]
        self.prop_ticks = [0, 50, 100]
        self.prop_ticklabels = [0, 50, 100]
        self.prop_axlines = [50]
        # # Fitted parameters (median, distribution parameters)
        self.par_lims = [[20, 85]]
        self.par_ticks = [[25, 50, 75]]
        self.par_ticklabels = self.par_ticks
        self.par_axlines = [50]

        self.par_lim_dict = {
            'p1': [-40, 40], 'p0': [0, 100],
            'a_pos': [-0.2, 0.2], 'a_neg': [-0.2, 0.2],  'b': [0, 100],
            'a_log': [-150, 150], 'b_log': [-20, 20],
        }

        # RMSE and MSE
        self.rmse_label = f'RMSE\n{self.label}\n({self.unit})'
        self.mse_label = f'MSE\n{self.label}\n({self.unit}2)'
        self.rmse_ticks_dict = {'larva': [2, 5, 8],  'juvie': [0, 3, 6]}     # %
        self.mse_ticks_dict = {'larva': [10, 35, 60], 'juvie': [0, 10, 20]}  # %2
        self.rmse_ticklabels_dict = self.rmse_ticks_dict    # %
        self.mse_ticklabels_dict = self.mse_ticks_dict      # %2


class PercentageLeftChen2021(PercentageLeft):
    def __init__(self):
        super().__init__()
        self.prop_name = 'percentage_left'
        # self.model = Chen2021Model()


class PercentageLeftKarpenko2020(PercentageLeft):
    def __init__(self):
        super().__init__()
        self.prop_name = 'percentage_left'
        # self.model = Karpenko2020Model()


class TotalDuration(PropClass, MyGamma):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'total_duration'
        self.label = 'Inter-swim interval'  # str
        self.unit = 's'
        self.prop_min, self.prop_max = 0, 2  # s
        self.prop_resolution = 2 * 0.1  # s
        # Distribution properties
        self.set_bins()

        # Spatial-temporal model
        # self.model = InterswimModel()

        # # Fitted parameters (median, distribution parameters)
        self.par_names = ['mode', 'var']
        self.par_labels = ['mode (s)', r'variance (s$^2$)']
        self.par_bounds = [(self.epsilon, 2), (self.epsilon, 0.4)]

        # Plot settings
        self.prop_lim = [0, 2]
        self.prop_ticks = [0, 1, 2]
        self.prop_ticklabels = [0, 1, 2]
        # # Fitted parameters (median, distribution parameters)
        self.par_lims = [[0.5, 1.5], [0, 0.3]]
        self.par_ticks = [[0.5, 0.75, 1, 1.25, 1.5], [0, 0.15, 0.3]]    # seconds
        self.par_ticklabels = self.par_ticks  # s
        self.par_lim_dict = {
            'p1': [-0.4, 0.4], 'p0': [0, 2],
            'a_pos': [-0.004, 0.004], 'a_neg': [-0.004, 0.004], 'b': [0, 2],
            'a_log': [-2, 2], 'b_log': [-0.4, 0.4],
        }

        # RMSE and MSE
        self.rmse_label = f'RMSE\n{self.label}\n({self.unit})'
        self.mse_label = f'MSE\n{self.label}\n({self.unit}2)'
        self.rmse_ticks_dict = {'larva': [0, 0.1, 0.2],  'juvie': [0, 0.1, 0.2]}     # s
        self.mse_ticks_dict = {'larva': [0, 0.1, 0.2], 'juvie': [0, 0.1, 0.2]}  # s2
        self.rmse_ticklabels_dict = self.rmse_ticks_dict    # s
        self.mse_ticklabels_dict = self.mse_ticks_dict      # s2


class EventFrequency(PropClass, MyGamma):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'event_freq'
        self.label = 'Event frequency'  # str
        self.unit = '1/s'
        self.prop_min, self.prop_max = 0, 4  # 1/s
        self.prop_resolution = 0.1  # 1/s

        # Spatial-temporal model
        # self.model = SpaceModel()

        # Distribution properties
        self.set_bins()
        self.par_names = ['median', 'var']
        self.par_labels = ['median (1/s)', r'variance (1/s$^2$)']
        self.par_bounds_pdf = [(self.epsilon, 4), (self.epsilon, 0.5)]  # mode, var
        self.par_names_pdf = ['median', 'var']

        # Plot settings
        self.prop_lim = [0, 4]
        self.prop_ticks = [0, 2, 4]
        self.prop_ticklabels = [0, 2, 4]
        # Median, fitted parameters (MyGamma)
        self.par_lims = [[0.5, 1.5], [0, 0.5]]
        self.par_ticks = [[0.5, 1, 1.5], [0, 0.25, 0.5]]
        self.par_ticklabels = self.par_ticks

        self.par_lim_dict = {
            'p1': [-0.4, 0.4], 'p0': [0, 2],
            'a_pos': [-0.02, 0.02], 'a_neg': [-0.02, 0.02], 'b': [0, 2],
            'a_log': [-4, 4], 'b_log': [-0.2, 0.2],
        }


class Speed(PropClass, MyGamma):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'average_speed'
        self.label = 'Speed'  # str
        self.unit = 'cm/s'
        self.prop_min, self.prop_max = 0, 2  # cm/s
        self.prop_resolution = 0.05  # cm/s
        # Distribution properties
        self.set_bins()

        # Spatial-temporal model
        # self.model = SpaceModel()

        # Fitted parameters (Gamma)
        self.par_names = ['mode', 'var']
        self.par_labels = ['mode (cm/s)', r'variance (cm$^2$/s$^2$)']
        self.par_bounds = [(self.epsilon, 2), (self.epsilon, 0.5)]  # mode, var

        # Plot settings
        self.prop_lim = [0, 2]  # cm/s
        self.prop_ticks = [0, 1, 2]  # cm/s
        self.prop_ticklabels = [0, 1, 2]  # cm/s
        # # Fitted parameters (median, distribution parameters)
        self.par_lims = [[0, 1.5], [0, .4]]
        self.par_ticks = [[0, 0.5, 1, 1.5], [0, 0.2, 0.4]]
        self.par_ticklabels = self.par_ticks
        self.par_lim_dict = {
            'p1': [-0.4, 0.4], 'p0': [0, 2],
            'a_pos': [-0.02, 0.02], 'a_neg': [-0.02, 0.02], 'b': [0, 2],
            'a_log': [-2, 2], 'b_log': [-0.22, 0.22],
        }


class Brightness(PropClass):
    def __init__(self):
        super().__init__()

        # General properties
        self.prop_name = 'brightness'
        self.label = 'Brightness'
        self.unit = 'Lux'
        self.prop_min, self.prop_max = 0, 600   # lux
        self.prop_resolution = 10               # lux

        # Plot settings
        self.par_lims = [[0, 600]]
        self.par_ticks = [[0, 100, 200, 300, 400, 500, 600]]
        self.par_ticklabels = [[0, 100, 200, 300, 400, 500, 600]]

