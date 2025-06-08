
# Third party library imports
import numpy as np
from numba import jit
from scipy.signal import lfilter, lfiltic

# Local library imports
from utils.general_utils import get_b_values


# Helper functions ############################################################
@jit(nopython=True)
def weighted_signed_sum(x, w_pos, w_neg):
    return x * (w_pos * (x > 0) + w_neg * (x < 0))


# Parent class for all models #################################################
class Model:
    def __init__(self):
        self.label = 'Model'
        self.name = 'model'
        self.all_par_names = [
            'tau_lpf',  # Time-constant for low-pass filter
            'a',        # Constant a
            'bA',       # Eye-averaging pathway (logarithmic modulation)
            # Eye-averaging derivative pathway
            'wAD_pos', 'wAD_neg',   # positive changes (increase); negative changes (decrease)
            # Eye-specific derivative pathway
            'wD_pos_left', 'wD_neg_left', 'wD_pos_right', 'wD_neg_right',
            # Spatial contrast pathway
            'wC_pos', 'wC_neg',     # right brighter than left; right darker than left
            # Eye-specific derivative average pathway
            'wDA_pos', 'wDA_neg',   # positive changes (increase); negative changes (decrease)
        ]

        self.par_bounds_dict = {
            'tau_lpf': (0, 10),
            'a': (-np.inf, np.inf),
            'bA': (-300, 300),
            'wAD_pos': (-300, 300), 'wAD_neg': (-300, 300),
            'wD_pos_left': (-300, 300), 'wD_neg_left': (-300, 300), 'wD_pos_right': (-300, 300), 'wD_neg_right': (-300, 300),
            'wC_pos': (-300, 300), 'wC_neg': (-300, 300),
            'wDA_pos': (-300, 300), 'wDA_neg': (-300, 300),
            # Also include constrained weights
            'wAD': (-300, 300),
            'wD': (-300, 300), 'wD_pos': (-300, 300), 'wD_neg': (-300, 300),
            'wC': (-300, 300),
            'wDA': (-300, 300),
        }
        self.par_x0_dict = {
            'tau_lpf': 1.5,
            'a': 50,
            'bA': 100,
            'wAD_pos': 100, 'wAD_neg': 100,
            'wD_pos_left': 100, 'wD_neg_left': 100, 'wD_pos_right': 100, 'wD_neg_right': 100,
            'wC_pos': 100, 'wC_neg': 100,
            'wDA_pos': 100, 'wDA_neg': 100,
            # Also include constrained possibilities
            'wAD': 100,
            'wD': 100, 'wD_pos': 100, 'wD_neg': 100,
            'wC': 100,
            'wDA': 100,
        }
        self.par_names = []

        self.param_dict = {}

        # Fitting settings: set in set_x0() and set_bounds()
        self.epsilon = 1e-5
        self.n_par = None
        self.x0 = None
        self.bounds_curve_fit = None

        # Stimulus settings: set in set_stimulus()
        self.dt = None
        self.b_left, self.b_right = None, None
        self.b_left_all, self.b_right_all = None, None  # for integrator data

        # Plot settings: labels
        self.par_labels_dict = {
            'tau_lpf': 'Time constant\n(s)',
            'a': 'Offset',
            'bA': 'Ave. weight',
            'wAD_pos': 'Pos. AD weight',
            'wAD_neg': 'Neg. AD weight',
            'wD_pos_left': 'Left Pos. D weight', 'wD_neg_left': ' Left Neg. D weight',
            'wD_pos_right': 'Right Pos. D weight', 'wD_neg_right': ' Right Neg. D weight',
            'wC_pos': 'Pos. C weight',
            'wC_neg': 'Neg. C weight',
            'wDA_pos': 'Pos. D weight',
            'wDA_neg': 'Neg. D weight',
            # Also include constrained weights
            'wAD': 'AD weight',
            'wD': 'D weight',
            'wD_pos': 'Increase weight', 'wD_neg': 'Decrease weight',
            'wC': 'Contrast weight\n',  # Add extra line to align with other labels
            'wDA': 'DA weight',
        }

    # General functions #######################################################
    def get_param_dict(self, params):
        """Apply constraints and convert to dictionary.
        Can be overwritten by other child classes"""
        self.param_dict = dict(zip(self.par_names, params))

    def add_absolute_operator(self, par_name):
        """Add antisymmetric constraint to parameter dictionary.
        This results in a weighted absolute operator"""
        self.param_dict[f'{par_name}_pos'] = self.param_dict[par_name]
        self.param_dict[f'{par_name}_neg'] = -1 * self.param_dict[par_name]

    def add_linear_operator(self, par_name):
        """Add symmetric constraint to parameter dictionary.
        This results in a linear transfer function."""
        self.param_dict[f'{par_name}_pos'] = self.param_dict[par_name]
        self.param_dict[f'{par_name}_neg'] = self.param_dict[par_name]

    def add_eye_symmetric_constraint(self, par_name):
        """Constrain left weights to have same sign as right weights."""
        self.param_dict[f'{par_name}_left'] = self.param_dict[par_name]
        self.param_dict[f'{par_name}_right'] = self.param_dict[par_name]

    def add_anti_eye_symmetric_constraint(self, par_name):
        """Constrain left weights to have opposite sign to right weights."""
        self.param_dict[f'{par_name}_left'] = -1 * self.param_dict[par_name]
        self.param_dict[f'{par_name}_right'] = self.param_dict[par_name]

    def add_absolute_operator_around_offset(self, par_name):
        a = self.param_dict['a']
        self.param_dict[f'{par_name}_pos'] = self.param_dict[par_name] + a
        self.param_dict[f'{par_name}_neg'] = -1 * (self.param_dict[par_name] + a)

    def get_all_params(self, params):
        """Collect all parameters. Set unused parameters to 0."""
        self.get_param_dict(params)
        res = [self.param_dict.get(par_name, 0) for par_name in self.all_par_names]
        return res

    def set_x0(self):
        self.x0 = [self.par_x0_dict[par_name] for par_name in self.par_names]

    def set_bounds(self, prop_name=None):
        bounds_curve_fit = []
        for par_name in self.par_names:
            par_lim = self.par_bounds_dict[par_name]
            if prop_name == 'percentage_left' and par_name == 'a' and not 'bA' in self.par_names:
                # Set bounds for the offset to be 50, but not if logarithmic
                # modulation (bA) is included
                par_lim = [50 - self.epsilon, 50 + self.epsilon]
                # Update number of parameters accordingly
                self.set_npar(prop_name)
            bounds_curve_fit.append(par_lim)
        self.bounds_curve_fit = np.transpose(bounds_curve_fit)

    def set_npar(self, prop_name=None):
        """Get number of parameters for curve_fit."""
        self.n_par = len(self.par_names)
        if prop_name == 'percentage_left' and 'a' in self.par_names and not 'bA' in self.par_names:
            # Bounds for the offset are fixed to be 50
            self.n_par -= 1
        return self.n_par

    def set_stimulus(self, ts, t_ns, b_left_ns, b_right_ns):
        self.dt = np.mean(np.diff(ts))  # should be same for each ts
        self.b_left, self.b_right = get_b_values(ts, t_ns, b_left_ns, b_right_ns)

    def set_stimulus_integrator(self, ts_all, b_left_all, b_right_all):
        self.dt = np.mean(np.diff(ts_all))  # should be same for each ts
        self.b_left_all, self.b_right_all = b_left_all, b_right_all

    # Low-pass filter #########################################################
    @staticmethod
    def lpf_continuous(y, tau, sampling_period):
        # Calculate the alpha value
        alpha = sampling_period / (tau + sampling_period)

        # Define the filter coefficients
        b = [alpha]
        a = [1, alpha - 1]

        # Compute initial conditions such that we start with the first value of the data
        zi = lfiltic(b, a, y=[y[0]])

        # Apply the filter using lfilter
        filtered_data, _ = lfilter(b, a, y, zi=zi)
        return filtered_data

    def lpf_discrete(self, y, tau, sampling_period):
        y_prev = y[0]  # Initial boundary condition: assume steady state
        filtered_data = []
        for x in y:
            filtered_data.append(
                self.lpf_discrete_step(x, y_prev, tau, sampling_period)
            )
            y_prev = filtered_data[-1]

        return filtered_data

    @staticmethod
    @jit(nopython=True)
    def lpf_discrete_step(x, y_prev, tau, sampling_period):
        # alpha = (2 * np.pi * cutoff_freq * sampling_period) / (2 * np.pi * cutoff_freq * sampling_period + 1)
        alpha = sampling_period / (tau + sampling_period)
        return alpha * x + (1 - alpha) * y_prev

    # General functions for evaluating the model ##############################
    @staticmethod
    # @jit(nopython=True)  # TODO uncomment after debugging
    def _eval_step(
            # Inputs (visual stimulus)
            left_eye_lux, right_eye_lux, lpf_left_eye, lpf_right_eye,
            # Parameters
            tau_lpf,  # We keep this parameter here
            a,
            bA,
            wAD_pos, wAD_neg,
            wD_pos_left, wD_neg_left, wD_pos_right, wD_neg_right,
            wC_pos, wC_neg,
            wDA_pos, wDA_neg,
    ):
        """General function containing pathways for all models."""
        # Convert to kLux for higher resolution of weights-fitting
        factor = 1000
        left_eye_lux, right_eye_lux = left_eye_lux / factor, right_eye_lux / factor
        lpf_left_eye, lpf_right_eye = lpf_left_eye / factor, lpf_right_eye / factor

        # Process input values
        # # Compute average across the eyes
        average_eye_lux = (left_eye_lux + right_eye_lux) / 2
        # # Compute average low-pass filter (allowed since we have a linear low-pass filter)
        lpf_average = (lpf_left_eye + lpf_right_eye) / 2
        # # Compute change in light intensity by subtracting the low-pass filtered value
        diff_average = average_eye_lux - lpf_average
        diff_left = left_eye_lux - lpf_left_eye
        diff_right = right_eye_lux - lpf_right_eye
        # # Compute contrast between left and right eye
        contrast = left_eye_lux - right_eye_lux   # positive if left is brighter

        # Compute pathways
        # # Constant a
        values_a = a
        # # Eye-averaging pathway (logarithmic modulation)
        values_A = bA * np.log(average_eye_lux + 0.0001)    # Ensure average is always positive
        # # Eye-averaging derivative pathway
        values_AD = weighted_signed_sum(diff_average, wAD_pos, wAD_neg)
        # # Eye-specific derivative pathway
        values_D_left = weighted_signed_sum(diff_left, wD_pos_left, wD_neg_left)
        values_D_right = weighted_signed_sum(diff_right, wD_pos_right, wD_neg_right)
        # # Spatial contrast pathway
        values_C = weighted_signed_sum(contrast, wC_pos, wC_neg)
        # # Eye-specific derivative average pathway
        values_DA_left = weighted_signed_sum(diff_left, wDA_pos, wDA_neg)
        values_DA_right = weighted_signed_sum(diff_right, wDA_pos, wDA_neg)
        values_DA = (values_DA_left + values_DA_right) / 2

        # Combine all pathways
        return (
            values_a
            + values_A
            + values_AD
            + values_D_left
            + values_D_right
            + values_C
            + values_DA
        )

    def _eval_cont(self, left_eye_lux, right_eye_lux, sampling_period, all_params: list):
        """General function for evaluating the model with continuous stimulus input."""
        # Split parameters
        tau_lpf = all_params[0]
        model_params = all_params[1:]

        # Calculate low-pass filtered (LPF) values
        lpf_left_eye = self.lpf_continuous(left_eye_lux, tau_lpf, sampling_period)
        lpf_right_eye = self.lpf_continuous(right_eye_lux, tau_lpf, sampling_period)

        return self._eval_step(
            # Inputs
            left_eye_lux, right_eye_lux, lpf_left_eye, lpf_right_eye,
            # Parameters
            *all_params
        )

    def eval_cont(
            self,
            # Inputs (visual stimulus)
            b_left, b_right, dt,
            # Parameters
            *params,
    ):
        all_params = self.get_all_params(params)
        return self._eval_cont(
            # Inputs
            b_left, b_right, dt,
            # Parameters
            all_params  # Give list of parameters to _eval_cont
        )

    def eval_step(self, left_eye_lux, right_eye_lux, lpf_left_eye, lpf_right_eye, *params):
        all_params = self.get_all_params(params)
        return self._eval_step(
            # Inputs
            left_eye_lux, right_eye_lux, lpf_left_eye, lpf_right_eye,
            # Parameters
            *all_params
        )

    # # Model specific functions (to be overwritten in subclasses) ##############
    # def fitfunc(self, x, *parameters):
    #     """Wrapper for scipy curve_fit. Will be overwritten for each model"""
    #     raise NotImplementedError


# Full model including all pathways and fully unconstrained
class FullModel(Model):
    def __init__(self):
        super().__init__()
        self.label = 'A+AD+D+C+DA'
        self.name = 'a_ad_d_c_da'
        self.par_names = self.all_par_names
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    # Model specific functions (to be overwritten in subclasses) ##############
    def fitfunc(
            self,
            x,  # Scipy curve_fit input (just a placeholder)
            tau_lpf,
            a,
            bA,
            wAD_pos, wAD_neg,
            wD_pos_left, wD_neg_left, wD_pos_right, wD_neg_right,
            wC_pos, wC_neg,
            wDA_pos, wDA_neg,
    ):
        """Wrapper for scipy curve_fit. Uses all parameters from Model.all_par_names."""
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            tau_lpf,
            a,
            bA,
            wAD_pos, wAD_neg,
            wD_pos_left, wD_neg_left, wD_pos_right, wD_neg_right,
            wC_pos, wC_neg,
            wDA_pos, wDA_neg,
        )

    def fitfunc_integrator(
            self,
            x,  # Scipy curve_fit input (just a placeholder)
            tau_lpf,
            a,
            bA,
            wAD_pos, wAD_neg,
            wD_pos_left, wD_neg_left, wD_pos_right, wD_neg_right,
            wC_pos, wC_neg,
            wDA_pos, wDA_neg,
    ):
        """Wrapper to fit integrator data for scipy curve_fit."""
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    tau_lpf,
                    a,
                    bA,
                    wAD_pos, wAD_neg,
                    wD_pos_left, wD_neg_left, wD_pos_right, wD_neg_right,
                    wC_pos, wC_neg,
                    wDA_pos, wDA_neg,
                )
            )

        return np.reshape(y_hat, -1)


class FullModelFig2and3(FullModel):
    def __init__(self):
        super().__init__()
        self.name = 'full_fig2and3'


# Constant: only having the offset
class BlindModel(Model):
    def __init__(self):
        super().__init__()
        self.label = 'None'
        self.name = 'none'
        self.par_names = ['a', ]
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def fitfunc(self, x, a):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            a,
        )

    def fitfunc_integrator(self, x, a):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    a,
                )
            )
        return np.reshape(y_hat, -1)


# Eye-averaging pathway
class ModelA(Model):
    def __init__(self):
        super().__init__()
        self.label = 'A'
        self.name = 'a'
        self.par_names = ['a', 'bA']
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def fitfunc(self, x, a, bA):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            a, bA
        )

    def fitfunc_integrator(self, x, a, bA):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    a, bA
                )
            )
        return np.reshape(y_hat, -1)


class ModelAFig2(ModelA):
    def __init__(self):
        super().__init__()
        self.name = 'a_fig2'


class ModelAFig3(ModelA):
    def __init__(self):
        super().__init__()
        self.name = 'a_fig3'


# Eye-averaging derivative pathway, allowing different weights for pos and negative changes
# # Following Karpenko et al. (2020) for turn probability
class ModelAD(Model):
    def __init__(self):
        super().__init__()
        self.label = 'AD'
        self.name = 'ad'
        self.par_names = ['tau_lpf', 'a', 'wAD_pos', 'wAD_neg']
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def fitfunc(self, x, tau_lpf, a, wAD_pos, wAD_neg):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            tau_lpf, a, wAD_pos, wAD_neg
        )

    def fitfunc_integrator(self, x, tau_lpf, a, wAD_pos, wAD_neg):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    tau_lpf, a, wAD_pos, wAD_neg
                )
            )
        return np.reshape(y_hat, -1)


class ModelADFig2(ModelAD):
    def __init__(self):
        super().__init__()
        self.name = 'ad_fig2'


class ModelADFig3(ModelAD):
    def __init__(self):
        super().__init__()
        self.name = 'ad_fig3'


# Eye-specific derivative pathway
# Following Chen et al. 2021 for left turns
class ModelD(Model):
    def __init__(self):
        super().__init__()
        self.label = 'D'
        self.name = 'd'
        self.par_names = [
            'tau_lpf',
            'a',
            'wD_pos', 'wD_neg',
        ]
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def get_param_dict(self, params):
        self.param_dict = dict(zip(self.par_names, params))
        # Ensure left eye weights are opposite of right eye weights
        # If left and right weights are of same sign, the model is equivalent
        # to the ModelDA (eye-specific derivative averaging pathway)
        self.add_anti_eye_symmetric_constraint('wD_pos')
        self.add_anti_eye_symmetric_constraint('wD_neg')

    def fitfunc(self, x, tau_lpf, a, wD_pos, wD_neg):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            tau_lpf, a, wD_pos, wD_neg,
        )

    def fitfunc_integrator(self, x, tau_lpf, a, wD_pos, wD_neg):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    tau_lpf, a, wD_pos, wD_neg,
                )
            )
        return np.reshape(y_hat, -1)


# Eye-specific derivative averaging pathway, allowing different weights for pos and negative changes
class ModelDA(Model):
    def __init__(self):
        super().__init__()
        self.label = 'DA'
        self.name = 'da'
        self.par_names = [
            'tau_lpf',
            'a',
            'wDA_pos', 'wDA_neg',
        ]
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def fitfunc(self, x, tau_lpf, a, wDA_pos, wDA_neg):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            tau_lpf, a, wDA_pos, wDA_neg
        )

    def fitfunc_integrator(self, x, tau_lpf, a, wDA_pos, wDA_neg):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    tau_lpf, a, wDA_pos, wDA_neg,
                )
            )
        return np.reshape(y_hat, -1)


# Spatial contrast pathway
class ModelC(Model):
    def __init__(self):
        super().__init__()
        self.label = 'C'
        self.name = 'c'
        self.par_names = ['a', 'wC_pos', 'wC_neg']
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def fitfunc(self, x, a, wC_pos, wC_neg):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            a, wC_pos, wC_neg
        )

    def fitfunc_integrator(self, x, a, wC_pos, wC_neg):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    a, wC_pos, wC_neg,
                )
            )
        return np.reshape(y_hat, -1)


# # Following Karpenko et al. (2020) for left turns
class ModelCSign(Model):
    def __init__(self):
        super().__init__()
        self.label = 'C (Sign.)'
        self.name = 'c_sign'
        self.par_names = ['a', 'wC']
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def get_param_dict(self, params):
        self.param_dict = dict(zip(self.par_names, params))
        # Create weighted linear operator: maintain direction of contrast
        self.add_linear_operator('wC')

    def fitfunc(self, x, a, wC):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            a, wC
        )

    def fitfunc_integrator(self, x, a, wC):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    a, wC,
                )
            )
        return np.reshape(y_hat, -1)


class ModelCAbs(Model):
    def __init__(self):
        super().__init__()
        self.label = 'C (Abs.)'
        self.name = 'c_abs'
        self.par_names = ['a', 'wC']
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def get_param_dict(self, params):
        self.param_dict = dict(zip(self.par_names, params))
        # Create weighted absolute operator: detection of contrast presence
        self.add_absolute_operator('wC')

    def fitfunc(self, x, a, wC):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            a, wC
        )

    def fitfunc_integrator(self, x, a, wC):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    a, wC,
                )
            )
        return np.reshape(y_hat, -1)


# Figure 3 model: Eye-specific derivative pathway + Spatial contrast pathway
class ModelD_C(Model):
    def __init__(self):
        super().__init__()
        self.label = 'D+C'
        self.name = 'd_c'
        self.par_names = [
            'tau_lpf', 'a',
            'wD_pos', 'wD_neg',
            'wC',
        ]
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def get_param_dict(self, params):
        self.param_dict = dict(zip(self.par_names, params))
        # Ensure left eye weights are opposite of right eye weights
        self.add_anti_eye_symmetric_constraint('wD_pos')
        self.add_anti_eye_symmetric_constraint('wD_neg')
        # Create weighted linear operator: maintain direction of contrast
        self.add_linear_operator('wC')

    def fitfunc(self, x, tau_lpf, a, wD_pos, wD_neg, wC):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            tau_lpf, a, wD_pos, wD_neg, wC
        )

    def fitfunc_integrator(self, x, tau_lpf, a, wD_pos, wD_neg, wC):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    tau_lpf, a, wD_pos, wD_neg, wC,
                )
            )
        return np.reshape(y_hat, -1)


# Eye-averaging pathway + Eye-specific derivative averaging pathway
class ModelA_DA(Model):
    def __init__(self):
        super().__init__()
        self.label = 'A+DA'
        self.name = 'a_da'
        self.par_names = [
            'tau_lpf', 'a', 'bA',
            'wDA_pos', 'wDA_neg',
        ]
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def fitfunc(self, x, tau_lpf, a, bA, wDA_pos, wDA_neg):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            tau_lpf, a, bA, wDA_pos, wDA_neg
        )

    def fitfunc_integrator(self, x, tau_lpf, a, bA, wDA_pos, wDA_neg):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    tau_lpf, a, bA, wDA_pos, wDA_neg,
                )
            )
        return np.reshape(y_hat, -1)


class ModelA_CAbs(Model):
    def __init__(self):
        super().__init__()
        self.label = 'A+C (Abs.)'
        self.name = 'a_c_abs'
        self.par_names = [
            'a', 'bA',
            'wC',
        ]
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def get_param_dict(self, params):
        self.param_dict = dict(zip(self.par_names, params))
        # Create weighted absolute operator: detection of contrast presence
        self.add_absolute_operator('wC')

    def fitfunc(self, x, a, bA, wC):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            a, bA, wC
        )

    def fitfunc_integrator(self, x, a, bA, wC):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    a, bA, wC,
                )
            )
        return np.reshape(y_hat, -1)


class ModelA_D_CAbs(Model):
    def __init__(self):
        super().__init__()
        self.label = 'A+D+C (Abs.)'
        self.name = 'a_d_c_abs'
        self.par_names = [
            'tau_lpf', 'a', 'bA',
            'wD_pos', 'wD_neg',
            'wC',
        ]
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def get_param_dict(self, params):
        self.param_dict = dict(zip(self.par_names, params))
        # Ensure left eye weights are opposite of right eye weights
        self.add_anti_eye_symmetric_constraint('wD_pos')
        self.add_anti_eye_symmetric_constraint('wD_neg')
        # Create weighted absolute operator: detection of contrast presence
        self.add_absolute_operator('wC')

    def fitfunc(self, x, tau_lpf, a, bA, wD_pos, wD_neg, wC):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            tau_lpf, a, bA, wD_pos, wD_neg, wC
        )

    def fitfunc_integrator(self, x, tau_lpf, a, bA, wD_pos, wD_neg, wC):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    tau_lpf, a, bA, wD_pos, wD_neg, wC,
                )
            )
        return np.reshape(y_hat, -1)



class ModelDA_CAbs(Model):
    def __init__(self):
        super().__init__()
        self.label = 'DA+C (Abs.)'
        self.name = 'da_c_abs'
        self.par_names = [
            'tau_lpf', 'a',
            'wDA_pos', 'wDA_neg',
            'wC',
        ]
        self.set_x0()
        self.set_bounds()
        self.set_npar()

    def get_param_dict(self, params):
        self.param_dict = dict(zip(self.par_names, params))
        # Create weighted absolute operator: detection of contrast presence
        self.add_absolute_operator('wC')

    def fitfunc(self, x, tau_lpf, a, wDA_pos, wDA_neg, wC):
        return self.eval_cont(
            # Inputs: based on set_stimulus()
            self.b_left, self.b_right, self.dt,
            # Parameters
            tau_lpf, a, wDA_pos, wDA_neg, wC
        )

    def fitfunc_integrator(self, x, tau_lpf, a, wDA_pos, wDA_neg, wC):
        y_hat = []
        # Inputs: based on set_stimulus()
        for b_left, b_right in zip(self.b_left_all, self.b_right_all):
            y_hat.append(
                self.eval_cont(
                    b_left, b_right, self.dt,
                    # Parameters
                    tau_lpf, a, wDA_pos, wDA_neg, wC,
                )
            )
        return np.reshape(y_hat, -1)

