from numba import jit
from numba.np.extensions import cross2d
import numpy as np
import pandas as pd
from pathlib import Path

from settings.agent_settings import Larva, Juvie
from settings.prop_settings import PercentageTurns, PercentageLeft, TurnAngle, OrientationChange, Distance, EventFrequency, TotalDuration, Speed
from utils.models import *


# Agent class #################################################################
class MyAgent:
    """Agent class for simulating fish behavior in a virtual environment.

    Attributes:
        rng (np.random.Generator): Random number generator for the agent.
        agent_index (int): Unique index for the agent.
        agent_age (int): Age of the agent (e.g., 5 or 27 dpf).
        agent_genotype_dict (dict): Genotype configuration for the agent.
        path_to_input_folder (Path): Path to input data for model parameters.
        r_max (float): Maximum radius of the arena (cm).
        verbose (bool): If True, prints debug information.
        tracking_columns (list): Columns for tracking data.
        bout_columns (list): Columns for bout data.
        agent_state_variables (dict): State variables for the agent.
        meta_popt_dict (dict): Model parameters for each property.
    """

    def __init__(
            self, seed: int,
            agent_index: int, agent_age: int, agent_genotype_dict: dict,
            path_to_input_folder: Path,
            r_max: float = 6,
    ):
        """Initializes a MyAgent instance.

        Args:
            seed (int): Random seed for reproducibility.
            agent_index (int): Unique agent identifier.
            agent_age (int): Age of the agent in days (5 or 27).
            agent_genotype_dict (dict): Genotype configuration.
            path_to_input_folder (Path): Path to input data.
            r_max (float, optional): Arena radius in cm. Defaults to 6.
        """
        self.rng = np.random.default_rng(seed)
        self.agent_index = agent_index
        self.agent_age = agent_age
        self.agent_genotype_dict = agent_genotype_dict
        self.path_to_input_folder = path_to_input_folder
        self.r_max = r_max  # cm
        self.verbose = False  # for debugging

        # Prepare data storage ################################################
        self.tracking_columns = [
            'time', 'fish_contour_area', 'x_position', 'y_position',
            'accumulated_path', 'accumulated_orientation', 'accumulated_orientation_windowed_variance',
            'additional_tracking_info0', 'additional_tracking_info1', 'additional_tracking_info2',
            'time_absolute',
        ]
        self.bout_columns = [
            # Start of bout
            'start_time', 'start_contour_area', 'start_x_position', 'start_y_position',
            'start_accumulated_path', 'start_accumulated_orientation',
            'start_accumulated_orientation_windowed_variance',
            'start_additional_tracking_info0', 'start_additional_tracking_info1', 'start_additional_tracking_info2',
            # End of bout
            'end_time', 'end_contour_area', 'end_x_position', 'end_y_position',
            'end_accumulated_path', 'end_accumulated_orientation', 'end_accumulated_orientation_windowed_variance',
            'end_additional_tracking_info0', 'end_additional_tracking_info1', 'end_additional_tracking_info2',
            # Bout properties
            'duration', 'x_position_change', 'y_position_change',
            'distance_change', 'estimated_orientation_change', 'average_speed', 'interbout_interval',
            'same_direction_as_previous_bout',
            'start_time_absolute', 'end_time_absolute'
        ]

        # Create empty agent state variables
        self.agent_state_variables = dict({})

        # Prepare model #######################################################
        # Load swim property classes
        self.percentage_turns_class = PercentageTurns()
        self.percentage_left_class = PercentageLeft()
        self.turn_angle_class = TurnAngle()
        self.bout_duration_class = TotalDuration()
        self.displacement_class = Distance()
        # Prepare dictionaries for model parameters
        self.meta_popt_dict = {
            self.percentage_turns_class.prop_name: {},
            self.percentage_left_class.prop_name: {},
            self.turn_angle_class.prop_name: {},
            self.bout_duration_class.prop_name: {},
            self.displacement_class.prop_name: {},
        }
        # Load model and model parameter for this fish_age and fish_genotype
        self.load_model(agent_index, agent_age, agent_genotype_dict)

    # #########################################################################
    # Loading model parameters
    # #########################################################################
    def load_model(self, agent_index: int, agent_age: int, agent_genotype_dict: dict):
        """Loads model parameters for the agent based on age and genotype.

        Args:
            agent_index (int): Agent identifier.
            agent_age (int): Age of the agent.
            agent_genotype_dict (dict): Genotype configuration.

        Raises:
            UserWarning: If a required genotype is not provided.
            ValueError: If an unknown genotype is encountered.
        """
        # Age specific settings ###############################################
        if agent_age == 5:
            self.speed_decay = 1 / 10  # 1/s
            self.boundary_turn_angle = 90 + 15  # deg
            self.boundary_turn_angle = 174  # deg
            self.boundary_turn_std = 31     # deg
            self.agent_class = Larva()
            # Symmetric and Left-biased distribution (mode1, mode2, ratio1, ratio2, sigma3)
            self.d_ang_popt_sym = [27, 27,  0.23,  0.23,  4.5]   # fig2.py
            self.d_ang_popt_st = [42.42, 28.29,  0.63,  0.12,  5.03]   # fig3.py
        else:
            self.speed_decay = 1 / 5  # 1/s
            self.boundary_turn_angle = 90 + 15  # deg
            self.boundary_turn_angle = 150  # deg
            self.boundary_turn_std = 56     # deg
            self.agent_class = Juvie()
            # Symmetric and Left-biased distribution (mode1, mode2, ratio1, ratio2, sigma3)
            self.d_ang_popt_sym = [30, 30,  0.25,  0.25, 38.04]   # fig2.py
            self.d_ang_popt_st = [36.62, 43.27,  0.36,  0.18, 23.96]   # fig3.py

        # Load property model parameters ######################################
        for prop_class in [self.percentage_turns_class, self.percentage_left_class, self.turn_angle_class, self.bout_duration_class, self.displacement_class]:
            prop_name = prop_class.prop_name

            if self.verbose:
                print(f"Loading model parameters for {prop_name} ...", end='')
            prop_genotype = agent_genotype_dict.get(prop_name, None)

            if isinstance(prop_genotype, type(None)):
                raise UserWarning("prop_genotype not given!")

            # Set prop_genotype to lower case to avoid case sensitivity issues
            prop_genotype = prop_genotype.lower()

            # Split up genotype and multiplication factor if given
            factor = 1  # Default multiplication factor is 1
            if '_x' in prop_genotype:
                prop_genotype, factor_str = prop_genotype.split('_x')
                factor = float(factor_str)

            if isinstance(prop_genotype, int):
                # Assign a fixed, integer value
                self.load_model_pars_int(prop_name, prop_genotype)
            elif prop_genotype == 'blind' or prop_genotype == 'azimuth_virtual':
                self.load_model_pars(prop_name, self.agent_class, agent_index, 'azimuth_left_dark_right_bright_virtual_yes', 'log', factor)
            elif prop_genotype == 'azimuth':
                self.load_model_pars(prop_name, self.agent_class, agent_index, 'azimuth_left_dark_right_bright', 'log', factor)
            elif prop_genotype == 'contrast':
                self.load_model_pars(prop_name, self.agent_class, agent_index, 'contrast', 'lin', factor)
            elif prop_genotype == 'temporal':
                self.load_model_pars(prop_name, self.agent_class, agent_index, 'temporal', 'double_linear', factor)
            elif 'st' in prop_genotype:
                if prop_genotype == 'st_a':
                   prop_class.model = ModelAFig3()
                elif prop_genotype == 'st_c':
                    prop_class.model = ModelC()  # Fig. 3s data
                elif prop_genotype == 'st_d':
                    prop_class.model = ModelD()
                elif prop_genotype == 'st_d_c':
                    prop_class.model = ModelD_C()
                elif prop_genotype == 'st_a_da':
                    prop_class.model = ModelA_DA()
                elif prop_genotype == 'st_ad':
                    prop_class.model = ModelADFig3()
                elif prop_genotype == 'st_da':
                    prop_class.model = ModelDA()
                elif prop_genotype == 'st_da_c':
                    prop_class.model = ModelDA_CAbs()
                elif prop_genotype == 'st_superfit':
                    prop_class.model = FullModelFig2and3()
                else:
                    raise ValueError(f"Unknown genotype for {prop_name}: {prop_genotype}")
                # Factor only applies to wC in spatial-temporal models
                self.load_spatial_temporal_model_pars(prop_name, prop_class.model, self.agent_class, prop_class.model.name, factor)
            else:
                raise ValueError(f"Unknown genotype for {prop_name}: {prop_genotype}")
            if self.verbose:
                print(f"{prop_genotype} done.")

    def load_model_pars(self, prop_name, agent_class, agent_index, key_base, model_type, factor=1):
        """Loads model parameters for a property from HDF5 files.

        Args:
            prop_name (str): Property name.
            agent_class: Agent class (Larva or Juvie).
            agent_index (int): Agent index.
            key_base (str): Key for HDF5 file.
            model_type (str): Model type ('log', 'lin', 'double_linear').
            factor (float, optional): Multiplicative factor. Defaults to 1.

        Raises:
            ValueError: If model_type is unknown.
        """
        agent_index = 0  # We now take only one model for all fish
        meta_fit_df = pd.read_hdf(self.path_to_input_folder.joinpath(f'fit_df_{key_base}.hdf5'), key=f"{key_base}_meta_mean")

        if model_type == 'log':
            col_names = ['a_log', 'b_log']
        elif model_type == 'lin':
            col_names = ['p0', 'p1']
        elif model_type == 'double_linear':
            col_names = ['a_pos', 'a_neg', 'b']
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        par_names = ['median']  # Replace this with ['mode', 'var'] when drawing from distributions
        par_dict = {}
        for par_name in par_names:
            par_dict[par_name] = dict()
            for col_name in col_names:
                par_dict[par_name][col_name] = (
                    meta_fit_df
                    .xs(prop_name, level='prop_name')
                    .query(agent_class.query)
                    .xs(agent_index, level='experiment_ID')
                    .xs(par_name, level='par_name')
                    [col_name]
                    .values[0]
                )

                if col_name == 'b_log' or col_name == 'p1' or col_name == 'a_pos' or col_name == 'a_neg':
                    # Multiply by factor if given
                    par_dict[par_name][col_name] *= factor

        # Store directly in self.meta_popt_dict
        self.meta_popt_dict[prop_name] = par_dict

    def load_spatial_temporal_model_pars(self, prop_name, model, agent_class, key_base, factor=1):
        """Loads spatial-temporal model parameters for a property.

        Args:
            prop_name (str): Property name.
            model: Model object.
            agent_class: Agent class (Larva or Juvie).
            key_base (str): Key for HDF5 file.
            factor (float, optional): multiplicative factors for wC
        """
        meta_fit_df = pd.read_hdf(self.path_to_input_folder.joinpath(f'fit_df_{key_base}.hdf5'), key=f"{key_base}_meta_mean")

        col_names = model.par_names
        par_names = ['median']
        par_dict = {}
        for par_name in par_names:
            par_dict[par_name] = dict()
            for col_name in col_names:
                par_dict[par_name][col_name] = (
                    meta_fit_df
                    .xs(prop_name, level='prop_name')
                    .query(agent_class.query)
                    .xs(par_name, level='par_name')
                    [col_name]
                    .values[0]
                )

                if col_name == 'wC':
                    # Multiply Contrast pathway weight with factor, if given
                    par_dict[par_name][col_name] *= factor

        # Store directly in self.meta_popt_dict
        self.meta_popt_dict[prop_name] = par_dict

    def load_model_pars_int(self, prop_name, par_value):
        """Sets fixed model parameters for a property.

        Args:
            prop_name (str): Property name.
            par_value (int): Fixed property value.
        """
        # Store directly in self.meta_popt_dict
        self.meta_popt_dict[prop_name] = {
            'median': {
                'p0': par_value,
                'p1': 0,  # no slope
            }
        }

    # #########################################################################
    # Getting swim properties
    # #########################################################################
    def low_pass_filter(self, time_since_last_processed_frame, left_eye_lux, right_eye_lux, agent_genotype_dict: dict):
        """Applies a low-pass filter to brightness values for each property.

        Args:
            time_since_last_processed_frame (float): Time since last frame.
            left_eye_lux (float): Left eye brightness.
            right_eye_lux (float): Right eye brightness.
            agent_genotype_dict (dict): Genotype configuration.
        """
        # Low-pass filter brightness value for each property with its own tau
        for prop_class in [self.percentage_turns_class, self.percentage_left_class, self.turn_angle_class, self.bout_duration_class, self.displacement_class]:
            prop_name = prop_class.prop_name
            prop_genotype = agent_genotype_dict.get(prop_name, None)

            if not 'tau_lpf' in self.meta_popt_dict[prop_name]['median'].keys():
                # Low-pass filter not required for this property
                continue

            # Get previous low-pass filtered brightness values
            prev_lpf_left_eye = self.agent_state_variables['left_eye_brightness_lowpass'][prop_name]
            prev_lpf_right_eye = self.agent_state_variables['right_eye_brightness_lowpass'][prop_name]

            # Get tau for this property
            tau = self.meta_popt_dict[prop_name]['median']['tau_lpf']
            alpha = time_since_last_processed_frame / (tau + time_since_last_processed_frame)

            # Compute low-pass filtered brightness values
            lpf_left_eye = alpha * left_eye_lux + (1 - alpha) * prev_lpf_left_eye
            lpf_right_eye = alpha * right_eye_lux + (1 - alpha) * prev_lpf_right_eye

            # Store low-pass filtered brightness values
            self.agent_state_variables['left_eye_brightness_lowpass'][prop_name] = lpf_left_eye
            self.agent_state_variables['right_eye_brightness_lowpass'][prop_name] = lpf_right_eye

    def get_value(self, prop_class, par_name='median', direction=False):
        """Gets the value of a swim property based on the agent's genotype and current state.

        This method determines the value of a swim property (e.g., turn probability, turn angle, etc.)
        for the agent, using the appropriate model and parameters based on the agent's genotype and
        the current sensory state. For spatial-temporal genotypes, it delegates to the spatial-temporal
        model evaluation.

        Args:
            prop_class: The property class (e.g., PercentageTurns, TurnAngle).
            par_name (str, optional): Parameter name to use from the model (default: 'median').
            direction (bool, optional): If True, use signed contrast for contrast-based genotypes (default: False).

        Returns:
            float: The computed value for the swim property.

        Raises:
            ValueError: If the genotype or model type is unknown.
        """
        # Get property name
        prop_name = prop_class.prop_name

        # Get input value based on agent genotype
        prop_genotype = self.agent_genotype_dict[prop_name]

        # Split up genotype and multiplication factor if given
        factor = 1  # Default multiplication factor is 1
        if '_x' in prop_genotype:
            prop_genotype, factor_str = prop_genotype.split('_x')
            factor = float(factor_str)

        if isinstance(prop_genotype, int) or isinstance(prop_genotype, float):
            # prop_genotype is fixed value and already given
            return prop_genotype
        elif prop_genotype == 'blind':
            x = 150
            model_type = 'log'  # Use log model for blind
        elif prop_genotype == 'homogeneous' or prop_genotype == 'azimuth' or prop_genotype == 'azimuth_virtual':
            x = self.agent_state_variables['homogeneous_lux']
            model_type = 'log'  # Use log model for homogeneous
        elif prop_genotype == 'contrast':
            if direction:
                # Get the signed contrast value
                x = self.agent_state_variables['contrast_lux']
            else:
                x = self.agent_state_variables['abs_contrast_lux']
            model_type = 'lin'  # Use linear model for contrast
        elif prop_genotype == 'temporal':
            x = self.agent_state_variables['temporal_lux']
            model_type = 'double_linear'
        elif 'st' in prop_genotype:
            # We evaluate the spatial-temporal model and return the value
            return self.get_value_spatial_temporal(prop_name, prop_class.model)
        else:
            raise ValueError(f"Unknown genotype for {prop_name}: {prop_genotype}")

        if model_type == 'log':
            a_log = self.meta_popt_dict[prop_name][par_name]['a_log']
            b_log = self.meta_popt_dict[prop_name][par_name]['b_log']
            x = np.maximum(x, 1e-6)     # Add small offset to avoid log(0)
            return a_log + b_log * np.log(x)
        elif model_type == 'lin':
            p0 = self.meta_popt_dict[prop_name][par_name]['p0']
            p1 = self.meta_popt_dict[prop_name][par_name]['p1']
            return p0 + p1 * x
        elif model_type == 'double_linear':
            a_pos = self.meta_popt_dict[prop_name][par_name]['a_pos']
            a_neg = self.meta_popt_dict[prop_name][par_name]['a_neg']
            b = self.meta_popt_dict[prop_name][par_name]['b']
            return a_pos * x + b if x > 0 else a_neg * x + b

        raise ValueError(f"Unknown genotype for {prop_name}: {prop_genotype}")

    def get_value_spatial_temporal(self, prop_name, model, par_name='median'):
        """Evaluates the spatial-temporal model for a swim property.

        This method retrieves the spatial-temporal model parameters for the given property,
        evaluates the model using the agent's current and low-pass filtered eye brightness,
        and returns the resulting value.

        Args:
            prop_name (str): The property name (e.g., 'percentage_left').
            model: The spatial-temporal model object to evaluate.
            par_name (str, optional): Parameter name to use from the model (default: 'median').

        Returns:
            float: The computed value for the swim property from the spatial-temporal model.
        """
        # Get spatial-temporal model parameters
        par_meta_popt_dict = self.meta_popt_dict[prop_name][par_name]
        meta_popt = []
        for meta_par_name in model.par_names:
            meta_popt.append(par_meta_popt_dict[meta_par_name])

        # Evaluate model
        return model.eval_step(
            self.agent_state_variables['left_eye_brightness'],
            self.agent_state_variables['right_eye_brightness'],
            self.agent_state_variables['left_eye_brightness_lowpass'][prop_name],
            self.agent_state_variables['right_eye_brightness_lowpass'][prop_name],
            *meta_popt,
        )

    def get_bout_direction(self):
        """Determines the direction of the next bout (turn or straight).

        This method uses the agent's model to probabilistically decide whether the next
        movement is a straight swim or a turn. If a turn is chosen, it further decides
        whether the turn is to the left or right.

        Returns:
            int: 0 for straight, +1 for left turn, -1 for right turn.
        """
        # Turn or straight bout? Probability to turn ##########################
        p_turn = self.get_value(self.percentage_turns_class)
        # # Draw turn or straight based on p_turn
        p_draw = self.rng.uniform(0, 100)
        if p_draw > p_turn:
            # Straight
            bout_direction = 0
            return bout_direction
        else:
            # Turn
            # Probability of turning left
            p_left = self.get_value(self.percentage_left_class, direction=True)
            p_draw = self.rng.uniform(0, 100)
            # +1 for left turn and -1 for right turns
            bout_direction = 1 if p_draw < p_left else -1
            return bout_direction

    # #########################################################################
    # Simulation functions
    # #########################################################################
    def start_agent(self):
        """Initializes the agent's state variables and starting position.

        Sets the agent at a random position and orientation within the arena,
        and initializes all state variables required for simulation, including
        position, orientation, bout timing, brightness, and low-pass filtered values.
        """
        # Set initial conditions: random position and orientation #############
        ang_deg = self.rng.uniform(low=0, high=360)  # deg
        x, y = self.r_max*10, self.r_max*10  # start outside arena
        while np.sqrt(x**2 + y**2) >= self.r_max:  # Make sure fish starts inside r_max
            x = self.rng.uniform(low=-self.r_max, high=self.r_max)  # cm
            y = self.rng.uniform(low=-self.r_max, high=self.r_max)  # cm

        # Keep track of agent state variables #################################
        # # Coordinates
        self.agent_state_variables['current_orientation'] = ang_deg # deg
        self.agent_state_variables['current_x_position'] = x        # cm
        self.agent_state_variables['current_y_position'] = y        # cm
        # # Agent bout variables
        self.agent_state_variables['time_of_last_bout'] = 0         # s, simulation time
        self.agent_state_variables['bout_total_duration'] = -1      # s
        self.agent_state_variables['bout_total_displacement'] = 0       # cm
        self.agent_state_variables['bout_start_speed'] = 0          # cm/s
        # # Agent accumulated path and orientation
        self.agent_state_variables['accumulated_path'] = 0         # cm
        self.agent_state_variables['accumulated_orientation'] = 0  # deg
        # # Agent eye brightness. Start at 150 lux
        self.agent_state_variables['homogeneous_lux'] = 150  # lux
        self.agent_state_variables['temporal_lux'] = 150     # lux
        self.agent_state_variables['abs_contrast_lux'] = 0       # lux
        self.agent_state_variables['contrast_lux'] = 0       # lux
        self.agent_state_variables['left_eye_brightness'] = 150     # lux
        self.agent_state_variables['right_eye_brightness'] = 150    # lux
        self.agent_state_variables['last_bout_brightness'] = 150    # lux
        self.agent_state_variables['left_eye_brightness_lowpass'] = dict()   # lux
        self.agent_state_variables['right_eye_brightness_lowpass'] = dict()  # lux
        for prop_class in [self.percentage_turns_class, self.percentage_left_class, self.turn_angle_class, self.bout_duration_class, self.displacement_class]:
            prop_name = prop_class.prop_name
            self.agent_state_variables['left_eye_brightness_lowpass'][prop_name] = 150   # lux
            self.agent_state_variables['right_eye_brightness_lowpass'][prop_name] = 150  # lux

    def update_agent(self, current_time, left_eye_lux, right_eye_lux, time_since_last_processed_frame):
        """Updates the agent's position and interval variables.
        Computes tracking and bout data.

        Advances the agent's state by one simulation step, updating position,
        orientation, and internal variables based on the current sensory input
        and elapsed time. If a new bout occurs, returns both tracking and bout data.

        Args:
            current_time (float): Current simulation time (s).
            left_eye_lux (float): Left eye brightness (lux).
            right_eye_lux (float): Right eye brightness (lux).
            time_since_last_processed_frame (float): Time since last frame (s).

        Returns:
            tuple:
                tracking_data (list): Tracking data for the current frame.
                bout_data (list or bool): Bout data if a bout occurred, else False.
        """
        # Get agents' current state variables #################################
        # # Coordinates
        ang_deg = self.agent_state_variables['current_orientation']
        x = self.agent_state_variables['current_x_position']
        y = self.agent_state_variables['current_y_position']
        # # Agent state
        time_of_last_bout = self.agent_state_variables['time_of_last_bout']
        bout_duration = self.agent_state_variables['bout_total_duration']
        bout_displacement = self.agent_state_variables['bout_total_displacement']
        bout_start_speed = self.agent_state_variables['bout_start_speed']
        time_since_last_bout = current_time - time_of_last_bout
        # # Agent accumulated path and orientation
        accumulated_path = self.agent_state_variables['accumulated_path']
        accumulated_orientation = self.agent_state_variables['accumulated_orientation']

        # Low-pass filter brightness values ###################################
        self.low_pass_filter(time_since_last_processed_frame, left_eye_lux, right_eye_lux, self.agent_genotype_dict)
        # Extract values for storage in bout and tracking data
        lpf_left_eye = self.agent_state_variables['left_eye_brightness_lowpass'][self.percentage_left_class.prop_name]
        lpf_right_eye = self.agent_state_variables['right_eye_brightness_lowpass'][self.percentage_left_class.prop_name]

        # Agent decision making ###############################################
        bout_data = False  # Initialize bout data
        if time_since_last_bout > bout_duration:  # Start new bout
            # Process brightness values #######################################
            brightness_of_last_bout = self.agent_state_variables['last_bout_brightness']
            brightness_homogeneous = (left_eye_lux + right_eye_lux) / 2
            self.agent_state_variables['homogeneous_lux'] = brightness_homogeneous
            self.agent_state_variables['temporal_lux'] = (brightness_homogeneous - brightness_of_last_bout) / bout_duration  # lux/s
            self.agent_state_variables['abs_contrast_lux'], self.agent_state_variables['contrast_lux'] = self.calc_michelson_contrast(left_eye_lux, right_eye_lux)
            self.agent_state_variables['left_eye_brightness'] = left_eye_lux
            self.agent_state_variables['right_eye_brightness'] = right_eye_lux

            # Orientation #####################################################
            # Straight or turn?
            bout_direction = self.get_bout_direction()  # 0 for straight, +1 for left turn, -1 for right turn

            # Get orientation change
            if bout_direction == 0:
                # Straight
                d_ang = 0  # deg
            else:
                # Turn
                d_ang = self.get_value(self.turn_angle_class)  # deg
                # Multiply by direction, +1 for left turn and -1 for right turns
                d_ang *= bout_direction  # deg
            # # Add some noise to turn angle to avoid being stuck in loops
            d_ang += self.rng.normal(loc=0, scale=5)  # deg
            # # Update orientation
            ang_deg += d_ang    # deg
            
            # Displacement and duration #######################################
            bout_displacement = self.get_value(self.displacement_class)  # cm
            bout_duration = self.get_value(self.bout_duration_class)     # s
            # Compute start speed
            bout_start_speed = self.get_agent_start_speed(bout_displacement, bout_duration, self.speed_decay)  # cm/s

            # Update agent state variables ####################################
            time_since_last_bout = 0
            self.agent_state_variables['time_of_last_bout'] = current_time
            self.agent_state_variables['bout_total_duration'] = bout_duration
            self.agent_state_variables['bout_total_displacement'] = bout_displacement
            self.agent_state_variables['bout_start_speed'] = bout_start_speed
            self.agent_state_variables['last_bout_brightness'] = brightness_homogeneous

            # Store bout data
            contour_area = 0  # simulations don't have a contour area
            bout_data = [
                # Start of bout
                current_time, contour_area, x, y,
                accumulated_path, accumulated_orientation, 0,
                left_eye_lux, right_eye_lux, lpf_left_eye,
                # End of bout
                current_time + bout_duration, 0,
                x + bout_displacement * np.cos(np.deg2rad(ang_deg)),
                y + bout_displacement * np.sin(np.deg2rad(ang_deg)),
                accumulated_path + bout_displacement, accumulated_orientation + d_ang, 0,
                left_eye_lux, right_eye_lux, lpf_left_eye,
                # Bout properties
                bout_duration, bout_displacement * np.cos(np.deg2rad(ang_deg)), bout_displacement * np.sin(np.deg2rad(ang_deg)),
                bout_displacement, d_ang, bout_displacement / bout_duration, time_since_last_bout,
                0,
                current_time, current_time + bout_duration
            ]

        # Update agent speed and position #####################################
        current_speed = self.get_agent_speed(bout_start_speed, time_since_last_bout, self.speed_decay)  # cm/s
        x += current_speed * np.cos(np.deg2rad(ang_deg)) * time_since_last_processed_frame  # cm
        y += current_speed * np.sin(np.deg2rad(ang_deg)) * time_since_last_processed_frame  # cm

        accumulated_path += current_speed * time_since_last_processed_frame  # cm
        accumulated_orientation += ang_deg  # deg

        # Apply boundary conditions ###########################################
        x, y, ang_deg = self.perpendicular_boundary(
            self.r_max,
            x, y, ang_deg,
            # Add some noise to boundary turn angle and new radius
            self.boundary_turn_angle + self.rng.normal(loc=0, scale=self.boundary_turn_std),
            self.r_max * self.rng.uniform(0.9, 1)
        )

        # Store new agent state variables #####################################
        self.agent_state_variables['current_orientation'] = ang_deg
        self.agent_state_variables['current_x_position'] = x
        self.agent_state_variables['current_y_position'] = y
        self.agent_state_variables['accumulated_path'] = accumulated_path
        self.agent_state_variables['accumulated_orientation'] = accumulated_orientation
        self.agent_state_variables['left_eye_brightness'] = left_eye_lux
        self.agent_state_variables['right_eye_brightness'] = right_eye_lux
        # self.agent_state_variables['left_eye_brightness_lowpass'] = lpf_left_eye
        # self.agent_state_variables['right_eye_brightness_lowpass'] = lpf_right_eye

        # Compile tracking data
        tracking_data = [
            current_time, 0,
            x, y, accumulated_path,
            accumulated_orientation, 0,
            left_eye_lux, right_eye_lux, lpf_left_eye,
            current_time,
        ]
        return tracking_data, bout_data

    # #########################################################################
    # Simulation helper functions
    # #########################################################################
    # Speed ###################################################################
    @staticmethod
    @jit(nopython=True)
    def get_agent_start_speed(bout_displacement: float, event_duration: float, speed_decay: float):
        """Returns agent start speed to achieve bout_displacement in event_duration with speed_decay"""
        return bout_displacement / (speed_decay * (1 - np.exp(-1 * event_duration / speed_decay)))  # cm/s

    @staticmethod
    @jit(nopython=True)
    def get_agent_speed(bout_start_speed: float, time_since_last_bout: float, speed_decay: float):
        """returns agent speed as exponential decay of initial speed with time constant speed_decay"""
        return bout_start_speed * np.exp(-1 * time_since_last_bout / speed_decay)  # arena unit/s

    # Boundary conditions #####################################################
    @staticmethod
    @jit(nopython=True)
    def perpendicular_boundary(
            r_max,
            x, y, ang_deg,
            ang_deg_new, r_new
    ):
        """If the agent is too close to the wall, face away from the wall
        while maintaining the same general direction"""
        if np.sqrt(x ** 2 + y ** 2) <= r_max:
            # Agent is still inside the arena
            return x, y, ang_deg

        # Agent too close to the wall, face perpendicular to wall
        # Compute new orientation
        phi = np.arctan2(y, x)  # rad, position of agent
        # # Compute whether orientation is towards or away from phi
        theta = np.deg2rad(ang_deg)  # rad, orientation of agent
        a1 = np.cos(theta)
        a2 = np.sin(theta)
        b1 = np.cos(phi)
        b2 = np.sin(phi)
        sign = np.sign(a1 * b2 - a2 * b1)

        # Update orientation
        ang_deg_new = np.rad2deg(phi) - sign * ang_deg_new  # deg
        theta_new = np.deg2rad(ang_deg_new)

        # Update x and y coordinates: shift agent back along phi-line
        x_new = r_new * np.cos(phi)
        y_new = r_new * np.sin(phi)

        # if self.verbose:  # print for debugging
        #     print(f"arena {arena_index}: perpendicular_boundary(): -> {ang_deg: .2f} deg (at phi {np.rad2deg(phi): .2f} deg)")
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot([0, r_max * np.cos(phi)], [0, r_max * np.sin(phi)], 'k--')  # phi
        # # Current position and orientation
        # plt.plot(x, y, 'o', color='k')  # head
        # plt.plot([x, x - np.cos(theta)], [y, y - np.sin(theta)], color='k')  # tail
        # # New position + orientation
        # plt.plot(x_new, y_new, 'o', color='k')  # head
        # plt.plot([x_new, x_new - np.cos(theta_new)], [y_new, y_new - np.sin(theta_new)], color='tab:red')  # tail
        #
        # plt.xlim([-6, 6])
        # plt.ylim([-6, 6])
        # plt.gca().set_aspect('equal')
        # plt.show()

        return x_new, y_new, ang_deg_new

    def get_current_coordinates(self):
        """Returns the agent's current position and orientation.

        Returns:
            tuple: (x, y, orientation)
                x (float): Current x position (cm).
                y (float): Current y position (cm).
                orientation (float): Current orientation (deg).
        """
        return self.agent_state_variables['current_x_position'], self.agent_state_variables['current_y_position'], self.agent_state_variables['current_orientation']

    @staticmethod
    @jit(nopython=True)
    def calc_michelson_contrast(b_left, b_right):
        """Calculates the Michelson contrast between left and right eye brightness.

        Args:
            b_left (float): Left eye brightness.
            b_right (float): Right eye brightness.

        Returns:
            tuple:
                abs_contrast (float): Absolute Michelson contrast.
                signed_contrast (float): Signed Michelson contrast (positive if right is brighter).
        """
        b_diff = b_right - b_left  # Positive contrast: brighter on the right
        b_total = b_left + b_right

        if b_total == 0:
            # Avoid division by zero
            return 0, 0

        # Return absolute and signed contrast
        return abs(b_diff) / b_total, b_diff / b_total

    @staticmethod
    # @jit(nopython=True)
    def get_eye_brightness(
            stim_name, do_fish_lock, stim_array, frame_num,
            xs_arena, ys_arena, x_agent, y_agent, theta_agent, r_view
    ):
        """Computes the brightness perceived by each eye given stimulus and agent
        position.

        Calculates the left and right eye brightness based on the agent's position,
        orientation, and the stimulus configuration. Supports both fish-locked and
        arena-locked stimuli.

        Args:
            stim_name (str): Name of the stimulus.
            do_fish_lock (bool): If True, stimulus is fish-locked.
            stim_array (np.ndarray): Stimulus array.
            frame_num (int): Current frame number.
            xs_arena (np.ndarray): X-coordinates of arena grid.
            ys_arena (np.ndarray): Y-coordinates of arena grid.
            x_agent (float): Agent x position (cm).
            y_agent (float): Agent y position (cm).
            theta_agent (float): Agent orientation (deg).
            r_view (float): View radius (cm).

        Returns:
            tuple:
                left_brightness (float): Brightness perceived by the left eye.
                right_brightness (float): Brightness perceived by the right eye.
        """
        if do_fish_lock:
            # Fish-locked: values given for each time frame
            # # Assert that stim_array has shape (n_frames, 2)
            assert stim_array.shape[1] == 2, f"get_eye_brightness(): stim_array has shape {stim_array.shape}, expected (n_frames, 2)"
            # # We can directly access the brightness values for this frame
            left_brightness = stim_array[frame_num, 0]
            right_brightness = stim_array[frame_num, 1]
        else:
            # Arena-locked: values given in space
            # Translate to agent coordinates: agent at the origin (0, 0)
            x_grid = xs_arena - x_agent
            y_grid = ys_arena - y_agent
            # Compute distance to fish
            r_grid = np.sqrt(x_grid**2 + y_grid**2)

            if stim_name == 'azimuth_left_dark_right_bright_virtual_yes':
                min_index = np.unravel_index(np.argmin(r_grid), r_grid.shape)
                brightness = stim_array[min_index]
                # Ensure values are not nan
                if np.isnan(brightness):
                    brightness = 0
                return brightness, brightness

            # Exclude everything outside the view radius
            mask = r_grid > r_view
            stim_array_masked = stim_array.copy()
            stim_array_masked[mask] = np.nan  # requires stim_array to be an array of floats

            # Rotate coordinates to align the agent's orientation with the positive x-axis
            # # We use the negative of theta_agent to rotate the grid to match the agent's orientation
            y_grid_rot = x_grid * np.sin(np.deg2rad(-1 * theta_agent)) + y_grid * np.cos(np.deg2rad(-1 * theta_agent))

            # # Split values into left and right eye
            # Positive y-axis corresponds to the left side of the agent
            # Negative y-axis corresponds to the right side of the agent
            left_grid = stim_array_masked.copy()
            right_grid = stim_array_masked.copy()
            left_grid[y_grid_rot < 0] = np.nan   # exclude right side
            right_grid[y_grid_rot > 0] = np.nan  # exclude left side

            # Take the mean of the brightness values
            def nan_mean(values):
                """Calculate the mean of the values, ignoring NaNs and empty slices"""
                if np.isnan(values).all():
                    # Return 0 in all-nan cases (e.g. when the agent is outside the arena)
                    return 0
                else:
                    return np.nanmean(values)

            left_brightness = nan_mean(left_grid)
            right_brightness = nan_mean(right_grid)

            # # For debugging
            # x_agent_tail, y_agent_tail = x_agent + -1 * np.cos(np.deg2rad(ang_agent)), y_agent + -1 * np.sin(np.deg2rad(ang_agent))
            # fig, axes = plt.subplots(2, 2, sharex='row', sharey='row')
            # axes[0, 0].imshow(stim_array, cmap=CMAP_grey, vmin=0, vmax=300, extent=(-r_arena, r_arena, -r_arena, r_arena), origin='lower')
            # axes[0, 0].plot([x_agent_tail, x_agent], [y_agent_tail, y_agent], 'r-')
            # axes[0, 0].plot(x_agent, y_agent, 'ro')  # head
            # axes[0, 1].imshow(r_grid, cmap=CMAP_grey, vmin=0, vmax=None, extent=(-r_arena, r_arena, -r_arena, r_arena), origin='lower')
            # axes[0, 1].plot([x_agent_tail, x_agent], [y_agent_tail, y_agent], 'r-')
            # axes[0, 1].plot(x_agent, y_agent, 'ro')  # head
            # axes[1, 0].imshow(stim_array_masked, cmap=CMAP_grey, vmin=0, vmax=300, extent=(-r_arena, r_arena, -r_arena, r_arena), origin='lower')
            # axes[1, 1].imshow(left_grid, cmap='Blues_r', vmin=0, vmax=300, extent=(-r_arena, r_arena, -r_arena, r_arena), origin='lower')
            # axes[1, 1].imshow(right_grid, cmap='Reds_r', vmin=0, vmax=300, extent=(-r_arena, r_arena, -r_arena, r_arena), origin='lower')
        return left_brightness, right_brightness

