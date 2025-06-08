# Standard library imports
from datetime import datetime
from pathlib import Path

# Third party library imports
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd, mode


from settings.general_settings import turn_threshold
from utils.general_utils import get_median_df_time


# #############################################################################
# Stand alone functions
# #############################################################################
def set_new_stim_name(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
    """Set new stimulus name
    df : pd.DataFrame
    rename_dict : dict
        Example:
        rename_dict = {
            'no_stimulus': {**kwargs}
            'gratings_left': {**kwargs},
            'gratings_right': {'new_stim_name': 'gratings_left', **kwargs}
        }
    """

    stim_names = df.index.unique('stimulus_name')

    # Create map_dict for renaming
    map_dict = dict()  # Init empty dict to store mapping of old stimulus names to new ones
    for stim_name in stim_names:  # Iterate through stimulus names
        rename_dict_value = rename_dict.get(stim_name)  # retrieve value (dict with info) associated with current stim_name
        if rename_dict_value:  #
            map_dict[stim_name] = rename_dict_value.get('new_stim_name', stim_name)
        else:
            print(f"\tstim_name '{stim_name}' not in rename_dict")
            map_dict[stim_name] = stim_name

    # Apply map: overwrite stim_name
    # # Store index names and clear index
    index_names = df.index.names
    df.reset_index(inplace=True)
    # # Actually rename
    with pd.option_context('mode.chained_assignment', None):
        df['stimulus_name'] = df.stimulus_name.map(map_dict)
    # # Restore index names
    df.set_index(index_names, inplace=True)
    df.sort_index(inplace=True)

    return df


def get_flip_column_names(do_flip: str, data_type: str):
    # Columns to be flipped (orientation dependent)
    if data_type == 'tracking':
        flip_column_names = ['accumulated_orientation',]
        flip_column_names_yaxis = ['x_position', ]  # will be added to flip_column_names
    elif data_type == 'bout':
        flip_column_names = ['accumulated_orientation', 'estimated_orientation_change', ]
        flip_column_names_yaxis = ['x_position', ] # will be added to flip_column_names
    else:
        raise NotImplementedError(f'get_flip_column_names(): data_type {data_type}')

    if not do_flip:
        return
    elif do_flip is True:
        return flip_column_names
    elif do_flip == 'y-axis':
        return flip_column_names + flip_column_names_yaxis
    elif do_flip == 'halve y-axis':
        return flip_column_names + flip_column_names_yaxis
    elif do_flip == 'halve fish':
        return flip_column_names
    else:
        raise NotImplementedError(f'get_flip_column_names(): do_flip {do_flip}')


def split_data(trial_df, split_dict: dict, verbose=True):
    """Split event data into multiple subtrials.
    If stim_names_list or stim_query are given, only stimulus_names that correspond to this match are split, all
    others are deleted."""
    if not isinstance(split_dict, dict) or not len(split_dict):
        return trial_df

    print(f'\t{datetime.now():%H:%M:%S} Split data...')
    output_df_list = []

    for num, split_d in enumerate(split_dict.values()):
        # Extract values
        t_start = split_d['t_start']
        t_end = split_d['t_end']
        t_shift = split_d.get('t_shift', 0)
        stim_suffix = split_d.get('stim_suffix', '')
        new_stim_name = split_d.get('new_stim_name', '')
        stim_names_list = split_d.get('stim_names_list', None)
        stim_query = split_d.get('stim_query', None)

        # Select rows within t_start and t_end that match stim_names
        if stim_names_list:
            # Select rows with stim_names in stim_names list
            _df = trial_df.loc[stim_names_list].query('@t_start <= time < @t_end').copy()
        elif stim_query:
            # Select
            _df = (trial_df[trial_df
                   .index.get_level_values('stimulus_name').str.contains(stim_query, case=True)]
                   .query('@t_start <= time < @t_end').copy())
        else:
            _df = trial_df.query('@t_start <= time < @t_end').copy()

        if verbose:
            print(f"\t\t{t_start} <= t < {t_end}\n"
                  f"\t\t\tstimulus names: {_df.index.unique('stimulus_name').tolist()}\n"
                  f"\t\t\tt_shift: {t_shift}")

            if stim_suffix:
                print(f"\t\t\tstim_suffix: {stim_suffix}")
            elif new_stim_name:
                print(f"\t\t\tnew_stim_name: {new_stim_name}")
            else:
                print(f'\t\t\tstim_suffix: \033[93m!empty\033[0m')

        # Update timestamps
        _df['time'] += t_shift

        # Add offset to trial number to ensure unique trials after renaming
        _df.reset_index('trial', inplace=True)
        _df['trial'] += 1_000 * num
        _df.set_index('trial', append=True, inplace=True)

        # Rename stim names by adding stim_suffix (if given)
        if stim_suffix:
            unique_stim_names = _df.index.unique('stimulus_name')
            rename_dict = dict()
            for stim_name in unique_stim_names:
                rename_dict[stim_name] = {'new_stim_name': stim_name + stim_suffix}
            _df = set_new_stim_name(_df, rename_dict)
        elif new_stim_name:
            unique_stim_names = _df.index.unique('stimulus_name')
            rename_dict = dict()
            for stim_name in unique_stim_names:
                rename_dict[stim_name] = {'new_stim_name': new_stim_name}
            _df = set_new_stim_name(_df, rename_dict)

        # Concat dataframes
        output_df_list.append(_df)

    print(f'\t\033[92mdone\033[0m')
    return pd.concat(output_df_list)


def flip_data(trial_df, flip_dict: dict, data_type: str = 'bout', verbose=True):
    """Flip event data based on flip_dict
    flip_dict (dict): Dictionary describing which stimuli to flip and how
        Example:
        flip_dict = {
            'gratings_left': {'flip': False, 'new_stim_name': 'gratings_left', **kwargs},
            'gratings_right': {'flip': True, 'new_stim_name': 'gratings_left', **kwargs},
        }
    """
    if not isinstance(flip_dict, dict):
        return trial_df

    print(f'\t{datetime.now():%H:%M:%S} Flipping data...')

    # Reset all indices to avoid problems with non-unique multi-index
    index_names = trial_df.index.names
    _df = trial_df.reset_index()

    # Loop over flip dict
    for stim_name, stim_flip_dict in flip_dict.items():
        do_flip = stim_flip_dict.get('flip', False)
        flip_column_names = get_flip_column_names(do_flip, data_type)

        if not do_flip:
            continue  # Do nothing
        elif do_flip is True:
            print(f"\t\t\033[94mfish-axis\033[0m\t{stim_name}")
            # Flip data in flip_column_names for stim_name rows
            with pd.option_context('mode.chained_assignment', None):  # ignore SettingwithCopyWarning
                mask = _df['stimulus_name'] == stim_name
                _df.loc[mask, flip_column_names] *= -1  # Flip data in flip_column_names
                _df.loc[mask, 'trial'] += 1_000_000  # Add offset to trial number to ensure unique trials after renaming
        elif do_flip == 'y-axis':
            print(f"\t\t\033[94my-axis\033[0m\t{stim_name}")
            # Flip x-coordinates around y-axis
            with pd.option_context('mode.chained_assignment', None):  # ignore SettingwithCopyWarning
                mask = _df['stimulus_name'] == stim_name
                _df.loc[mask, flip_column_names] *= -1  # Flip data in flip_column_names
                _df.loc[mask, 'trial'] += 1_000_000  # Add offset to trial number to ensure unique trials after renaming
        elif do_flip == 'halve y-axis':
            print(f"\t\t\033[94my-axis (half)\033[0m\t{stim_name}")
            # Flip x-coordinates around y-axis for half of the trials
            with pd.option_context('mode.chained_assignment', None):  # ignore SettingwithCopyWarning
                # Flip all even trials in odd repeats and all odd trials in even repeats
                # This way, also when we have only one trial but multiple repeats, half of the trials will be flipped
                mask = (_df['stimulus_name'] == stim_name) * (
                        (_df['trial'] % 2 == 0) == (_df['experiment_repeat'] % 2 == 1)
                )
                _df.loc[mask, flip_column_names] *= -1  # Flip data in flip_column_names
                _df.loc[mask, 'trial'] += 1_000_000  # Add offset to trial number to ensure unique trials after renaming
        elif do_flip == 'halve fish':
            print(f"\t\t\033[94mfish-axis (half)\033[0m\t{stim_name}")
            with pd.option_context('mode.chained_assignment', None):  # ignore SettingwithCopyWarning
                # Flip all even trials in odd repeats and all odd trials in even repeats
                # This way, also when we have only one trial but multiple repeats, half of the trials will be flipped
                mask = (_df['stimulus_name'] == stim_name) * (
                        (_df['trial'] % 2 == 0) == (_df['experiment_repeat'] % 2 == 1)
                )
                _df.loc[mask, flip_column_names] *= -1  # Flip data in flip_column_names
                _df.loc[mask, 'trial'] += 1_000_000  # Add offset to trial number to ensure unique trials after renaming
        else:
            raise NotImplementedError(f'flip_dict[\'{stim_name}\'][\'flip\'] = {do_flip}')

    # Set index back to original values
    trial_df = _df.set_index(index_names)
    trial_df.sort_index(inplace=True)

    # Set new stim name
    # Flipped columns need to be renamed
    trial_df = set_new_stim_name(trial_df, flip_dict)

    return trial_df


def set_label_values(df: pd.DataFrame, label_dict: dict) -> pd.DataFrame:
    print(f'\t{datetime.now():%H:%M:%S} Set label values ', end='')
    if isinstance(label_dict, type(None)):
        print(f'not given')
    elif isinstance(label_dict, int):
        print(f'\033[94mfrom time\033[0m (bin size = {label_dict} s)')
        # Map time points to (integer) label values
        time_bin_size = label_dict  # seconds
        df['label_value'] = (df['time'] // time_bin_size).astype(int)
    elif isinstance(label_dict, dict):
        print(f'\033[94mfrom label_dict\033[0m')
        df['label_value'] = df.index.get_level_values('stimulus_name').map(label_dict)
    else:
        print(f'\033[91mnot implemented label_dict={label_dict}\033[0m')

    return df


# #############################################################################
# AnalysisStart Class
# #############################################################################
class AnalysisStart:
    """
    Class to prepare, process, and analyze event data from multiple trials and fish.

    This class provides methods to preprocess, transform, and analyze behavioral
    event data (swim bouts, tracking data) from experiments. It supports
    operations such as flipping and splitting data, computing derived properties,
    filtering, and aggregating statistics over trials and fish.

    Args:
        path_to_analysed_file (Path): Path to the output file where processed
            data will be stored.
        direction_threshold_angle (float): Threshold angle (in degrees) to
            distinguish left vs right events.
        max_radius (float): Maximum radius (in cm) for valid events; events
            outside this radius are excluded.
        ddof (int): Delta Degrees of Freedom for variance calculations. Used in
            standard deviation computations.
        do_mode (bool): Whether to compute the mode of each property over trials
            within fish.

    Attributes:
        path_to_analysed_file (Path): Path to the output file for processed data.
        direction_threshold_angle (float): Angle threshold for left/right event
            classification.
        max_radius (float): Maximum allowed radius for events.
        ddof (int): Delta Degrees of Freedom for variance calculations.
        do_mode (bool): Whether to compute mode statistics.
        midline_factor (float): Factor for estimating midline length from contour
            area.
        required_column_names (list): List of required column names for the data
            type.
        windowing_column_names (list): List of columns used for windowed
            statistics.
        required_index_names (list): List of required index names for the data.
        distance_column_names (list): List of columns related to distance/position.
        data_type (str): Type of data being processed ('bout' or 'tracking').
        experiment_type (str): Type of experiment ('free', 'embedded', or
            'fictive').
        key (str): Storage key for processed data.
        trial_df (pd.DataFrame): DataFrame containing processed trial data.

        - The class is designed to be flexible for different experiment and data
          types.
        - Methods support both 'bout' and 'tracking' data, with appropriate column
          requirements.
        - Data is typically processed in a pipeline: loading, transformation,
          and aggregation.
    """
    def __init__(
            self,
            path_to_analysed_file: Path,
            direction_threshold_angle: float = turn_threshold,  # degrees
            max_radius: float = 5,  # cm
            ddof: int = 0,
            do_mode: bool = False,
    ):
        """
        Initialize the AnalysisStart class for event data processing and analysis.

        Args:
            path_to_analysed_file (Path): Path to the output file for processed data.
            direction_threshold_angle (float, optional): Threshold angle (in degrees)
                to distinguish left vs right events. Defaults to turn_threshold.
            max_radius (float, optional): Maximum allowed radius (in cm) for valid
                events. Events outside this radius are excluded. Defaults to 5.
            ddof (int, optional): Delta Degrees of Freedom for variance calculations.
                Used in standard deviation computations. Defaults to 0.
            do_mode (bool, optional): Whether to compute the mode of each property
                over trials within fish. Defaults to False.

        Notes:
            - This constructor sets up analysis parameters and prepares internal
              variables for further data processing.
            - Most configuration for data transformation and aggregation is handled
              in subsequent methods.
        """
        # Store user input
        self.path_to_analysed_file = Path(path_to_analysed_file)
        # # Store user input for analysis settings
        self.direction_threshold_angle = direction_threshold_angle
        self.max_radius = max_radius
        self.ddof = ddof
        self.do_mode = do_mode

        # Estimate midline_length (cm) based on contour_area (px), using
        # midline_length = np.sqrt(contour_area / midline_factor)
        self.midline_factor = 2985

        print(
            f'Initialise AnalysisStart\n'
            f'\tpath_to_analysed_file: {self.path_to_analysed_file}'
        )

        # Equal for all data_types and experiment_types
        self.required_column_names = list  # will be set in prepare_data_type()
        self.windowing_column_names = list  # will be set in prepare_data_type()
        self.required_index_names = [
            'stimulus_name', 'trial',
            'fish_genotype', 'fish_age',
            'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
            'arena_index', 'setup_index', 'folder_name',
            'contour_area', 'midline_length',
        ]
        self.distance_column_names = [
            'x_position', 'y_position', 'accumulated_path', 'midline_length',
        ]

        # Initialise other variables
        self.data_type = str
        self.experiment_type = str
        self.key = str
        self.trial_df = pd.DataFrame

    # ####################################################
    # Trial data
    # ####################################################
    def get_trial_df(
            self,
            df: pd.DataFrame | object,
            flip_dict: dict = None, split_dict: dict = None, label_dict: dict = None,
            included_stim_names: list = None,
            data_type: str = 'bout',  # 'bout' or 'tracking'
            experiment_type: str = 'free',  # 'free', 'embedded', or 'fictive'
    ):
        """
        Process and prepare trial-level event data for analysis.

        This method applies data type and experiment type settings, splits and
        flips data as needed, sets label values, selects included stimuli,
        computes derived properties, resets accumulated data, computes
        response times, and handles wall interactions. The processed DataFrame
        is stored and returned.

        Args:
            df (pd.DataFrame or object): Input DataFrame containing event or
                tracking data.
            flip_dict (dict, optional): Dictionary specifying how to flip data
                for each stimulus. Defaults to None.
            split_dict (dict, optional): Dictionary specifying how to split data
                into subtrials. Defaults to None.
            label_dict (dict or int, optional): Mapping or bin size for label
                assignment. Defaults to None.
            included_stim_names (list, optional): List of stimulus names to
                include. Defaults to None.
            data_type (str, optional): Type of data ('bout' or 'tracking').
                Defaults to 'bout'.
            experiment_type (str, optional): Type of experiment ('free',
                'embedded', or 'fictive'). Defaults to 'free'.

        Returns:
            pd.DataFrame: Processed trial-level event data.
        """
        self.trial_df = df.copy()
        self.prepare_data_type(data_type, experiment_type)
        self.prepare_data()
        self.trial_df = split_data(self.trial_df, split_dict)
        self.trial_df = flip_data(self.trial_df, flip_dict, data_type=self.data_type)
        self.trial_df = set_label_values(self.trial_df, label_dict)
        self.select_stimuli(included_stim_names)
        self.compute_properties()
        self.reset_accumulated_event_data()
        self.compute_event_response_time()
        self.wall_interactions()
        self.store_data(self.path_to_analysed_file, df=self.trial_df, key=self.key + '_event')

        return self.trial_df

    def prepare_data_type(
        self,
        data_type: str,  # 'bout' or 'tracking'
        experiment_type: str,  # 'free', 'embedded', or 'fictive'
    ):
        """
        Set up data and experiment type specific settings.

        This method configures required columns, windowing columns, and storage
        keys based on the data type ('bout' or 'tracking') and experiment type.

        Args:
            data_type (str): Type of data ('bout' or 'tracking').
            experiment_type (str): Type of experiment ('free', 'embedded', or
                'fictive').

        Raises:
            AssertionError: If an invalid data_type or experiment_type is given.
            NotImplementedError: If 'embedded' experiment_type is selected.
        """
        self.data_type = data_type
        self.experiment_type = experiment_type

        print(
            f'\t{datetime.now():%H:%M:%S} Set data type\n'
            f'\t\tdata_type:         \033[94m{self.data_type}\033[0m\n'
            f'\t\texperiment_type:   \033[94m{self.experiment_type}\033[0m'
        )

        # Check user input
        assert data_type == 'bout' or data_type == 'tracking', \
            f'data_type must be\n' \
            f'\t"bout" to combine bout-level data, or\n' \
            f'\t"tracking" to combine tracking-level data.'
        assert experiment_type == 'free' or experiment_type == 'embedded' or experiment_type == 'fictive', \
            f'experiment_type must be\n' \
            f'\t"free" to combine freely_swimming data, or\n' \
            f'\t"embedded" to combine embedded tail-free data, or\n' \
            f'\t"fictive" to combine fictive data.'

        if experiment_type == 'embedded':
            raise NotImplementedError(f'experiment_type {experiment_type}')

        # Set data and experiment specific settings ###########################
        # Set storage key
        if self.data_type == 'bout':
            self.key = 'all_bout_data_pandas'
        elif self.data_type == 'tracking':
            self.key = 'all_freely_swimming_tracking_data_pandas'
        elif self.data_type == 'embedded':
            self.key = 'all_head_embedded_tracking_data_pandas'

        if self.data_type == 'bout':
            self.required_column_names = [
                # Time
                'time', 'time_absolute',
                # Orientation
                'estimated_orientation_change', 'accumulated_orientation', 'turn_angle',
                # Position
                'x_position', 'y_position', 'accumulated_path',
                # Additional information
                'additional_tracking_info0', 'additional_tracking_info1', 'additional_tracking_info2',
            ]
            # Columns from which to take the mode, mean and std when windowing
            self.windowing_column_names = sorted([
                # Time
                'total_duration', 'event_freq',
                'response_time',
                # Orientation --> directional statistics
                'estimated_orientation_change', 'accumulated_orientation',
                'estimated_orientation', 'turn_angle',
                # Position
                'total_distance', 'total_distance_per_body_length', 'accumulated_path',
                # Position / Time
                'average_speed',
                # Additional information
                'additional_tracking_info0', 'additional_tracking_info1', 'additional_tracking_info2',
            ])
        elif self.data_type == 'tracking':
            self.required_column_names = [
                # Time
                'time', 'time_absolute',
                # Orientation
                'accumulated_orientation',
                # Position
                'x_position', 'y_position', 'accumulated_path',
                # Additional information
                'additional_tracking_info0', 'additional_tracking_info1', 'additional_tracking_info2',
            ]
            # Columns from which to take the mode, mean and std when windowing
            self.windowing_column_names = sorted([
                # Time: not relevant for tracking data
                # Orientation --> directional statistics
                'accumulated_orientation',
                # Position
                'accumulated_path',
                # Additional information
                'additional_tracking_info0', 'additional_tracking_info1', 'additional_tracking_info2',
            ])

    # Modifying event data ###############################
    def prepare_data(self):
        """
        Prepare and standardize the data for analysis.

        This method resets index names, renames columns to standardized names,
        ensures required columns and indices are present, and sets the DataFrame
        to the correct multi-index and order.

        Raises:
            ValueError: If required columns or indices are missing after
                preparation.
        """
        print(f'\t{datetime.now():%H:%M:%S} Prepare data')

        # Reset index names to allow easier handling
        trial_df_reset = self.trial_df.reset_index(inplace=False)

        # Rename columns (if existing) to match tracking data columns
        trial_df_reset.rename(columns={
            # Time
            'start_time': 'time', 'start_time_absolute': 'time_absolute',
            # Position
            'start_x_position': 'x_position', 'start_y_position': 'y_position',
            'end_accumulated_path': 'accumulated_path',
            # Orientation
            'start_estimated_orientation_change': 'estimated_orientation_change',
            'end_accumulated_orientation': 'accumulated_orientation',
            # Contour area
            'end_contour_area': 'contour_area', 'fish_contour_area': 'contour_area',
            'end_midline_length': 'midline_length', 'bout_end_curvature': 'curvature',
            # Additional tracking info
            'start_additional_tracking_info0': 'additional_tracking_info0',
            'start_additional_tracking_info1': 'additional_tracking_info1',
            'start_additional_tracking_info2': 'additional_tracking_info2',
            # Rename indices to match updated names
            'fish_index': 'fish_or_agent_name'
        }, inplace=True)

        # Ensure fish age is an integer
        if self.trial_df.index.get_level_values('fish_age').dtype == 'O':
            trial_df_reset['fish_age'] = trial_df_reset['fish_age'].str.extract(r'(\d+)', expand=False).astype(int)

        # Set fish_genotype as lower-case
        trial_df_reset['fish_genotype'] = trial_df_reset['fish_genotype'].str.lower()

        # Only keep the indices and columns that are needed for now, and ensure order
        # Ensure all required columns are present
        for col in self.required_index_names + self.required_column_names:
            if col not in trial_df_reset.columns:
                print(f'\t{datetime.now():%H:%M:%S} Add column: {col}')
                trial_df_reset[col] = np.nan  # Add the column with default value None

        trial_df_reset = trial_df_reset[self.required_index_names + self.required_column_names]
        # Set multi-index and sort
        self.trial_df = trial_df_reset.set_index(self.required_index_names, drop=True)
        self.trial_df.sort_index(inplace=True)

        # Check if all required columns and indices are present
        existing_columns = self.trial_df.columns
        missing_columns = [c for c in self.required_column_names if c not in existing_columns]
        existing_indices = self.trial_df.index.names
        missing_indices = [c for c in self.required_index_names if c not in existing_indices]
        if missing_columns or missing_indices:
            raise ValueError(
                f'trial_df is missing the following columns:\n'
                f'\t{missing_columns}\n'
                f'and/or the following indices:\n'
                f'\t{missing_indices}'
            )

    def select_stimuli(self, included_stim_names=None):
        """
        Filter the trial data to include only specified stimulus names.

        Args:
            included_stim_names (list, optional): List of stimulus names to keep.
                If None, all stimuli are retained.
        """
        if included_stim_names:
            print(f'\t{datetime.now():%H:%M:%S} Only keep stimuli', *included_stim_names)
            stim_names_set = set(included_stim_names)
            self.trial_df = self.trial_df.loc[self.trial_df.index.get_level_values('stimulus_name').isin(stim_names_set)]

    def compute_properties(self, arena_radius: int = 6):
        """
        Compute derived properties and statistics for event data.

        This method converts positions to centimeters, estimates midline length,
        computes median contour area and midline length per fish, calculates
        distances, durations, speeds, orientation changes, and classifies events
        as left, right, or straight.

        Args:
            arena_radius (int, optional): Arena radius in centimeters for
                position scaling. Defaults to 6.
        """
        print(f'\t{datetime.now():%H:%M:%S} Compute properties')

        # Reset index names to allow easier handling
        trial_df_reset = self.trial_df.reset_index()

        # First convert arena_unit to cm
        trial_df_reset[self.distance_column_names] *= arena_radius

        # Estimate midline_length if not given
        trial_df_reset['midline_length'] = trial_df_reset['midline_length'].fillna(np.sqrt(trial_df_reset['contour_area'] / self.midline_factor))  # cm

        # Compute median contour_area and midline_length per fish
        trial_df_reset[['contour_area', 'midline_length']] = trial_df_reset.groupby([
            'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
            'fish_genotype', 'fish_age', 'folder_name',
        ])[['contour_area', 'midline_length']].transform('median')
        trial_df_reset['contour_area'].round(decimals=0)  # Round to nearest integer
        trial_df_reset['midline_length'].round(decimals=2)  # Round to 2 decimals

        # Recompute values between the start of each swim and the start of its consecutive swim
        if self.data_type == 'bout':
            # Sort values on time to ensure correct calculation of values when using diff()
            trial_df_reset.sort_values('time', inplace=True)

            # total distance change
            grouped = trial_df_reset.groupby([
                'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
                'arena_index', 'setup_index', 'folder_name',
                'fish_genotype', 'fish_age', 'stimulus_name', 'trial'
            ])
            dx = grouped['x_position'].diff()
            dy = grouped['y_position'].diff()
            trial_df_reset['total_distance'] = np.sqrt(dx ** 2 + dy ** 2)
            trial_df_reset['total_distance_per_body_length'] = trial_df_reset['total_distance'] / trial_df_reset['midline_length']
            # total duration between the start of each swim
            trial_df_reset['total_duration'] = grouped['time'].diff()
            trial_df_reset['event_freq'] = 1 / trial_df_reset['total_duration']  # Hz
            # Average speed (cm/s) based on total_distance and total_duration
            trial_df_reset['average_speed'] = trial_df_reset['total_distance'] / trial_df_reset['total_duration']

            # Compute orientation in arena coordinates
            trial_df_reset['estimated_orientation'] = np.arctan2(dy, dx)

            # Compute forward and sideward position change relative to last event
            trial_df_reset['forward_position_change'] = np.cos(np.deg2rad(trial_df_reset['estimated_orientation_change'])) * trial_df_reset['total_distance']
            trial_df_reset['sideward_position_change'] = np.sin(np.deg2rad(trial_df_reset['estimated_orientation_change'])) * trial_df_reset['total_distance']

            # Compute angle of turns
            trial_df_reset['estimated_orientation_change_abs'] = trial_df_reset['estimated_orientation_change'].abs()
            # Set all orientation changes straight swims (below 10 degrees) to NaN
            trial_df_reset['turn_angle'] = trial_df_reset['estimated_orientation_change_abs'].where(trial_df_reset['estimated_orientation_change_abs'] > self.direction_threshold_angle)

            # Detect left vs right events
            prop_name = 'estimated_orientation_change'
            left_events_mask = trial_df_reset[prop_name] > self.direction_threshold_angle
            right_events_mask = trial_df_reset[prop_name] < - self.direction_threshold_angle
            straight_events_mask = (abs(trial_df_reset[prop_name]) <= self.direction_threshold_angle)
            with pd.option_context('mode.chained_assignment', None):  # ignore SettingwithCopyWarning
                trial_df_reset.loc[left_events_mask, 'left_events'] = 1
                trial_df_reset.loc[right_events_mask, 'right_events'] = 1
                trial_df_reset.loc[straight_events_mask, 'straight_events'] = 1

        # Set multi-index and sort
        self.trial_df = trial_df_reset.set_index(self.required_index_names, drop=True)
        self.trial_df.sort_index(inplace=True)

    def reset_accumulated_event_data(self):
        """Values are set relative to the row closest before time is zero"""
        print(f'\t{datetime.now():%H:%M:%S} Reset accumulated values')

        # To find values corresponding to the row where the time is closest to AND before zero
        #   we first define _time as only the negative values of time
        self.trial_df['_time'] = self.trial_df['time']
        self.trial_df.loc[self.trial_df['time'] >= 0, '_time'] = np.nan
        #   then descending-sort the dataframe based on _time
        self.trial_df.sort_values('_time', inplace=True, ascending=False)
        #   then take the first value in each group as the reference value and subtract this from all values in that group
        self.trial_df['accumulated_orientation'] -= \
        self.trial_df.groupby(['stimulus_name', 'trial', 'experiment_ID', 'fish_or_agent_name', 'experiment_repeat', 'fish_genotype', 'fish_age', ])[
            'accumulated_orientation'].transform('first')
        if 'start_accumulated_orientation' in self.trial_df.columns:
            # Subtract same 'accumulated_orientation' reference value
            self.trial_df['start_accumulated_orientation'] -= \
            self.trial_df.groupby(['stimulus_name', 'trial', 'experiment_ID', 'fish_or_agent_name', 'experiment_repeat', 'fish_genotype', 'fish_age', ])[
                'accumulated_orientation'].transform('first')
        self.trial_df['accumulated_path'] -= \
        self.trial_df.groupby(['stimulus_name', 'trial', 'experiment_ID', 'fish_or_agent_name', 'experiment_repeat', 'fish_genotype', 'fish_age', ])[
            'accumulated_path'].transform('first')
        #   finally, we sort the dataframe based on time again and sort the index.
        self.trial_df.sort_values('time', inplace=True)

        # Clean up dataframe
        self.trial_df.drop(columns='_time', inplace=True)
        self.trial_df.sort_index(inplace=True)

    def compute_event_response_time(self):
        """Compute response time: Find first event after transition"""
        print(f'\t{datetime.now():%H:%M:%S} Compute response time')
        if self.data_type != 'bout':
            return

        # Define a '_time' column keeping only the positive entries (happening after the transition at t = 0)
        self.trial_df['_time'] = self.trial_df['time']
        self.trial_df.loc[self.trial_df['time'] <= 0, '_time'] = np.nan
        # Find the first response for each trial
        self.trial_df['_response_time'] = self.trial_df.groupby(self.trial_df.index.names)['_time'].transform('min')

        # Store the first response only for each first response event, set all other events to nan
        self.trial_df['response_time'] = np.nan
        self.trial_df.loc[self.trial_df['_time'] == self.trial_df['_response_time'], 'response_time'] =\
            self.trial_df.loc[self.trial_df['_time'] == self.trial_df['_response_time'], '_time']

        # Clean up DataFrame
        self.trial_df.drop(columns=['_time', '_response_time'], inplace=True)

    def wall_interactions(self):
        """Exclude events that happen too close to the wall (Calovi et al. 2018), and
        valid events are assigned a sequence number (Dunn et al. 2016)"""
        print(f'\t{datetime.now():%H:%M:%S} Wall interactions')

        # Compute radius
        self.trial_df['radius'] = np.sqrt(self.trial_df['x_position'] ** 2 + self.trial_df['y_position'] ** 2)

        grouped = self.trial_df.groupby(['stimulus_name', 'experiment_ID', 'fish_or_agent_name', 'experiment_repeat', 'fish_genotype', 'fish_age', 'trial'])
        group_list = []
        for group_ID, group in grouped:
            # Get rows where fish enters area too close to the border
            group['inside'] = group['radius'] < self.max_radius
            group['entering_boundary'] = 1 * group['inside'].diff()  # -1: leaving, +1: entering, 0: no change

            # Start each new sequence only when fish enter
            mask = (group['entering_boundary'] == +1)
            group.loc[~mask, 'entering_boundary'] = 0

            # Take the cumsum of 'entering_edge' (which is +1) to get a unique ID for each sequence, within each trial
            group.loc[:, 'sequence'] = group['entering_boundary'].cumsum().astype(int)

            # Combine new_group and concat to create new event_df
            group_list.append(group)
        self.trial_df = pd.concat(group_list)

        # Remove working columns
        self.trial_df.drop(columns=['inside', 'entering_boundary'], inplace=True)

    # ####################################################
    # Statistics over trials and over fish
    # ####################################################
    def get_median_df(
            self,
            trial_df: pd.DataFrame,
            resampling_window: pd.Timedelta,
            data_type: str = 'bout',  # 'bout' or 'tracking'
            experiment_type: str = 'free',  # 'free', 'embedded', or 'fictive'
            groupby_labels: list = None,
    ) -> pd.DataFrame:
        """
        Compute the median of event data over time windows for each group.

        This method groups the input DataFrame by the specified labels and computes
        the median for each resampling window. Events too close to the wall are
        excluded. The result is optionally stored to disk.

        Args:
            trial_df (pd.DataFrame): Input DataFrame containing event or tracking data.
            resampling_window (pd.Timedelta): Time window for resampling and median
                calculation.
            data_type (str, optional): Type of data ('bout' or 'tracking').
                Defaults to 'bout'.
            experiment_type (str, optional): Type of experiment ('free', 'embedded',
                or 'fictive'). Defaults to 'free'.
            groupby_labels (list, optional): List of columns to group by. If None,
                a default set of grouping labels is used.

        Returns:
            pd.DataFrame: DataFrame containing the median values for each group and
                time window.
        """
        if isinstance(groupby_labels, type(None)):
            groupby_labels = [
                'stimulus_name',
                'fish_genotype', 'fish_age',
                'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
                'arena_index', 'setup_index', 'folder_name',
            ]

        # Prepare datatype (setting correct column names)
        self.prepare_data_type(data_type, experiment_type)

        # Create some local copies
        _df = trial_df.copy()

        # Remove swims that are too close to the wall
        _df = _df[_df['radius'] <= self.max_radius]

        # Compute median_df
        median_df = get_median_df_time(_df, resampling_window, groupby_labels)

        # Store median_df #####################################################
        if not self.path_to_analysed_file == Path(''):
            self.store_data(self.path_to_analysed_file, df=median_df, key=self.key + '_median')
        return median_df

    def roll_over_trials(
            self,
            trial_df: pd.DataFrame,
            rolling_window: pd.Timedelta,
            resampling_window: pd.Timedelta,
            data_type: str = 'bout',  # 'bout' or 'tracking'
            experiment_type: str = 'free',  # 'free', 'embedded', or 'fictive'
            float_precision: int = 3,
            groupby_labels: list = None,
            min_periods: float = 5,
    ) -> pd.DataFrame:
        """
        Compute rolling statistics of event data over trials within each fish.

        This method applies a rolling window to the grouped event data, computing
        median, mean, std, and optionally mode for each window. It also calculates
        event frequencies and percentages of left/right/straight events. The result
        is resampled and optionally stored to disk.

        Args:
            trial_df (pd.DataFrame): Input DataFrame containing event or tracking data.
            rolling_window (pd.Timedelta): Size of the rolling window for statistics.
            resampling_window (pd.Timedelta): Time window for resampling after rolling.
            data_type (str, optional): Type of data ('bout' or 'tracking').
                Defaults to 'bout'.
            experiment_type (str, optional): Type of experiment ('free', 'embedded',
                or 'fictive'). Defaults to 'free'.
            float_precision (int, optional): Decimal precision for rounding windowed
                columns before computing mode. Defaults to 3.
            groupby_labels (list, optional): List of columns to group by. If None,
                a default set of grouping labels is used.
            min_periods (float, optional): Minimum number of observations in a window
                required to compute a value. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame containing rolling statistics for each group and
                time window.
        """
        if isinstance(groupby_labels, type(None)):
            groupby_labels = [
                'stimulus_name',
                'fish_genotype', 'fish_age',
                'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
                'arena_index', 'setup_index', 'folder_name',
            ]

        # Prepare datatype (setting correct column names)
        self.prepare_data_type(data_type, experiment_type)

        # Create some local copies
        _df = trial_df.copy()

        # Remove swims that are too close to the wall
        _df = _df[_df['radius'] <= self.max_radius]

        # Prepare rolling
        # reset_index to allow to count the number of trials after groupby
        _df.reset_index(['trial', 'fish_age'], inplace=True)
        # Map index fish_age 26 to 27 to keep as one age category in groupby
        _df['fish_age'] = _df['fish_age'].replace(26, 27)
        # important! rolling requires time_datetime to be monotonic
        _df['time_datetime'] = pd.to_datetime(_df['time'], unit='s')
        _df.sort_values(by='time_datetime', inplace=True)
        # Lower float precision in order to compute mode (this might also affect mean and std)
        _df[self.windowing_column_names] = _df[self.windowing_column_names].round(float_precision)

        # Actual rolling ######################################################
        print(f"\t{datetime.now():%H:%M:%S} Roll over trials with grouping on:", *groupby_labels, "...", end='')
        grouped = _df.groupby(groupby_labels)
        rolling = grouped.rolling(
            rolling_window, on='time_datetime', min_periods=min_periods,
            center=False,  # window labels are set as right edge of the window index
        )

        # Compute median
        rolled_df = rolling[self.windowing_column_names].median()

        # Compute mode, mean and std of groups, excluding missing values.
        #   All within fish
        if self.do_mode:
            print("\t\tcomputing mode...", end='')
            rolled_df[[c + '_mode' for c in self.windowing_column_names]] = rolling[self.windowing_column_names].apply(
                lambda x: mode(x, keepdims=True)[0])
            print(' \033[92mdone\033[0m')
        rolled_df[[c + '_mean' for c in self.windowing_column_names]] = rolling[self.windowing_column_names].mean()
        rolled_df[[c + '_std' for c in self.windowing_column_names]] = rolling[self.windowing_column_names].std(ddof=self.ddof)
        if 'estimated_orientation_change' in self.windowing_column_names:
            # Calculate circular mean and std for estimated_orientation_change
            rolled_df['estimated_orientation_change_mean'] = rolling['estimated_orientation_change'].apply(self._circmean)
            rolled_df['estimated_orientation_change_std'] = rolling['estimated_orientation_change'].apply(self._circstd)

        # Overwrite event_freq using number of events within a window
        rolled_df['number_of_events'] = rolling['time'].count()  # Calculate the rolling count of non NaN observations.
        number_of_trials = grouped['trial'].nunique()
        rolled_df['event_freq'] = rolled_df['number_of_events'] / (number_of_trials * pd.to_timedelta(rolling_window).total_seconds())

        # Sum events
        rolled_df['left_events'] = rolling['left_events'].sum()
        rolled_df['right_events'] = rolling['right_events'].sum()
        rolled_df['straight_events'] = rolling['straight_events'].sum()
        # Add percentages
        rolled_df['percentage_turns'] = (rolled_df['left_events'] + rolled_df['right_events']) / rolled_df['number_of_events'] * 100
        rolled_df['percentage_left'] = rolled_df['left_events'] / (rolled_df['left_events'] + rolled_df['right_events']) * 100
        rolled_df['percentage_right'] = rolled_df['right_events'] / (rolled_df['left_events'] + rolled_df['right_events']) * 100

        # Preserve time of each event
        rolled_df['time'] = rolling['time'].max()

        print('\033[92mdone\033[0m')

        # Resampling ##########################################################
        # When taking the mean over fish we want to weigh data equally with fish
        # instead by occurrence. Therefore, we resample over trials within fish.
        # Prepare resampling: set time_datetime as first index
        print(f"\t{datetime.now():%H:%M:%S} Resample over trials within fish ...", end='')
        rolled_df.reset_index(inplace=True)
        rolled_df.set_index(['time_datetime'] + groupby_labels, inplace=True)
        # Resample using groupby
        resampled_df = rolled_df.groupby([pd.Grouper(freq=resampling_window, level=0)] + groupby_labels).first()
        # Set time as start of each resampling bin
        resampled_df.reset_index('time_datetime', inplace=True)
        resampled_df['time'] = resampled_df['time_datetime'].astype('int64') / int(1e9)
        print(' \033[92mdone\033[0m')

        fish_df = resampled_df.sort_index()

        # Store fish_df #######################################################
        if not self.path_to_analysed_file == Path(''):
            self.store_data(self.path_to_analysed_file, df=fish_df, key=self.key + '_rolled')
        return fish_df

    # #########################################################################
    # Helper functions
    # #########################################################################
    def _circmean(self, samples):
        """Compute circular mean if experiment_type is 'free'"""
        if self.experiment_type == 'free':
            return np.rad2deg(circmean(
                np.deg2rad(samples),
                high=np.pi, low=-np.pi,
                nan_policy='omit',
            ))
        elif self.experiment_type == 'fictive':
            return np.nanmean(samples)
        elif self.experiment_type == 'embedded':
            return np.nanmean(samples)
        else:
            raise NotImplementedError(f'experiment_type {self.experiment_type} not implemented')

    def _circstd(self, samples):
        """Compute circular std if experiment_type is 'free'"""
        if self.experiment_type == 'free':
            return np.rad2deg(circstd(
                np.deg2rad(samples),
                high=np.pi, low=-np.pi,
                nan_policy='propagate',
            ))
        elif self.experiment_type == 'fictive':
            return np.nanstd(samples)
        elif self.experiment_type == 'embedded':
            return np.nanstd(samples)
        else:
            raise NotImplementedError(f'experiment_type {self.experiment_type} not implemented')

    # ####################################################
    # Loading and storing DataFrames
    # ####################################################
    @staticmethod
    def read_df(path_to_df: Path, key: str = None) -> pd.DataFrame | object:
        """ Convenience function to read dataframe from hdf5 file
        Args:
            path_to_df (Path): path to hdf5 file
            key (str): The group identifier in the store. Can be omitted if the HDF file contains a single pandas object.

        Returns:
            df (pd.DataFrame): DataFrame stored in hdf5 file or
            None if path_to_df or key does not exist
        """
        print(f"\t{datetime.now():%H:%M:%S} Loading data with key {key}... ", end='')
        if not path_to_df.exists():
            print(f"\t\033[91mfile does not exist\033[0m")
            return
        try:
            df = pd.read_hdf(path_to_df, key=key)
        except KeyError:
            print(f"\t\033[91mkey does not exist\033[0m")
            return
        print("\033[92mdone\033[0m")
        return df

    @staticmethod
    def store_data(path_to_output_file: Path, df: pd.DataFrame, key: str, complevel: int = 9):
        """ Convenience function to store dataframe to a hdf5 file
        Args:
            path_to_output_file (Path): path to store the hdf5 file
            df (pd.Dataframe): dataframe to store
            key (str): identifier for the group in the store.
            complevel (int): Specifies a compression level for data. A value of 0 or None disables compression.
                Default to 9 (maximum compression)
        """
        print(f"\t{datetime.now():%H:%M:%S} Storing data with key \033[94m{key}\033[0m... ", end='')

        df.to_hdf(
            path_or_buf=path_to_output_file, key=key,
            mode='a',
            complevel=complevel
        )
        print("\033[92mdone\033[0m")

    def return_df(self):
        """Convenience function to return event DataFrame in class"""
        return self.trial_df


