"""Class to perform prescreening of multifish behaviour data."""
# Standard library imports
import csv
from pathlib import Path

# Third party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local library imports
from .combine_data import CombineData
from .sanity_check import SanityCheck
from .analysis_start import AnalysisStart


class Prescreening:
    """Class to include or exclude experiment based on fish behaviour data."""

    def __init__(
            self,
            path_to_experiment_folders,
            path_to_analysed_folder,
            min_percentage_grating_left=.50,
            min_percentage_no_stimulus=.20,
            max_percentage_no_stimulus=.80,
            max_IBI_converging_shader=10,
            # min_IBI_grating_back=0.2,
            store_csv=False,
    ):
        """
        Initialize the Prescreening class for fish behaviour data.

        Args:
            path_to_experiment_folders (str or Path): Path to experiment folders.
            path_to_analysed_folder (str or Path): Path to analysed data folder.
            min_percentage_grating_left (float, optional): Minimum percentage of
                bouts directed left for leftward moving grating. Defaults to 0.50.
            min_percentage_no_stimulus (float, optional): Minimum percentage of
                bouts directed left when no stimulus is present. Defaults to 0.20.
            max_percentage_no_stimulus (float, optional): Maximum percentage of
                bouts directed left when no stimulus is present. Defaults to 0.80.
            max_IBI_converging_shader (float, optional): Maximum interbout
                interval for converging shader. Defaults to 10.
            store_csv (bool, optional): Whether to store prescreening results to
                a CSV file. Defaults to False.
        """
        # Set prescreening properties
        self.min_percentage_grating_left = min_percentage_grating_left
        self.min_percentage_no_stimulus = min_percentage_no_stimulus
        self.max_percentage_no_stimulus = max_percentage_no_stimulus
        self.max_IBI_converging_shader = max_IBI_converging_shader
        # self.min_IBI_grating_back = min_IBI_grating_back

        # Set other variables
        self.store_csv = store_csv

        # Set paths
        self.path_to_experiment_folders = Path(path_to_experiment_folders)
        self.path_to_analysed_folder = Path(path_to_analysed_folder)
        self.path_to_combined_data_file = self.path_to_analysed_folder.joinpath('combined_data_prescreening.hdf5')
        self.path_to_local_csv = self.path_to_analysed_folder.joinpath('prescreening.csv')

        # Preallocate variables
        self.binned_data = pd.DataFrame()
        self.log_str = str()
        self.csv_data = list()

        self.adaptation_duration = 20
        self.flip_dict = {
            'converging_shader': {'new_stim_name': 'converging_shader', 'flip': False},
            'grating_right': {'new_stim_name': 'grating_left', 'flip': True},
            'grating_left': {'new_stim_name': 'grating_left', 'flip': False},
            # 'grating_back': {'new_stim_name': 'grating_back', 'flip': False},
            # 'grating_front': {'new_stim_name': 'grating_front', 'flip': False},
            'no_stimulus': {'new_stim_name': 'no_stimulus', 'flip': False},
        }

    def run(self, **kwargs):
        """
        Run the prescreening pipeline on fish behaviour data.

        This method combines data, performs a sanity check, analyzes event data,
        computes median statistics, and applies prescreening criteria to include
        or exclude experiments.

        Args:
            **kwargs: Optional keyword arguments to update class attributes.

        Returns:
            tuple: (included_experiment_IDs, excluded_experiment_IDs)
        """
        print("\033[1mStart Prescreening\033[0m")

        # Update class attributes if given
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Combine Data
        prescreening_cd = CombineData(
            path_to_experiment_folders=self.path_to_experiment_folders,
            path_to_combined_data_file=self.path_to_combined_data_file,
            combine_bout_data=True,
            combine_tracking_data=False,
            prescreening=True,
            excluded_data_file_names=[],
            excluded_experiment_IDs=[],
            overwrite_output_file=False,
        )
        combined_event_df = prescreening_cd.run()

        # Sanity check
        prescreening_sc = SanityCheck(
            path_to_combined_data_file=self.path_to_combined_data_file,
            plot_statistics=False,
        )
        sanity_prescreening_df = prescreening_sc.run()

        # Analyse event data
        analysis_start = AnalysisStart(
            path_to_analysed_file=self.path_to_analysed_folder.joinpath('analysed_prescreening.hdf5'),
        )
        # Create event_df
        trial_event_df = analysis_start.get_trial_df(
            sanity_prescreening_df,
            flip_dict=self.flip_dict,
            split_dict=None,
            label_dict=None,
            data_type='bout',
        )

        median_df = analysis_start.get_median_df(
            trial_df=trial_event_df,
            resampling_window=pd.Timedelta(0.5, unit='s'),
        )

        self.run_prescreening(median_df)

        return self.included_experiment_IDs, self.excluded_experiment_IDs

    def run_prescreening(self, binned_data=None, **kwargs):
        """
        Apply prescreening criteria to experiments and update inclusion lists.

        This method checks each experiment for various stimulus conditions and
        applies thresholds to determine inclusion or exclusion. Optionally,
        results are stored in a CSV file.

        Args:
            binned_data (pd.DataFrame, optional): Binned event data for
                prescreening. If None, data is loaded from file.
            **kwargs: Optional keyword arguments to update class attributes.

        Generates:
            included_experiment_IDs, excluded_experiment_IDs
        """
        self.__dict__.update(kwargs)

        if self.store_csv and not self.path_to_local_csv.exists():
            header = [
                'experiment_ID',
                'converging_value',
                'grating_left_value',
                # 'grating_back_value',
                'no_stimulus_value',
            ]
            with open(self.path_to_local_csv, 'w+') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)

        # Get input data
        if isinstance(binned_data, type(pd.DataFrame())):
            self.binned_data = binned_data
        elif self.path_to_combined_data_file.exists():
            self.binned_data = pd.read_hdf(self.path_to_combined_data_file, key='analysed')
        else:
            raise UserWarning(f'Prescreening input data cannot be found at {self.path_to_combined_data_file.name}')

        if self.binned_data.empty:
            print(f"\033[93m!\tNo input data found for {self.path_to_combined_data_file.name}\033[0m")
            return None, None

        # Prepare experiment IDs
        self.all_experiment_ID = self.binned_data.index.unique('experiment_ID')
        self.excluded_experiment_IDs = []
        self.included_experiment_IDs = []

        # Loop over all experiment ID and perform prescreening
        for experiment_ID in tqdm(self.all_experiment_ID, desc='\tPrescreening'):
            # Extract experiment df
            experiment_df = self.binned_data.xs(experiment_ID, level='experiment_ID')

            # Set values to nan
            self.converging_value = np.nan
            self.grating_left_value = np.nan
            # self.grating_back_value = np.nan
            self.no_stimulus_value = np.nan

            # Check values
            self.check_converging_shader(experiment_ID, experiment_df)
            self.check_grating_left(experiment_ID, experiment_df)
            # self.check_grating_back(experiment_ID, experiment_df)
            self.check_no_stimulus(experiment_ID, experiment_df)

            # Include experiment
            self.include_experiment(experiment_ID)

            # Store to csv
            if self.store_csv:
                self.csv_data = [
                    experiment_ID,
                    self.converging_value,
                    self.grating_left_value,
                    # self.grating_back_value,
                    self.no_stimulus_value,
                ]

                with open(self.path_to_local_csv, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(self.csv_data)

        if len(self.log_str):
            print(f'\tPrescreening result:\n{self.log_str[:-1]}')

        print(f'\t\033[96mPercentage included:\033[0m {len(self.included_experiment_IDs) / (len(self.included_experiment_IDs) + len(self.excluded_experiment_IDs)) * 100:3.1f} % '
              f'({len(self.included_experiment_IDs)} of {(len(self.included_experiment_IDs) + len(self.excluded_experiment_IDs))})')

    def check_converging_shader(self, experiment_ID, df):
        """Prescreen experiment for converging shader."""

        if self.check_stimulus_name(experiment_ID, df, 'converging_shader'):
            value = df.xs('converging_shader', level='stimulus_name')['interbout_interval'].mean()
            if value > self.max_IBI_converging_shader:
                self.exclude_experiment(experiment_ID, 'converging shader', value)

            # Store data for csv file
            self.converging_value = value

    def check_grating_left(self, experiment_ID, df):
        """Prescreen experiment for leftward moving grating"""

        if self.check_stimulus_name(experiment_ID, df, 'grating_left'):
            value = df.xs('grating_left', level='stimulus_name')['percentage_left'].mean()
            if self.min_percentage_grating_left > value:
                self.exclude_experiment(experiment_ID, 'grating left', value)

            # Store data for csv file
            self.grating_left_value = value

    def check_grating_back(self, experiment_ID, df):
        """Prescreen experiment for backwards moving grating"""

        if self.check_stimulus_name(experiment_ID, df, 'grating_back'):
            value = df.xs('grating_back', level='stimulus_name')['interbout_interval'].mean()
            if self.min_IBI_grating_back > value:
                self.exclude_experiment(experiment_ID, 'grating back', value)

            # Store data for csv file
            self.grating_back_value = value

    def check_no_stimulus(self, experiment_ID, df):
        """Prescreen experiment for no stimulus present"""

        if self.check_stimulus_name(experiment_ID, df, 'no_stimulus'):
            value = df.xs('no_stimulus', level='stimulus_name')['percentage_left'].mean()
            if np.isnan(value).any:
                pass
                # self.log_str += f"   experiment_ID {experiment_ID} no stimulus contains nan values, ignored for now\n"

            elif not self.min_percentage_no_stimulus < value < self.max_percentage_no_stimulus:
                self.exclude_experiment(experiment_ID, 'no stimulus', value)

            # Store data for csv file
            self.no_stimulus_value = value

    def check_stimulus_name(self, experiment_ID, df, stim_name):
        """Check if stimulus name can be found"""
        if stim_name not in df.index.unique('stimulus_name'):
            # self.log_str += f"   experiment_ID {experiment_ID} {stim_name} cannot be found, ignored for now\n"
            # self.exclude_experiment(experiment_ID, stim_name, 'not found')
            return False

        return True

    def exclude_experiment(self, experiment_ID, stim_name, value):
        """Add experiment to list of excluded experiment."""
        if experiment_ID not in self.excluded_experiment_IDs:
            self.excluded_experiment_IDs.append(experiment_ID)

        self.log_str += f"\texperiment_ID{int(experiment_ID): 4.0f} excluded: {stim_name} {value:3.2f}\n"

    def include_experiment(self, experiment_ID):
        """All experiment that are not excluded, are included."""
        if experiment_ID in self.excluded_experiment_IDs:
            return  # This experiment is excluded based on earlier checks. I am sorry experiment.

        self.included_experiment_IDs.append(experiment_ID)