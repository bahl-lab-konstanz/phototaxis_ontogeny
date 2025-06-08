# Standard library imports
from datetime import datetime
from pathlib import Path

# Third party library imports
import numpy as np
import pandas as pd
from scipy import stats
import pylab as plt


class SanityCheck:
    def __init__(
        self,
        path_to_combined_data_file,
        max_interbout_interval=10,  # Set to None if this should be ignored
        max_contour_area=2000,  # Set to None if this should be ignored
        max_average_speed=1,  # Set to None if this should be ignored
        max_abs_orientation_change=150,  # Set to None if this should be ignored
        max_radius=5/6,  # Set to None if this should be ignored
        drop_trial_if_fraction_of_interbout_interval=0.05,  # Set to None if trial dropping should be switched off
        drop_trial_if_fraction_of_false_contour_area=0.05,  # Set to None if trial dropping should be switched off
        drop_trial_if_fraction_of_false_average_speed=0.05,  # Set to None if trial dropping should be switched off
        drop_trial_if_fraction_of_abs_orientation_change=0.05,  # Set to None if trial dropping should be switched off
        drop_trial_if_fraction_of_radius=0.05,  # Set to None if trial dropping should be switched off
        drop_remaining_false_bout_data=True,  # After trials have been removed, remove remaining false bouts
        compression=9,
        plot_statistics=True,
        do_overwrite=False,
    ):
        """
        Initialize the SanityCheck class for filtering and validating bout data.

        Args:
            path_to_combined_data_file (str or Path): Path to the combined data
                file.
            max_interbout_interval (float, optional): Maximum allowed interbout
                interval. Set to None to ignore. Defaults to 10.
            max_contour_area (float, optional): Maximum allowed contour area.
                Set to None to ignore. Defaults to 2000.
            max_average_speed (float, optional): Maximum allowed average speed.
                Set to None to ignore. Defaults to 1.
            max_abs_orientation_change (float, optional): Maximum allowed
                absolute orientation change. Set to None to ignore. Defaults to
                150.
            max_radius (float, optional): Maximum allowed radius (fraction of
                arena). Set to None to ignore. Defaults to 5/6.
            drop_trial_if_fraction_of_interbout_interval (float, optional):
                Fraction threshold for dropping trials based on interbout
                interval. Set to None to disable. Defaults to 0.05.
            drop_trial_if_fraction_of_false_contour_area (float, optional):
                Fraction threshold for dropping trials based on contour area.
                Set to None to disable. Defaults to 0.05.
            drop_trial_if_fraction_of_false_average_speed (float, optional):
                Fraction threshold for dropping trials based on average speed.
                Set to None to disable. Defaults to 0.05.
            drop_trial_if_fraction_of_abs_orientation_change (float, optional):
                Fraction threshold for dropping trials based on orientation
                change. Set to None to disable. Defaults to 0.05.
            drop_trial_if_fraction_of_radius (float, optional): Fraction
                threshold for dropping trials based on radius. Set to None to
                disable. Defaults to 0.05.
            drop_remaining_false_bout_data (bool, optional): Whether to remove
                remaining false bouts after trial removal. Defaults to True.
            compression (int, optional): Compression level for HDF5 output.
                Defaults to 9.
            plot_statistics (bool, optional): Whether to plot statistics for
                each property. Defaults to True.
            do_overwrite (bool, optional): Whether to overwrite existing checked
                data. Defaults to False.
        """
        self.path_to_combined_data_file = Path(path_to_combined_data_file)

        self.max_interbout_interval = max_interbout_interval
        self.max_contour_area = max_contour_area
        self.max_average_speed = max_average_speed
        self.max_abs_orientation_change = max_abs_orientation_change
        self.max_radius = max_radius
        self.drop_trial_if_fraction_of_interbout_interval = drop_trial_if_fraction_of_interbout_interval
        self.drop_trial_if_fraction_of_false_contour_area = drop_trial_if_fraction_of_false_contour_area
        self.drop_trial_if_fraction_of_false_average_speed = drop_trial_if_fraction_of_false_average_speed
        self.drop_trial_if_fraction_of_abs_orientation_change = drop_trial_if_fraction_of_abs_orientation_change
        self.drop_trial_if_fraction_of_radius = drop_trial_if_fraction_of_radius
        self.drop_remaining_false_bout_data = drop_remaining_false_bout_data
        self.compression = compression
        self.plot_statistics = plot_statistics
        self.do_overwrite = do_overwrite

    def drop_bout_data(self, prop_name, max_value):
        """
        Remove bouts from the DataFrame where the property exceeds max_value.

        Args:
            prop_name (str): Name of the property to check.
            max_value (float): Maximum allowed value for the property. If None,
                no data is dropped.
        """
        if max_value is None:
            return

        n_bouts_before = self.df.shape[0]
        self.df = self.df[self.df[prop_name] <= max_value]
        n_bouts_after = self.df.shape[0]

        if n_bouts_before == 0:
            return  # there are no trials left anymore
        print(f'\t{(n_bouts_before - n_bouts_after)/n_bouts_before * 100: 3.2f} % of remaining bouts  dropped ({n_bouts_before - n_bouts_after} of {n_bouts_before}): {prop_name} > {max_value}.')

    def drop_trial_data(self, prop_name, max_value, max_ratio_false):
        """
        Remove entire trials where the fraction of false values exceeds threshold.

        Args:
            prop_name (str): Name of the property to check.
            max_value (float): Maximum allowed value for the property. If None,
                no data is dropped.
            max_ratio_false (float): Maximum allowed fraction of false values in
                a trial. If None, no trials are dropped.
        """
        if max_value is None or max_ratio_false is None:
            return

        # Group based on all indices, to get individual trials
        grouped = self.df.groupby(self.df.index.names)
        n_trials_before = len(grouped)

        self.df = grouped.filter(lambda x: (sum(x[prop_name] > max_value) / len(x[prop_name])) <= max_ratio_false)

        n_trials_after = len(self.df.groupby(self.df.index.names))

        if n_trials_before == 0:
            return  # there are no trials left anymore
        print(f'\t{(n_trials_before - n_trials_after)/n_trials_before * 100: 3.2f} % of remaining trials dropped ({n_trials_before - n_trials_after} of {n_trials_before}): too many ({max_ratio_false}) {prop_name} > {max_value} per trial.')

    def plot_property_statistics(self, prop_name, max_value, max_ratio_false,
                                xlabel, bin_min=None, bin_max=None):
        """
        Plot statistics and histograms for a given property and exclusion rule.

        Args:
            prop_name (str): Name of the property to plot.
            max_value (float): Maximum allowed value for the property.
            max_ratio_false (float): Maximum allowed fraction of false values in
                a trial.
            xlabel (str): Label for the x-axis in plots.
            bin_min (float, optional): Minimum value for histogram bins.
                Defaults to None.
            bin_max (float, optional): Maximum value for histogram bins.
                Defaults to None.

        Notes:
            - Plots are saved to the 'sanity_check_plots' directory next to the
              combined data file.
            - No plots are generated if plotting is disabled or thresholds are
              None.
        """
        if max_value is None or max_ratio_false is None or not self.plot_statistics:
            return

        print(f"\t{datetime.now().strftime('%H:%M:%S')} Plotting {prop_name} (max_value={max_value}, max_ratio_false={max_ratio_false}) ...")

        # Remove nans
        df = self.df_original[self.df_original[prop_name].notna()][prop_name]

        vals = df.values
        # mode, count = stats.mode(vals, keepdims=True)  # Make sure the latest version of scipy is installed
        # mode = float(mode)
        mean = np.mean(vals)
        std = np.std(vals)

        if bin_min is None:
            bin_min = int(np.min(vals))

        if bin_max is None:
            bin_max = max(int(mean + 5 * std), bin_min + 1)

        df_group = df.groupby(df.index.names)
        n_trials_before = len(df_group)  # Grouping by the entire multi-index gives access to separate trials
        new_df = df_group.filter(lambda x: (sum(x > max_value) / len(x)) <= max_ratio_false)
        f_bouts = df_group.apply(lambda x: (sum(x > max_value) / len(x)))
        n_trials_after = len(new_df.groupby(new_df.index.names))

        n_excluded_trials = n_trials_before - n_trials_after
        p_excluded = n_excluded_trials / n_trials_before  # percentage of trials with too high fraction of "wrong" bout

        # Create figure
        fig = plt.figure(figsize=(8, 8))
        spec = fig.add_gridspec(6, 3)

        # Inspect random trials (zoomed in from bin_min to max_value)
        groups = [group for name, group in df_group]
        random_is = np.random.randint(0, len(df_group), 12)

        for i, random_i in enumerate(random_is):
            row_num = i // 2
            col_num = i % 2
            ax = fig.add_subplot(spec[row_num, col_num])

            f = f_bouts.values[random_i]

            # Get the name of the stimulus, and trial number
            stim_name = f_bouts.index.get_level_values("stimulus_name")[random_i]
            trial = f_bouts.index.get_level_values("trial")[random_i]

            group = groups[random_i]
            ax.hist(group, bins=np.linspace(bin_min, bin_max, 41))
            ax.axvline(max_value, color='tab:red')
            ax.set_title(f'{stim_name}\nTrial {trial}: {f * 100:.0f}%')

            if f > max_ratio_false:
                ax.title.set_color('tab:red')

            if row_num == 5:
                ax.set_xlabel(xlabel)

            ax.set_xlim([bin_min, bin_max * 1.01])

        # Histogram number of bouts per prop_value (zoomed in from bin_min to max_value)
        ax0 = fig.add_subplot(spec[0:2, 2])
        ax0.hist(vals, bins=np.linspace(bin_min, bin_max, 41))
        ax0.axvline(max_value, color='tab:red', label=f'condition ({max_value})') if max_value else None
        ax0.axvline(mean, color='k', label=r'$\mu$', linestyle='solid') if mean else None
        # ax0.axvline(mode, color='k', label=r'$Mo$', linestyle='solid') if mode else None
        # ax0.axvline(median, color='k', label=r'$\mu_{1/2}$', linestyle='solid') if median else None
        ax0.axvline(mean + std, color='k', label=r'$\mu + \sigma$', linestyle='dashed') if mean else None
        # ax0.axvline(mode + std, color='k', label=r'$Mo + \sigma$', linestyle='dashed') if mode else None
        ax0.set_xlabel(xlabel)
        ax0.legend(frameon=False)
        ax0.set_title('To set condition')
        ax0.set_xlim([bin_min, bin_max * 1.01])
        # Ratio of trials with condition
        ax1 = fig.add_subplot(spec[2:4, 2])
        ax1.hist(f_bouts * 100, bins=np.linspace(0, 100, 100))
        ax1.set_yscale('log')
        ax1.axvline(max_ratio_false * 100, color='tab:red')
        ax1.axvspan(max_ratio_false * 100, 100, alpha=0.1, color='tab:red', label='Excluded')
        ax1.set_ylabel('N trials')
        ax1.set_xlabel('Ratio')
        ax1.set_title(f'Excluded: {p_excluded * 100:.0f} %')
        ax1.legend(frameon=True)
        ax1.set_xlim([-0.1, 100])

        fig.supylabel('N bouts')
        fig.suptitle(xlabel)
        fig.tight_layout()

        # Store this alongside the combined datafile
        (self.path_to_combined_data_file.parent / 'sanity_check_plots').mkdir(exist_ok=True)
        plt.savefig(self.path_to_combined_data_file.parent / 'sanity_check_plots' / f'{prop_name}.pdf')
        plt.close()

    def run(self):
        """
        Run the full sanity check pipeline on bout data.

        Loads the combined data, applies all filtering and trial dropping rules,
        generates plots if enabled, and stores the cleaned data.

        Returns:
            pd.DataFrame: The sanity-checked bout data.
        """
        print("Start SanityCheck")

        # Load dataframes
        self.df_original = pd.read_hdf(self.path_to_combined_data_file, "all_bout_data_pandas")
        self.orignal_folder_names = self.df_original.index.unique('folder_name')
        try:
            self.df_checked = pd.read_hdf(self.path_to_combined_data_file, "all_bout_data_sanity_checked_pandas")
            self.checked_folder_names = self.df_checked.index.unique('folder_name')
        except KeyError as e:
            self.df_checked = pd.DataFrame()
            self.checked_folder_names = []

        if self.do_overwrite:
            self.df_checked = pd.DataFrame()
            self.checked_folder_names = []

        # Add radius and absolute estimated orientation change
        with pd.option_context("mode.chained_assignment", None):
            self.df_original['radius'] = np.sqrt(self.df_original['end_x_position'] ** 2 + self.df_original['end_y_position'] ** 2)
            self.df_original['abs_estimated_orientation_change'] = self.df_original['estimated_orientation_change'].abs()

        # Get dataframe with only folder_names that have not been checked yet
        # This dataframe will sequentially be reduced
        self.df = self.df_original.copy()
        if len(self.checked_folder_names):
            self.df = self.df.drop(self.checked_folder_names, level='folder_name')

        if self.df.empty:
            print(f'\tSanityCheck \033[96malready completed\033[0m')
            return self.df_checked

        # Drop all bouts with a NaN value
        n_bouts_before = self.df.shape[0]
        self.df = self.df[self.df["estimated_orientation_change"].notna()]
        n_bouts_after = self.df.shape[0]
        print(f'\t{(n_bouts_before - n_bouts_after)/n_bouts_before * 100: 3.2f} % of bouts dropped ({n_bouts_before - n_bouts_after} of {n_bouts_before}): NaN.')

        self.plot_property_statistics("end_contour_area", self.max_contour_area, self.drop_trial_if_fraction_of_false_contour_area, 'Contour area [px^2]')
        self.plot_property_statistics("average_speed", self.max_average_speed, self.drop_trial_if_fraction_of_false_average_speed, 'Speed [cm/s]')
        self.plot_property_statistics("interbout_interval", self.max_interbout_interval, self.drop_trial_if_fraction_of_interbout_interval, 'IBI [s]')
        self.plot_property_statistics("abs_estimated_orientation_change", self.max_abs_orientation_change, self.drop_trial_if_fraction_of_abs_orientation_change, 'Abs orientation change\n[deg]')
        self.plot_property_statistics("radius", self.max_radius, self.drop_trial_if_fraction_of_radius, 'Radius [-]', bin_min=0, bin_max=1)

        self.drop_trial_data("end_contour_area", self.max_contour_area, self.drop_trial_if_fraction_of_false_contour_area)
        self.drop_trial_data("average_speed", self.max_average_speed, self.drop_trial_if_fraction_of_false_average_speed)
        self.drop_trial_data("interbout_interval", self.max_interbout_interval, self.drop_trial_if_fraction_of_interbout_interval)
        self.drop_trial_data("abs_estimated_orientation_change", self.max_abs_orientation_change, self.drop_trial_if_fraction_of_abs_orientation_change)
        self.drop_trial_data("radius", self.max_radius, self.drop_trial_if_fraction_of_radius)

        if self.drop_remaining_false_bout_data:
            self.drop_bout_data("end_contour_area", self.max_contour_area)
            self.drop_bout_data("average_speed", self.max_average_speed)
            self.drop_bout_data("interbout_interval", self.max_interbout_interval)
            self.drop_bout_data("abs_estimated_orientation_change", self.max_abs_orientation_change)
            self.drop_bout_data("radius", self.max_radius)

        n_bouts_before = self.df_original.shape[0]
        n_trials_before = len(self.df_original.groupby(self.df_original.index.names))
        n_bouts_after = self.df.shape[0]
        n_trials_after = len(self.df.groupby(self.df.index.names))

        print(f'\t\033[96mTotal:\033[0m{(n_bouts_before - n_bouts_after)/n_bouts_before * 100: 3.2f} % of original bouts  dropped ({n_bouts_before - n_bouts_after} of {n_bouts_before}).')
        print(f'\t\033[96mTotal:\033[0m{(n_trials_before - n_trials_after)/n_trials_before * 100: 3.2f} % of original trials dropped ({n_trials_before - n_trials_after} of {n_trials_before}).')

        # Remove the columns we added
        self.df.drop(columns=['radius', ], inplace=True)
        self.df.drop(columns=['abs_estimated_orientation_change', ], inplace=True)

        # Store sanity checked data
        print(f"\t{datetime.now().strftime('%H:%M:%S')} Storing all_bout_data_sanity_checked_pandas ...", end='')
        self.df_checked = pd.concat([self.df_checked, self.df])
        self.df_checked.to_hdf(self.path_to_combined_data_file, key="all_bout_data_sanity_checked_pandas", complevel=self.compression)
        print('\033[92mdone\033[0m')

        print(f"\tSanityCheck \033[92mdone\033[0m")
        return self.df_checked
