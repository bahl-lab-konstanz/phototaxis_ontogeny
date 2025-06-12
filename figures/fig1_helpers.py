"""
Figure 1: Through development, zebrafish invert their brightness preference
"""
import datetime
from itertools import product
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon, shapiro, circstd, pearsonr, linregress
from scipy.special import kl_div

from utils.plot_utils import *
from settings.general_settings import *

# #############################################################################
# User settings
# #############################################################################
# Import stimulus settings
from settings.stim_arena_locked import *

# Bins ########################################################################
nbins = 10  # set to even number to have a divide at 0
nbinedges = nbins + 1
max_radius = 5  # due to wall interactions
x_bins = np.linspace(-max_radius, max_radius, nbinedges)
y_bins = np.linspace(-max_radius, max_radius, nbinedges)
radius_bins = np.linspace(0, max_radius, nbinedges)
azimuth_bins = np.linspace(0, 360, nbinedges)
brightness_bins = np.arange(0, 601, 30)

x_bin_centers = (x_bins[1:] + x_bins[:-1]) / 2
y_bin_centers = (y_bins[1:] + y_bins[:-1]) / 2
radius_bin_centers = (radius_bins[1:] + radius_bins[:-1]) / 2
azimuth_bin_centers = (azimuth_bins[1:] + azimuth_bins[:-1]) / 2
brightness_bin_centers = (brightness_bins[1:] + brightness_bins[:-1]) / 2


# Stimuli to plot, in correct order
stim_dict = {
    'splitview_left_dark_right_bright': {
        'stim_name': 'splitview_left_dark_right_bright', 'stim_label': 'Half circles',
        'column_name': 'x_position',
        'bin_name': 'x_bin', 'bin_label': r'$X$-position (cm)', 'bin_label_avg': 'Average\n$x$-position (cm)',
        'bins': x_bins, 'bin_centers': x_bin_centers,
        'ticks': x_ticks, 'ticklabels': x_ticklabels,
        'ref_line': 0,
        'df_mean': 'x_df', 'df_ind': 'x_ind_df',  # later retrieved via eval
    },
    'azimuth_left_dark_right_bright_a': {
        'stim_name': 'azimuth_left_dark_right_bright', 'stim_label': 'Circular gradient',
        'column_name': 'azimuth',
        'bin_name': 'azimuth_bin', 'bin_label': r'Angle (deg)', 'bin_label_avg': f"Average cosine\nweighted angle ()",
        'bins': azimuth_bins, 'bin_centers': azimuth_bin_centers,
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
    # 'azimuth_left_dark_right_bright_x': {
    #     'stim_name': 'azimuth_left_dark_right_bright', 'stim_label': 'Circular gradient',
    #     'column_name': 'x_position',
    #     'bin_name': 'x_bin', 'bin_label': r'$x$-position (cm)', 'bin_label_avg': 'Average\n$x$-position (cm)',
    #     'bins': x_bins, 'bin_centers': x_bin_centers,
    #     'ticks': x_ticks, 'ticklabels': x_ticklabels, 'ref_line': 0,
    #     'df_mean': 'x_df', 'df_ind': 'x_ind_df',  # later retrieved via eval
    # },
    'center_bright_outside_dark': {
        'stim_name': 'center_bright_outside_dark', 'stim_label': 'Inward gradient',
        'column_name': 'radius',
        'bin_name': 'radius_bin', 'bin_label': r'Radius (cm)', 'bin_label_avg': 'Average\nradial position (cm)',
        'bins': radius_bins, 'bin_centers': radius_bin_centers,
        'ticks': radius_ticks, 'ticklabels': radius_ticklabels,
        'df_mean': 'radius_df', 'df_ind': 'radius_ind_df',  # later retrieved via eval
    },
    'center_dark_outside_bright': {
        'stim_name': 'center_dark_outside_bright', 'stim_label': 'Outward gradient',
        'column_name': 'radius',
        'bin_name': 'radius_bin', 'bin_label': r'Radius (cm)', 'bin_label_avg': 'Average\nradial position (cm)',
        'bins': radius_bins, 'bin_centers': radius_bin_centers,
        'ticks': radius_ticks, 'ticklabels': radius_ticklabels,
        'df_mean': 'radius_df', 'df_ind': 'radius_ind_df',  # later retrieved via eval
    },
    'control': {
        'stim_name': 'control', 'stim_label': 'Control',
        'column_name': 'radius',
        'bin_name': 'radius_bin', 'bin_label': r'Radius (cm)', 'bin_label_avg': 'Average\nradial position (cm)',
        'bins': radius_bins, 'bin_centers': radius_bin_centers,
        'ticks': radius_ticks, 'ticklabels': radius_ticklabels,
        'df_mean': 'radius_df', 'df_ind': 'radius_ind_df',  # later retrieved via eval
    },
    # 'control': {
    #     'stim_name': 'control', 'stim_label': 'Control',
    #     'column_name': 'x_position',
    #     'bin_name': 'x_bin', 'bin_label': r'$x$-position (cm)', 'bin_label_avg': 'Average\n$x$-position (cm)',
    #     'bins': x_bins, 'bin_centers': x_bin_centers,
    #     'ticks': x_ticks, 'ticklabels': x_ticklabels, 'ref_line': 0,
    #     'df_mean': 'x_df', 'df_ind': 'x_ind_df',  # later retrieved via eval
    # }
    'azimuth_left_dark_right_bright_virtual_yes': {
        'stim_name': 'azimuth_left_dark_right_bright_virtual_yes', 'stim_label': 'Circular gradient',
        'column_name': 'azimuth',
        'bin_name': 'azimuth_bin', 'bin_label': r'Angle (deg)', 'bin_label_avg': f"Average cosine\nweighted angle ()",
        'bins': azimuth_bins, 'bin_centers': azimuth_bin_centers,
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
}

# Include for figure 2
stim_dict_fig2 = {
    'azimuth_left_dark_right_bright_virtual_yes': {
        'stim_name': 'azimuth_left_dark_right_bright_virtual_yes', 'stim_label': 'Virtual circular gradient',
        'column_name': 'azimuth',
        'bin_name': 'azimuth_bin', 'bin_label': r'Angle (deg)', 'bin_label_avg': f"Average cosine\nweighted angle ()",
        'bins': azimuth_bins, 'bin_centers': azimuth_bin_centers,
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
    'azimuth_left_dark_right_bright': {
        'stim_name': 'azimuth_left_dark_right_bright', 'stim_label': 'Circular gradient',
        'column_name': 'azimuth',
        'bin_name': 'azimuth_bin', 'bin_label': r'Angle (deg)', 'bin_label_avg': f"Average cosine\nweighted angle ()",
        'bins': azimuth_bins, 'bin_centers': azimuth_bin_centers,
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
}


stim_dict_quick = {
    'splitview_left_dark_right_bright': stim_dict['splitview_left_dark_right_bright'],
    'azimuth_left_dark_right_bright_a': stim_dict['azimuth_left_dark_right_bright_a'],
    'center_bright_outside_dark': stim_dict['center_bright_outside_dark'],
    'center_dark_outside_bright': stim_dict['center_dark_outside_bright'],
    'control': stim_dict['control'],
}


# Colormap range for 2D density plots
ref_vmin, ref_vmax = 0.5, 1.5  # for larvae and juveniles
test_vmin, test_vmax = 1, 1.5  # for agents

# #############################################################################
# Misc helper functions
# #############################################################################
def store_binned_data(
        tracking_df,
        test_x_df, test_radius_df, test_azimuth_df,
        x_ind_df, radius_ind_df, azimuth_ind_df,
        path_to_binned_folder, agents_str
):
    for df, key_suffix in zip(
            [tracking_df, test_x_df, test_radius_df, test_azimuth_df, x_ind_df, radius_ind_df, azimuth_ind_df],
            ['tracking', 'x', 'radius', 'azimuth', 'x', 'radius', 'azimuth']
    ):
        (
            df
            # Remove unnecessary index levels
            .droplevel(level=['fish_or_agent_name', 'experiment_repeat', 'folder_name'])
            # Store!
            .to_hdf(
                path_to_binned_folder.joinpath(f'{agents_str}.hdf5'),
                key=f'{agents_str}_{key_suffix}', mode='a', format='table')
        )


def load_binned_data(path_to_binned_folder, ref_agents_str):
    df_list = []
    for key_suffix in ['x_ind', 'radius_ind', 'azimuth_ind']:
        try:
            df = pd.read_hdf(
                path_to_binned_folder.joinpath(f'{ref_agents_str}.hdf5'),
                key=f'{ref_agents_str}_{key_suffix}'
            )
            df_list.append(df)
        except KeyError as e:
            print(f"\033[91mload_binned_data(): {e}\033[0m")
    return df_list


# #############################################################################
# Analysis functions
# #############################################################################
# Compute weighted azimuth histogram
def compute_weighted_histogram(group):
    values = group['azimuth']
    hist, _ = np.histogram(values, bins=azimuth_bins, density=True)
    return np.sum(np.cos(np.deg2rad(azimuth_bin_centers)) * hist) * 100  # no units


def compute_bins(tracking_df):
    # Compute bins ############################################################
    print(f"\tComputing bins | ", end='')
    tracking_df['radius'] = np.sqrt(tracking_df['x_position'] ** 2 + tracking_df['y_position'] ** 2)
    tracking_df['azimuth'] = np.rad2deg(np.arctan2(tracking_df['y_position'], tracking_df['x_position']))
    tracking_df['azimuth'] = (tracking_df['azimuth'] + 360) % 360  # Convert to [0, 360] deg
    # Use bin centers as labels
    tracking_df['x_bin'] = pd.cut(tracking_df['x_position'], bins=x_bins, labels=x_bin_centers, include_lowest=True)
    tracking_df['y_bin'] = pd.cut(tracking_df['y_position'], bins=y_bins, labels=y_bin_centers, include_lowest=True)
    tracking_df['radius_bin'] = pd.cut(tracking_df['radius'], bins=radius_bins, labels=radius_bin_centers, include_lowest=True)
    tracking_df['azimuth_bin'] = pd.cut(tracking_df['azimuth'], bins=azimuth_bins, labels=azimuth_bin_centers, include_lowest=True)

    # Group individual fish and stimuli
    print(f"grouping | ", end='')
    group = (
        tracking_df
        .groupby([
            'fish_age', 'fish_genotype', 'experiment_ID', 'fish_or_agent_name',
            'experiment_repeat', 'folder_name', 'stimulus_name', 'midline_length',
        ])
    )
    # Compute total number of frames per group
    total_frames = group.size()
    # Normalise such that the sum of probabilities is 1 within each group
    print(f"prob | ", end='')
    tracking_df['prob'] = 1 / group.transform('size')

    # Compute median within fish ##############################################
    print(f"median | ", end='')
    median_ind_df = group[['x_position', 'radius']].median()
    median_ind_df['azimuth'] = group.apply(compute_weighted_histogram)

    # # Compute mean within fish ################################################
    # print(f"mean | ", end='')
    # mean_ind_df = group[['x_position', 'radius']].mean()
    # mean_ind_df['azimuth'] = group.apply(compute_weighted_histogram)

    # # # apply circular mean for azimuth
    # def circular_mean_azimuth(x):
    #     # Remove NaN values; circmean does not handle them
    #     x = x.dropna()
    #     # Convert azimuth from degrees to radians, calculate circular mean, and convert back to degrees
    #     return np.rad2deg(circmean(np.deg2rad(x), low=0, high=2 * np.pi))
    #
    # # Apply the circular mean to azimuth
    # mean_ind_df['azimuth'] = group['azimuth'].apply(circular_mean_azimuth)

    # Compute std within fish #################################################
    print(f"std | ", end='')
    std_ind_df = group[['x_position', 'radius']].std()

    def circular_std_azimuth(x):
        # Remove NaN values; circstd does not handle them
        x = x.dropna()
        # Convert azimuth from degrees to radians, calculate circular mean, and convert back to degrees
        return np.rad2deg(circstd(np.deg2rad(x), low=0, high=2 * np.pi))

    std_ind_df['azimuth'] = group['azimuth'].apply(circular_std_azimuth)

    print(f"\033[92mdone\033[0m")
    # return tracking_df, mean_ind_df, std_ind_df, total_frames
    return tracking_df, median_ind_df, std_ind_df, total_frames


def compute_swim_properties_tracking(tracking_df, total_frames, bin_names: list | str):
    if not isinstance(bin_names, list):
        bin_names = [bin_names]

    print(f"\tComputing swim properties for", bin_names, end=' ')
    # Group data per fish and with different spatial bins
    print("| group ", end='')
    group = (
        tracking_df
        .groupby([
            'fish_age', 'fish_genotype', 'experiment_ID', 'fish_or_agent_name',
            'experiment_repeat', 'folder_name', 'stimulus_name',
            *bin_names
        ], observed=True)
    )

    # Compute probability of being in a bin: number of frames within a bin / total number of frames
    print("| prob ", end='')
    df = group.size() / total_frames
    df = df.to_frame(name='prob')  # Convert series to dataframe

    # Compute mean and sem over all fish
    print("| mean, median, sem, std ", end='')
    df_mean = df.groupby(['fish_age', 'fish_genotype', 'stimulus_name', *bin_names], observed=True)['prob'].agg(
        ['mean', 'median', 'sem', 'std'])
    print("| \033[92mdone\033[0m")

    return df_mean, df


def compute_swim_properties_event(event_df, bin_names: list | str):
    if not isinstance(bin_names, list):
        bin_names = [bin_names]

    print(f"Computing swim properties for", *bin_names, end=" ")
    # Group data with different spatial bins
    print("| group ", end='')
    group = event_df.groupby(
        ['fish_age', 'fish_genotype', 'experiment_ID', 'fish_or_agent_name',
         'experiment_repeat', 'folder_name', 'stimulus_name',
         *bin_names], observed=True)

    # Compute swim properties (median per fish and per bin)
    # TODO: fit distribution instead of median?
    print("| median within fish ", end='')
    group_median = group[['total_duration', 'total_distance', 'estimated_orientation_change', 'event_freq', 'average_speed', 'brightness']].median()
    # After taking the median, round brightness values again to their bin
    group_median['brightness_bin'] = pd.cut(group_median['brightness'], bins=brightness_bins, labels=brightness_bin_centers, include_lowest=True)
    print("| \033[92mdone\033[0m")
    return group_median


def get_stim_arena_locked(stim_name, nbins, r_max, c_min, c_mid, c_max):
    """Get brightness level for each x and y bin"""
    # Prepare arena-locked stimuli
    bins = np.linspace(-r_max, r_max, nbins)
    xs, ys = np.meshgrid(bins, bins)
    radius = np.sqrt(xs**2 + ys**2)
    angle = np.arctan2(ys, xs)  # radians: -pi to pi
    px = np.ones_like(radius) * np.nan  # must be an array of floats

    # Arena locked stimuli ####################################################
    if stim_name == 'control':
        px[radius <= r_max] = c_mid
    elif stim_name == 'splitview_left_dark_right_bright' or stim_name == 'splitview_left_dark_right_bright_virtual':
        px[xs < 0] = c_min
        px[xs >= 0] = c_mid
        px[radius > r_max] = np.nan
    elif stim_name == 'azimuth_left_dark_right_bright' or stim_name == 'azimuth_left_dark_right_bright_virtual' or stim_name == 'azimuth_left_dark_right_bright_virtual_yes':
        px = (c_mid - c_min) * (np.pi - np.abs(angle)) / np.pi + c_min
        px[radius > r_max] = np.nan
    elif stim_name == 'center_dark_outside_bright' or stim_name == 'center_dark_outside_bright_virtual':
        px = (c_max - c_min) * radius / r_max + c_min
        px[radius > r_max] = np.nan
    elif stim_name == 'center_bright_outside_dark' or stim_name == 'center_bright_outside_dark_virtual':
        px = (c_mid - c_min) * (r_max - radius) / r_max + c_min
        px[radius > r_max] = np.nan
    else:
        print(f"\033[91m{stim_name} not recognised\033[0m")

    return px, xs, ys   # (nbins, nbins), (nbins, nbins), (nbins, nbins)


# #############################################################################
# Stat functions
# #############################################################################
def get_bin_stats(mean_ind_df, agents, stim_dict, ):
    stat_str = (f'Computing bin statistics for {agents[0].name} and {agents[1].name}\n'
                f'\t{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}\n')
    stat_list = []
    stat_dict = {}

    # Loop over stimuli
    for i, stim_values in enumerate(stim_dict.values()):
        # Extract stimulus settings
        stim_name = stim_values['stim_name']
        column_name = stim_values['column_name']
        stat_str += f'\n{stim_name} | {column_name}\n'

        # Compare agents (diff.) to zero ######################################
        stat_str += f'\tBootstrapping vs 0 | {stim_name} | {column_name}\n'
        for agent0 in agents:
            try:
                group0 = mean_ind_df.query(agent0.query).xs(stim_name, level='stimulus_name')[column_name]
            except KeyError as e:
                print(f'get_bin_stats(): \033[91m{e}\033[0m')
                continue

            # # Compare to 0: bootstrapping
            boot_means0 = np.random.choice(group0, (10000, len(group0)), replace=True).mean(axis=1)
            ci_lower0, ci_upper0 = np.percentile(boot_means0, [2.5, 97.5])
            # # # Compute two-tailed p-value: proportion of samples at least as extreme as zero
            b_p_value0 = 2 * min(
                np.mean(boot_means0 >= 0),  # Right tail
                np.mean(boot_means0 <= 0)  # Left tail
            )
            # Store all results
            stat_dict['Bootstrapping'] = {
                'p0': b_p_value0, 'ci0': [ci_lower0, ci_upper0],
            }
            stat_list.append([
                agent0.name, 0, stim_name, None, column_name,
                None, None, None, None,
                b_p_value0, ci_upper0, ci_upper0,
            ])

            stat_str += (
                f'\t\t{p_value_to_stars(b_p_value0)}\t{b_p_value0:.5f}\t95% CI: [{ci_lower0:.2f}, {ci_upper0:.2f}]\t{agent0.name} vs 0\n'
            )

        # Compare combinations between agents and stimuli #####################
        combinations = list(product(agents, [stim_name, 'control']))  # all combinations of agents and stimuli
        for k, ((agent0, stim0), (agent1, stim1)) in enumerate(product(combinations, repeat=2)):
            # We do want to compare in both directions,
            # but are not interested self-comparisons or cross-comparisons
            if (agent0 == agent1) and (stim0 == stim1):
                # Skip self-comparisons
                W_stat, W_p_value, U_stat, M_p_value = None, None, None, None
            elif (agent0 != agent1) and (stim0 != stim1):
                # Skip cross comparisons
                W_stat, W_p_value, U_stat, M_p_value = None, None, None, None
            else:
                # Query the data for this combination
                try:
                    group1 = mean_ind_df.query(agent0.query).xs(stim0, level='stimulus_name')[column_name]
                    group2 = mean_ind_df.query(agent1.query).xs(stim1, level='stimulus_name')[column_name]
                except KeyError as e:
                    continue

                # Perform Mann-Whitney U test
                U_stat, M_p_value = mannwhitneyu(group1, group2)

                # Perform Welch's t-test, since variances are not necessarily equal
                W_stat, W_p_value = ttest_ind(group1, group2, equal_var=False)

                # Perform Shapiro-Wilk normality test
                shapiro_p1 = shapiro(group1).pvalue if len(group1) > 3 else np.nan
                shapiro_p2 = shapiro(group2).pvalue if len(group2) > 3 else np.nan

                # Paired test if within same agent
                if agent0 == agent1 and len(group1) == len(group2):
                    # Perform paired t-test if data is normally distributed
                    try:
                        if shapiro_p1 > 0.05 and shapiro_p2 > 0.05:
                            T_stat, T_p_value = ttest_rel(group1, group2)
                            test_name = "Paired t-test"
                        else:
                            # Use Wilcoxon signed-rank test if not normally distributed
                            T_stat, T_p_value = wilcoxon(group1, group2)
                            test_name = "Wilcoxon signed-rank test"
                    except Exception as e:
                        T_stat, T_p_value = None, None
                        print(f"\t\033[91mget_bin_stats(): {e}\033[0m")

                    # Bootstrap on paired-difference
                    n_boot = 1000
                    ci = 95
                    paired_differences = group2.values - group1.values  # stim_name - control
                    observed_mean = np.mean(paired_differences)
                    boot_means = [np.mean(np.random.choice(paired_differences, size=len(paired_differences), replace=True)) for _ in
                                  range(n_boot)]
                    # Compute confidence interval
                    lower_bound = np.percentile(boot_means, (100 - ci) / 2)
                    upper_bound = np.percentile(boot_means, 100 - (100 - ci) / 2)

                    # Compute p-value: Two-tailed test (proportion of samples as extreme as observed)
                    p_value = (np.sum(boot_means >= observed_mean) + np.sum(boot_means <= -observed_mean)) / n_boot

                else:
                    T_stat, T_p_value = None, None

                # Store result if test performed
                stat_str += (f'\t{agent0.name} vs {agent1.name} | {stim0} vs {stim1} | {column_name}\n'
                             f'\t\tMWU:    {p_value_to_stars(M_p_value)}\t{M_p_value:.10f} (U stat: {U_stat:.3f})\n'
                             f'\t\tWelchs: {p_value_to_stars(W_p_value)}\t{W_p_value:.10f} (W stat: {W_stat:.3f})\n')

                stat_dict['Mann-Whitney U test'] = {
                    f'{agent0.name} vs {agent1.name}': {
                        'p': M_p_value, 'stat': U_stat}
                }
                stat_dict['Welchs'] = {
                    f'{agent0.name} vs {agent1.name}': {
                        'p': W_p_value, 'stat': W_stat}
                }

            # Store all results
            stat_list.append([
                agent0.name, agent1.name, stim0, stim1, column_name,
                W_stat, W_p_value, U_stat, M_p_value,
                None, None, None,
            ])

    # Create dataframe
    stat_df = pd.DataFrame(
        stat_list, columns=[
            'agent0', 'agent1', 'stim0', 'stim1', 'column_name',
            'W_stat', 'W_p_value', 'U_stat', 'M_p_value',
            'Bootstrapping_p_value', 'ci_lower0', 'ci_upper0',
        ])

    return stat_list, stat_df, stat_str


def _get_agent_stats(
        ref_agent, test_agent, stim_dict,
        # Will be retrieved via eval
        x_ind_df, radius_ind_df, azimuth_ind_df,
        # Bootstrap settings
        do_bootstrap=False, i_bootstrap=0,
        ref_agent_exp_IDs=None, test_agent_exp_IDs=None,
):
    if isinstance(ref_agent_exp_IDs, type(None)):
        ref_agent_exp_IDs = x_ind_df.query(ref_agent.query).index.unique('experiment_ID').tolist()
    if isinstance(test_agent_exp_IDs, type(None)):
        test_agent_exp_IDs = x_ind_df.query(test_agent.query).index.unique('experiment_ID').tolist()

    _results_list = []
    # Loop over stimuli
    for j, stim_values in enumerate(stim_dict.values()):
        # Extract stimulus settings
        stim_name = stim_values['stim_name']
        df_ind = eval(stim_values['df_ind'])
        bin_name = stim_values['bin_name']

        # Select data for sampled individuals
        ref_df_ind = df_ind.query(ref_agent.query)
        ref_df_ind = ref_df_ind.loc[ref_df_ind.index.get_level_values('experiment_ID').isin(ref_agent_exp_IDs)]
        test_df_ind = df_ind.query(test_agent.query)
        test_df_ind = test_df_ind.loc[test_df_ind.index.get_level_values('experiment_ID').isin(test_agent_exp_IDs)]

        if ref_df_ind.empty or test_df_ind.empty:
            # Print in red
            print(f'_do_agent_stats():\t\033[93m No data for {ref_agent.name} vs {test_agent.name}: {stim_name}\033[0m')
            continue

        # Compute mean and std over individuals, within bins
        ref_group = ref_df_ind.xs(stim_name, level='stimulus_name').groupby(bin_name, observed=True)
        ref_mean = ref_group['prob'].mean()
        ref_std = ref_group['prob'].std()
        test_group = test_df_ind.xs(stim_name, level='stimulus_name').groupby(bin_name, observed=True)
        test_mean = test_group['prob'].mean()
        test_std = test_group['prob'].std()

        for do_subtract_control in [False, True]:
            if do_subtract_control:
                # Get control values
                ref_group_control = ref_df_ind.xs('control', level='stimulus_name').groupby(bin_name, observed=True)
                ref_mean_control = ref_group_control['prob'].mean()
                ref_std_control = ref_group_control['prob'].std()
                test_group_control = test_df_ind.xs('control', level='stimulus_name').groupby(bin_name, observed=True)
                test_mean_control = test_group_control['prob'].mean()
                test_std_control = test_group_control['prob'].std()
                # Subtract control
                ref_mean -= ref_mean_control
                test_mean -= test_mean_control
                ref_std = np.sqrt(ref_std ** 2 + ref_std_control ** 2)
                test_std = np.sqrt(test_std ** 2 + test_std_control ** 2)

            # Compute statistics for each bin and combine over bins
            # Kullback-Leibler divergence
            stim_kl = sum(kl_div(ref_mean.values, test_mean.values))
            # Sum of squared differences
            stim_ssd = sum((ref_mean.values - test_mean.values) ** 2)
            # Compute z-score for each bin and average over bins
            stim_z_scores = np.mean(abs(ref_mean - test_mean) / np.sqrt(ref_std ** 2 + test_std ** 2))
            # Compute MSE for each bin and average over bins
            stim_mses = np.mean((ref_mean - test_mean) ** 2)

            # Append results
            _results_list.append([
                ref_agent.name, test_agent.name, stim_name, bin_name, do_subtract_control, do_bootstrap, i_bootstrap,
                stim_kl, stim_ssd, stim_z_scores, stim_mses
            ])
    return _results_list


def get_agent_zscores(
        ref_agent, test_agent,
        stim_dict,
        # Will be retrieved via eval
        x_ind_df, radius_ind_df, azimuth_ind_df,
        do_bootstrap=True, n_bootstraps=1000,
):
    print(f"\tget_agent_zscores() | {ref_agent.name} vs {test_agent.name} | do_bootstrap={do_bootstrap} ", end='')

    # Compute for all individuals: no bootstrapping
    results_list = _get_agent_stats(
        ref_agent, test_agent, stim_dict,
        x_ind_df, radius_ind_df, azimuth_ind_df,
    )

    if do_bootstrap:
        # Get unique IDs for each agent
        exp_ID_list = []
        for agent in [ref_agent, test_agent]:
            exp_IDs = x_ind_df.query(agent.query).index.unique('experiment_ID').tolist()
            exp_ID_list.append(exp_IDs)

        for i_bootstrap in range(n_bootstraps):
            # Sample with replacement
            ref_agent_exp_IDs = rng.choice(exp_ID_list[0], len(exp_ID_list[0]), replace=True)
            test_agent_exp_IDs = rng.choice(exp_ID_list[1], len(exp_ID_list[1]), replace=True)

            # Compute with bootstrapping
            try:
                results_list.extend(
                    _get_agent_stats(
                        ref_agent, test_agent, stim_dict,
                        x_ind_df, radius_ind_df, azimuth_ind_df,
                        do_bootstrap, i_bootstrap,
                        ref_agent_exp_IDs=ref_agent_exp_IDs, test_agent_exp_IDs=test_agent_exp_IDs,
                    )
                )
            except Exception as e:
                print(e)

    print("| \033[92mdone\033[0m")
    return results_list


# #############################################################################
# Plot functions
# #############################################################################
# Trajectories ################################################################
def plot_single_trajectory(ax, exp_df, stim_name, agent, s=0.2, alpha=0.05):
    if stim_name not in exp_df.index.unique('stimulus_name'):
        return

    # Get data for this stimulus
    stim_df = exp_df.xs(stim_name, level='stimulus_name')
    ax.scatter(
        stim_df['x_position'], stim_df['y_position'],
        color=agent.color, edgecolors='none',  # edgecolor to 'none' to avoid transparency issues
        s=s, alpha=alpha,
    )

    set_lims(ax, [-6, 6], [-6, 6])
    hide_all_spines_and_ticks(ax)
    set_aspect(ax, 'equal')

    if stim_name == 'splitview_left_dark_right_bright':
        set_axlines(ax, axvlines=0)


def plot_exp_trajectories(
        exp_df, agent, stim_dict, k=0,
        s=0.2, alpha=0.05,
        ax_x_cm=2, ax_y_cm=2, x_offset=0.2,
):
    # Create figure for this individual
    fig = create_figure(fig_big_width, ax_y_cm + 2 * pad_y_cm)
    # loop over stimuli
    for i, stim_values in enumerate(stim_dict.values()):
        stim_name = stim_values['stim_name']
        if stim_name not in exp_df.index.unique('stimulus_name'):
            continue

        # Add axes
        j = 0  # assign row
        l, b, w, h = (
            pad_x_cm + i * 2 * ax_x_cm + k * ax_x_cm + (-x_offset if k == 1 else x_offset),
            pad_y_cm + j * ax_y_cm,
            ax_x_cm - pad_x_cm / 2,
            ax_y_cm - pad_y_cm / 2,
        )
        ax = add_axes(fig, l, b, w, h)

        plot_single_trajectory(ax, exp_df, stim_name, agent)

    # Add separate ax for scalebar
    i += 1
    l, b, w, h = (
        pad_x_cm + i * 2 * ax_x_cm + k * ax_x_cm + (-x_offset if k == 1 else x_offset),
        pad_y_cm + j * ax_y_cm,
        ax_x_cm - pad_x_cm / 2,
        ax_y_cm - pad_y_cm / 2,
    )
    ax = add_axes(fig, l, b, w, h)
    set_lims(ax, [-6, 6], [-6, 6])
    hide_all_spines_and_ticks(ax)
    set_aspect(ax, 'equal')
    add_scalebar(ax, size=1, label='1 cm', loc='lower right')

    return fig

def plot_all_trajectories(tracking_df_full, agents, stim_dict, path_to_fig_folder, **kwargs):
    # Plot separately for each agent
    for k, agent in enumerate(agents):
        # Loop over all individuals
        for exp_ID, exp_df in tracking_df_full.query(agent.query).groupby('experiment_ID'):
            print(f"\tplot_all_trajectories() | {agent.name} | {exp_ID}", end='\r')
            fig = plot_exp_trajectories(exp_df, agent, stim_dict, k, **kwargs)

            # Save figure, as png since we plot many data points
            savefig(fig, path_to_fig_folder.joinpath('trajectories', agent.name,
                                                     f'trajectory_{agent.name}_{exp_ID:03d}.png'), close_fig=True)
        print(f"\tplot_all_trajectories() | {agent.name} | \033[92mdone\033[0m")


# 2D density hexbin ###########################################################
def plot_2d_density_ax(ax, stim_df, n_fish_stim, agent, vmin=None, vmax=None):
    # We sum all probabilities normalized within fish and then divide by n_fish
    hb = ax.hexbin(
        stim_df['x_position'], stim_df['y_position'],
        # Compute mean over fish, weighting each fish equally
        # for this, we use 1 / n_fish and np.sum as reduce_C_function
        # we multiply with 100 to convert to percentage
        C=stim_df['prob'] / n_fish_stim * 100, reduce_C_function=np.sum,
        # C=stim_df['prob'] * 100, reduce_C_function=np.mean,
        gridsize=11, extent=(-6, 6, -6, 6),
        linewidths=0,  # no lines to avoid overlap
        cmap=agent.cmap,
        vmin=vmin, vmax=vmax,
    )

    # The outermost hexagons only include half of the data,
    # so we set their values to nan to avoid misinterpretation
    hex_x, hex_y = hb.get_offsets().T  # Extract (x, y) centers of hexagons
    radii = np.sqrt(hex_x ** 2 + hex_y ** 2)
    array = hb.get_array()
    array[radii >= 5] = np.nan
    hb.set_array(array)
    hb.set_clim(vmin, vmax)
    hb.set_cmap(agent.cmap)

    ax.set_aspect('equal')
    hide_all_spines_and_ticks(ax)
    set_ticks(ax, x_ticks=[], y_ticks=[])

    # # Plot circle with radius 5 and 6 cm
    # circle = plt.Circle((0, 0), 5, color=COLOR_ANNOT, fill=False, lw=1)
    # ax.add_artist(circle)

    # Print vmin and vmax
    if isinstance(vmax, type(None)):
        hb_vmin = hb.get_array().min()
        hb_vmax = hb.get_array().max()
        hb_vmean = hb.get_array().mean()
        print(f"{agent.name} | \t: vmin: {hb_vmin:.2f} | vmax: {hb_vmax:.2f} | vmean: {hb_vmean:.2f}")

    # Create colorbar based on vmin and vmax
    ticks = np.linspace(vmin, vmax, 4)
    ticklabels = [f'{tick:.1f}' for tick in ticks]
    cbar = get_colorbar(
        agent.cmap, ticks=ticks, ticklabels=ticklabels,
        orientation='vertical',
        figsize=(ax_x_cm * cm, ax_y_cm * cm)
    )
    return cbar


def plot_2d_density(
        tracking_df, agents, stim_dict,
        vmin=0, vmax=None,
        ax_x_cm=2, ax_y_cm=2, x_offset=0.2,
):

    cbars = []

    # Create figure
    fig = create_figure(fig_big_width, ax_y_cm + 2 * pad_y_cm)
    for k, agent in enumerate(agents):
        agent_tracking_df = tracking_df.query(agent.query)
        stim_names = agent_tracking_df.index.unique('stimulus_name')

        # loop over stimuli
        for i, stim_values in enumerate(stim_dict.values()):
            stim_name = stim_values['stim_name']
            print(f"Plotting 2D density hexbin | {agent.name} | {stim_name}", end='\r')
            if stim_name not in stim_names:
                continue

            # Add axes
            j = 0  # assign row
            l, b, w, h = (
                pad_x_cm + i * 2 * ax_x_cm + k * ax_x_cm + (-x_offset if k == 1 else x_offset),
                pad_y_cm + j * ax_y_cm,
                ax_x_cm - x_offset * 2,
                ax_y_cm - x_offset * 2,
            )
            ax = add_axes(fig, l, b, w, h)
            ax.set_title(stim_name)  # Add stimulus name for quick reference

            # Get data for this stimulus
            stim_df = agent_tracking_df.xs(stim_name, level='stimulus_name').copy()

            # We sum all probabilities normalized within fish and then divide by n_fish
            n_fish_stim = stim_df.index.unique('experiment_ID').size
            cbar = plot_2d_density_ax(ax, stim_df, n_fish_stim, agent, stim_values, vmin=vmin, vmax=vmax)
        cbars.append(cbar)  # Store one colorbar per agent

    fig.suptitle(
        f'{agents[0].cmap.name}, {agents[1].cmap.name} | vmin: {vmin:.1f} | vmax: {vmax:.1f}',
    )

    return fig, cbars


def compute_2d_density_control_ax(agent_tracking_df):
    # Create an empty axes to compute the control
    fig_empty, ax_empty = plt.subplots(1, 1)

    # Get control density histogram
    if 'control' in stim_names:
        control_df = agent_tracking_df.xs('control', level='stimulus_name')
        n_fish_control = control_df.index.unique('experiment_ID').size
        hb_control = ax_empty.hexbin(
            control_df['x_position'], control_df['y_position'],
            C=control_df['prob'] / n_fish_control * 100,  # Normalize by number of fish
            reduce_C_function=np.sum,
            gridsize=11, extent=(-6, 6, -6, 6),
            linewidths=0
        )
        control_array = hb_control.get_array()
    else:
        control_array = None  # No control available

    plt.close(fig_empty)  # Close the figure

    return control_array


def plot_2d_density_diff_ax(ax, stim_df, n_fish_stim, control_array, vmin=-0.4, vmax=0.4, cmap=CMAP_DIFF):
    # Compute hexagonal histogram for the stimulus
    hb_stim = ax.hexbin(
        stim_df['x_position'], stim_df['y_position'],
        C=stim_df['prob'] / n_fish_stim * 100,  # Normalize by number of fish
        reduce_C_function=np.sum,
        gridsize=11, extent=(-6, 6, -6, 6),
        linewidths=0
    )
    stim_array = hb_stim.get_array()

    # The outermost hexagons only include half of the data,
    # so we set their values to nan to avoid misinterpretation
    hex_x, hex_y = hb_stim.get_offsets().T  # Extract (x, y) centers of hexagons
    radii = np.sqrt(hex_x ** 2 + hex_y ** 2)
    stim_array[radii >= 5] = np.nan

    # Compute density difference if control exists
    if control_array is not None:
        density_diff = stim_array - control_array  # Both already normalized
        hb_stim.set_array(density_diff)
    else:
        hb_stim.set_array(stim_array)

    # Print vmin and vmax
    if isinstance(vmax, type(None)):
        hb_vmin = hb_stim.get_array().min()
        hb_vmax = hb_stim.get_array().max()
        hb_vmean = hb_stim.get_array().mean()
        print(f"\t: vmin: {hb_vmin:.2f} | vmax: {hb_vmax:.2f} | vmean: {hb_vmean:.2f}")

    hb_stim.set_clim(vmin, vmax)
    hb_stim.set_cmap(cmap)
    ax.set_aspect('equal')
    hide_all_spines_and_ticks(ax)
    set_ticks(ax, x_ticks=[], y_ticks=[])

    # Create colorbar based on vmin and vmax
    ticks = np.linspace(vmin, vmax, 5)
    ticklabels = [f'{tick:.1f}' for tick in ticks]
    cbar = get_colorbar(
        cmap, ticks=ticks, ticklabels=ticklabels,
        orientation='vertical',
        figsize=(ax_x_cm * cm, ax_y_cm * cm)
    )
    return cbar


def plot_2d_density_diff(
        tracking_df, agents, stim_dict,
        vmin=-0.4, vmax=0.4, cmap=CMAP_DIFF,
        ax_x_cm=2, ax_y_cm=2, x_offset=0.2,
):
    # Create figure
    fig = create_figure(fig_big_width, ax_y_cm + 2 * pad_y_cm)
    for k, agent in enumerate(agents):
        agent_tracking_df = tracking_df.query(agent.query)
        stim_names = agent_tracking_df.index.unique('stimulus_name')

        # Compute control density histogram for 2d density difference
        control_array = compute_2d_density_control_ax(agent_tracking_df)

        # Loop over stimuli
        for i, stim_values in enumerate(stim_dict.values()):
            stim_name = stim_values['stim_name']
            print(f"Plotting 2D density hexbin | {agent.name} | {stim_name}", end='\r')
            if stim_name not in stim_names:
                continue

            # Add axes
            j = 0  # assign row
            l, b, w, h = (
                pad_x_cm + i * 2 * ax_x_cm + k * ax_x_cm + (-x_offset if k == 1 else x_offset),
                pad_y_cm + j * ax_y_cm,
                ax_x_cm - x_offset * 2,
                ax_y_cm - x_offset * 2,
            )
            ax = add_axes(fig, l, b, w, h)
            ax.set_title(stim_name)  # Add stimulus name for quick reference

            # Get data for this stimulus
            stim_df = agent_tracking_df.xs(stim_name, level='stimulus_name')
            n_fish_stim = stim_df.index.unique('experiment_ID').size

            plot_2d_density_diff_ax(ax, stim_df, n_fish_stim, control_array, vmin=vmin, vmax=vmax, cmap=cmap)

    fig.suptitle(
        f'{agents[0].cmap.name}, {agents[1].cmap.name} | vmin: {vmin:.1f} | vmax: {vmax:.1f}',
    )

    # Create colorbars separately
    ticks = np.linspace(vmin, vmax, 5)
    ticklabels = [f'{tick:.1f}' for tick in ticks]
    cbar = get_colorbar(
        cmap, ticks=ticks, ticklabels=ticklabels,
        orientation='vertical',
        figsize=(ax_x_cm * cm, ax_y_cm * cm)
    )

    return fig, cbar


# 1D density ##################################################################
def get_chance_level(
    bin_name,
    x_bins, azimuth_bins, radius_bins,
):
    # Compute bin centers
    x_bin_centers = (x_bins[1:] + x_bins[:-1]) / 2
    azimuth_bin_centers = (azimuth_bins[1:] + azimuth_bins[:-1]) / 2
    radius_bin_centers = (radius_bins[1:] + radius_bins[:-1]) / 2

    if bin_name == 'x_bin':
        chance = 2 * np.sqrt(max_radius ** 2 - x_bin_centers ** 2)
        chance /= np.sum(chance)
        return x_bin_centers, chance
    elif bin_name == 'azimuth_bin':
        chance = 1 / len(azimuth_bin_centers) * np.ones_like(azimuth_bin_centers)
        return azimuth_bin_centers, chance
    elif bin_name == 'radius_bin':
        chance = radius_bin_centers / np.sum(radius_bin_centers)
        return radius_bin_centers, chance
    else:
        print(f"\033[93m\tplot_chance_level(): {bin_name} not defined\033[0m")
        return


def plot_chance_level(
        ax, bin_name,
        x_bins, azimuth_bins, radius_bins,
):
    x, chance = get_chance_level(bin_name, x_bins, azimuth_bins, radius_bins)
    ax.plot(
        x, chance, label='Chance',
        color=COLOR_ANNOT, linestyle='solid', linewidth=LW_ANNOT,
    )
    # Indicate center
    # if bin_name == 'x_bin':
    #     ax.vlines(
    #         0, ymin=0, ymax=y_max,
    #         color=COLOR_ANNOT, linestyle='solid', linewidth=LW_ANNOT,
    #     )


def plot_1d_density_ax(
        ax,
        agents, stim_name, stim_values, do_subtract,
        # Will be retrieved via eval
        x_df, radius_df, azimuth_df,
        # If the following are not None, individuals will be plotted
        x_ind_df=None, radius_ind_df=None, azimuth_ind_df=None,
        # If given, plot reference stimulus as line
        ref_stim_name=False,
):
    # Extract stimulus settings
    df_mean = eval(stim_values['df_mean'])
    df_ind = eval(stim_values['df_ind'])
    bin_name = stim_values['bin_name']
    bin_label = stim_values['bin_label']
    ticks = stim_values['ticks']
    ticklabels = stim_values['ticklabels']

    for agent in agents[::-1]:  # Plot larvae on top
        # Mean over individuals #######################################
        try:
            # Mean over individuals
            agent_df = df_mean.query(agent.query).xs(stim_name, level='stimulus_name')
            x = agent_df.index.get_level_values(bin_name).to_list()
            mean = agent_df['mean']
            median = agent_df['median']
            sem = agent_df['sem']
            std = agent_df['std']
        except KeyError as e:
            # Print warning
            print(f"\033[93m\t_plot_1d_density(): {agent} | {stim_name}:\033[0m {e}")
            continue

        if do_subtract == 'control':
            # Subtract mean over individuals for control stimulus
            control_df = df_mean.query(agent.query).xs('control', level='stimulus_name')
            mean -= control_df['mean']
            median -= control_df['median']
            sem = np.sqrt(sem ** 2 + control_df['sem'] ** 2)
        elif do_subtract == 'chance':
            # Subtract chance level
            x, chance = get_chance_level(bin_name, x_bins, azimuth_bins, radius_bins)
            mean -= chance
            median -= chance

        ax.errorbar(
            x, median.values,
            yerr=sem.values,
            color=agent.color, markerfacecolor=agent.markerfacecolor, markeredgecolor=agent.color,
            fmt='o', linestyle='none',
            label=agent.label,
        )

        # Plot reference data #################################################
        if ref_stim_name:
            # Mean over individuals
            ref_agent_df = df_mean.query(agent.query).xs(ref_stim_name, level='stimulus_name')
            ref_x = ref_agent_df.index.get_level_values(bin_name).to_list()
            ref_median = ref_agent_df['median']

            if do_subtract == 'control':
                # Subtract mean over individuals for control stimulus
                ref_median -= control_df['median']
            elif do_subtract == 'chance':
                # Subtract chance level
                ref_x, chance = get_chance_level(bin_name, x_bins, azimuth_bins, radius_bins)
                ref_median -= chance

            ax.plot(
                ref_x, ref_median.values,
                color=agent.color, linestyle='-', lw=LW_ANNOT,
                label=f'Chance', zorder=-50,
            )

        # Plot individuals ############################################
        if not isinstance(df_ind, type(None)):
            try:
                agent_ind_df = df_ind.query(agent.query).xs(stim_name, level='stimulus_name')
            except KeyError as e:
                continue

            ind_col_name = 'prob'
            if do_subtract == 'control':
                ind_col_name = 'prob_sub'
                # Subtract mean over individuals for control stimulus
                control_df = df_mean.query(agent.query).xs('control', level='stimulus_name')
                agent_ind_df[ind_col_name] = agent_ind_df['prob'] - control_df['mean']
            elif do_subtract == 'chance':
                ind_col_name = 'prob_sub'
                # Compute chance level
                x, chance = get_chance_level(bin_name, x_bins, azimuth_bins, radius_bins)
                chance_df = pd.DataFrame(data={bin_name: x, 'chance': chance})
                chance_df.set_index(bin_name, inplace=True)
                # Subtract chance level for each bin, by aligning the indices
                agent_ind_df[ind_col_name] = agent_ind_df['prob'] - (
                    agent_ind_df
                    .index.get_level_values(bin_name)   # Get all bin values
                    .astype(float)                      # Ensure categories are converted to float
                    .map(chance_df['chance'])           # Map chance level for each bin
                )

            sns.lineplot(
                data=agent_ind_df, x=bin_name, y=ind_col_name,
                hue='experiment_ID', palette=ListedColormap([agent.color]), alpha=ALPHA,
                zorder=-100,
                ax=ax, legend=False,
            )

    # Format
    hide_spines(ax, ['top', 'right', 'left'])
    # set_spine_position(ax, spines=['left'], distance=2)
    # Format y-axis
    if do_subtract == 'control' or do_subtract == 'chance':
        # set_labels(ax, y='Probability diff. (%)')
        set_axlines(ax, axhlines=0)
        # Set symmetric y-limits, with minimum ylim_min
        ylim = ax.get_ylim()
        if agent.name == 'larva' or agent.name == 'juvie':
            ylim_min = 0.07
            scalebar_height = 0.01
            scalebar_label = '1%'
        else:
            # print(ylim)
            ylim_min = 0.02
            scalebar_height = 0.005
            scalebar_label = '0.5%'
        data_abs_max = max(abs(ylim[0]), abs(ylim[1]), ylim_min)
        ylim = [-data_abs_max, data_abs_max]
        set_lims(ax, y=ylim)
        set_ticks(ax, y_ticks=[])
        set_labels(ax, y='')

        # Add scalebar in lower right corner
        add_scalebar_vertical(
            ax, size=scalebar_height, label=scalebar_label,
            thickness=1, loc='lower right', outside=True)
    else:
        set_labels(ax, y='Probability (%)')
        # # Plot chance level
        plot_chance_level(ax, bin_name, x_bins, azimuth_bins, radius_bins, )
        # Set y-limits and ticks
        if agent.name == 'larva' or agent.name == 'juvie':
            # scalebar_height = 0.01
            # scalebar_label = '1%'
            ylim = [-0.05, 0.35]
            y_ticks = [0, 0.1, 0.2, 0.3]
        else:
            ylim = [-0.05, 0.25]
            y_ticks = [0, 0.05, 0.1, 0.15, 0.2]
        y_ticklabels = [f'{tick * 100: 2.0f}' for tick in y_ticks]
        set_lims(ax, y=ylim)
        set_ticks(ax, y_ticks=y_ticks, y_ticklabels=y_ticklabels)
        set_bounds(ax, y=[y_ticks[0], y_ticks[-1]])
        show_spines(ax, ['left'])

    # Format x-axis
    set_labels(ax, x=bin_label)
    set_ticks(ax, x_ticks=ticks, x_ticklabels=ticklabels, )
    set_bounds(ax, x=(ticks[0], ticks[-1]))


def plot_1d_density(
        agents, stim_dict,
        # Will be retrieved via eval
        x_df, radius_df, azimuth_df,
        # If the following are not None, individuals will be plotted
        x_ind_df=None, radius_ind_df=None, azimuth_ind_df=None,
        ref_stim_name=None,
        ax_x_cm=4, ax_y_cm=3,
):
    # Ensure agents is a list
    if not isinstance(agents, list):
        agents = [agents]
    agent_str = '_and_'.join([agent.name for agent in agents])

    # Plot agents together
    fig_list = []
    for do_subtract in ['No', 'control', 'chance']:
        # Create figure for this agent
        fig = create_figure(fig_big_width, pad_y_cm + ax_y_cm)
        fig.suptitle(f"{agent_str.replace('_', ' ')} | {do_subtract}")  # Remove '_'
        # Loop over stimuli
        for i, stim_values in enumerate(stim_dict.values()):
            stim_name = stim_values['stim_name']
            print(f"Plotting 1D density plot | {stim_name}", end='\r')
            # Add axes
            j = 0  # assign row
            l, b, w, h = (
                pad_x_cm + i * ax_x_cm,
                pad_y_cm + j * ax_y_cm,
                ax_x_cm - pad_x_cm,
                ax_y_cm - pad_y_cm
            )
            ax = add_axes(fig, l, b, w, h)
            ax.set_title(stim_name)  # Add stimulus name for quick reference

            plot_1d_density_ax(
                ax, agents, stim_name, stim_values, do_subtract,
                x_df, radius_df, azimuth_df,
                x_ind_df, radius_ind_df, azimuth_ind_df,
                ref_stim_name=ref_stim_name,
            )

            # # Hide y-axis for all but first column
            # if i != 0:
            #     hide_spines(ax, ['left'])
            #     set_ticks(ax, y_ticks=[])
            #     set_labels(ax, y='')

        fig_list.append(fig)
    return fig_list


def plot_1d_density_all_bins(
        agents, stim_dict,
        # Will be retrieved via eval
        x_df, radius_df, azimuth_df,
        # If the following are not None, individuals will be plotted
        x_ind_df=None, radius_ind_df=None, azimuth_ind_df=None,
        ax_x_cm=4, ax_y_cm=3,
):
    # Ensure agents is a list
    if not isinstance(agents, list):
        agents = [agents]
    agent_str = '_and_'.join([agent.name for agent in agents])

    # Plot agents together
    fig_list = []
    for do_subtract in ['No', 'control', 'chance']:
        # Create figure for this agent
        fig = create_figure(fig_big_height, fig_big_width)
        fig.suptitle(agent_str)
        # Loop over stimuli
        for j, stim_values in enumerate(stim_dict.values()):  # Only loop over keys, since we will overwrite values
            stim_name = stim_values['stim_name']
            print(f"Plotting 1D density plot | {stim_name}", end='\r')
            # Loop over bins
            for i, (df_mean_name, df_ind_name, bin_name, bin_label, ticks, tick_labels) in enumerate(zip(
                ['x_df', 'radius_df', 'azimuth_df'],
                ['x_ind_df', 'radius_ind_df', 'azimuth_ind_df'],
                ['x_bin', 'radius_bin', 'azimuth_bin'],
                ['x position (cm)', 'radius (cm)', 'azimuth (deg)'],
                [x_ticks, radius_ticks, azimuth_ticks],
                [x_ticklabels, radius_ticklabels, azimuth_ticklabels],
            )):
                # Add axes
                l, b, w, h = (
                    pad_x_cm + i * ax_x_cm,
                    pad_y_cm + j * (ax_y_cm + pad_y_cm),  # add extra padding for title
                    ax_x_cm - pad_x_cm,
                    ax_y_cm - pad_y_cm
                )
                ax = add_axes(fig, l, b, w, h)

                # Overwrite stim_values for current bin settings
                stim_values = {
                    'df_mean': df_mean_name, 'df_ind': df_ind_name,
                    'bin_name': bin_name, 'bin_label': bin_label,
                    'ticks': ticks, 'ticklabels': tick_labels,
                }
                plot_1d_density_ax(
                    ax, agents, stim_name, stim_values, do_subtract,
                    x_df, radius_df, azimuth_df,
                    x_ind_df, radius_ind_df, azimuth_ind_df,
                )

                # Hide y-axis for all but first column
                if i != 0:
                    hide_spines(ax, ['left'])
                    set_ticks(ax, y_ticks=[])
                    set_labels(ax, y='')
                else:
                    set_spine_position(ax)

            # Add stimulus name for quick reference
            i += 1
            l, b, w, h = (
                pad_x_cm + i * ax_x_cm,
                pad_y_cm + j * (ax_y_cm + pad_y_cm),  # add extra padding for title
                ax_x_cm - pad_x_cm,
                ax_y_cm - pad_y_cm
            )
            ax = add_axes(fig, l, b, w, h)
            ax.set_title(stim_name)
            hide_all_spines_and_ticks(ax)
        fig_list.append(fig)
    return fig_list


def plot_1d_density_fig_s1(
        agents, stim_dict,
        # Will be retrieved via eval
        x_df, radius_df, azimuth_df,
        # If the following are not None, individuals will be plotted
        x_ind_df=None, radius_ind_df=None, azimuth_ind_df=None,
        ax_x_cm=4, ax_y_cm=3,
):
    do_subtract = 'No'

    # Ensure agents is a list
    if not isinstance(agents, list):
        agents = [agents]
    agent_str = '_and_'.join([agent.name for agent in agents])


    # Create figure for this agent
    fig = create_figure(5 * (pad_x_cm + ax_x_cm), 4*(pad_y_cm + ax_y_cm))
    fig.suptitle(agent_str)

    # First row: control stimulus, all bins
    j = 2
    stim_name = 'control'
    stim_values = stim_dict[stim_name]
    # Loop over bins
    for i, (df_mean_name, df_ind_name, bin_name, bin_label, ticks, tick_labels) in enumerate(zip(
            ['x_df', 'azimuth_df', 'radius_df'],
            ['x_ind_df', 'azimuth_ind_df', 'radius_ind_df'],
            ['x_bin', 'azimuth_bin', 'radius_bin'],
            ['x position (cm)', 'azimuth (deg)', 'radius (cm)'],
            [x_ticks, azimuth_ticks, radius_ticks],
            [x_ticklabels, azimuth_ticklabels, radius_ticklabels],
    )):
        # Add axes
        l, b, w, h = (
            pad_x_cm + (i + 1) * ax_x_cm,  # shift one column to right
            pad_y_cm + j * (ax_y_cm + pad_y_cm),  # add extra padding for title
            ax_x_cm - pad_x_cm,
            ax_y_cm - pad_y_cm
        )
        ax = add_axes(fig, l, b, w, h)

        # Overwrite stim_values for current bin settings
        stim_values = {
            'df_mean': df_mean_name, 'df_ind': df_ind_name,
            'bin_name': bin_name, 'bin_label': bin_label,
            'ticks': ticks, 'ticklabels': tick_labels,
        }
        plot_1d_density_ax(
            ax, agents, stim_name, stim_values, do_subtract,
            x_df, radius_df, azimuth_df,
            x_ind_df, radius_ind_df, azimuth_ind_df,
        )

        # Hide y-axis for all but first column
        if i != 0:
            hide_spines(ax, ['left'])
            set_ticks(ax, y_ticks=[])
            set_labels(ax, y='')
        else:
            set_spine_position(ax)
    # Add legend to last plot
    set_legend(ax)

    # Following rows: all stimuli, all bins
    for i, stim_values in enumerate(stim_dict.values()):  # Only loop over keys, since we will overwrite values
        stim_name = stim_values['stim_name']
        stim_label = stim_values['stim_label']
        # Loop over bins
        for j, (df_mean_name, df_ind_name, bin_name, bin_label, ticks, tick_labels) in enumerate(zip(
                ['x_df', 'azimuth_df', 'radius_df'],
                ['x_ind_df', 'azimuth_ind_df', 'radius_ind_df'],
                ['x_bin', 'azimuth_bin', 'radius_bin'],
                ['x position (cm)', 'azimuth (deg)', 'radius (cm)'],
                [x_ticks, azimuth_ticks, radius_ticks],
                [x_ticklabels, azimuth_ticklabels, radius_ticklabels],
        )):
            # Add axes
            l, b, w, h = (
                pad_x_cm + i * ax_x_cm,
                pad_y_cm + j * (ax_y_cm),  # add extra padding for title
                ax_x_cm - pad_x_cm,
                ax_y_cm - pad_y_cm
            )
            ax = add_axes(fig, l, b, w, h)

            # Overwrite stim_values for current bin settings
            stim_values = {
                'df_mean': df_mean_name, 'df_ind': df_ind_name,
                'bin_name': bin_name, 'bin_label': bin_label,
                'ticks': ticks, 'ticklabels': tick_labels,
            }
            plot_1d_density_ax(
                ax, agents, stim_name, stim_values, do_subtract,
                x_df, radius_df, azimuth_df,
                x_ind_df, radius_ind_df, azimuth_ind_df,
            )

            # Hide y-axis for all but first column
            if i != 0:
                hide_spines(ax, ['left'])
                set_ticks(ax, y_ticks=[])
                set_labels(ax, y='')
            else:
                set_spine_position(ax)

            # Add title
            if j == 2:
                ax.set_title(stim_label)
    # Add legend to last plot
    set_legend(ax)
    return fig


def _prepare_bin_stats_plot(
    mean_ind_df, stim_values, agents,
):
    agent_str = ' vs '.join([agent.name for agent in agents])

    # Extract stimulus settings
    stim_name = stim_values['stim_name']
    column_name = stim_values['column_name']
    bin_name = stim_values['bin_name']
    bin_label = stim_values['bin_label_avg']
    ref_line = stim_values.get('ref_line', None)

    if 'larva' in agent_str or 'juvie' in agent_str:
        ticks = stim_values['ticks']
        ticklabels = stim_values['ticklabels']
        if bin_name == 'azimuth_bin':
            # To properly represent the azimuth positions, we use the weighted
            # histogram of the azimuthal values based on a cosine projection.
            ticks = [-3, 0, 3]
            ticklabels = ticks
            ref_line = 0
    # Zoomed in for agents
    elif bin_name == 'x_bin':
        ticks, ticklabels = [-2, 0, 2], [-2, 0, 2]
    elif bin_name == 'azimuth_bin':
        ticks, ticklabels = [-1, 0, 1], [-1, 0, 1]
        ref_line = 0
    elif bin_name == 'radius_bin':
        ticks, ticklabels = [3, 4, 5], [3, 4, 5]
    else:
        raise ValueError(f"Bin name '{bin_name}' not recognized. "
                         f"Expected 'x_bin', 'azimuth_bin', or 'radius_bin'.")

    # Prepare data for plotting ###########################################
    # Mean within fish
    plot_df = mean_ind_df.reset_index()  # Reset index for easier filtering
    plot_df = plot_df[plot_df['stimulus_name'].isin([stim_name, 'control'])]  # Keep only the relevant stimuli
    plot_df['group'] = ''  # Add group column
    # # Assign groups, stimulus order and color
    palette_dict = {}
    stim_order = []
    counter = 0  # ensure unique groups in desired order
    for agent in agents:
        for current_stim_name in ['control', stim_name]:
            group_name = f"{counter} {agent.name} {current_stim_name}"
            stim_order.append(group_name)
            # Get rows corresponding to mean_ind_df.query(agent.query).xs(stim_name, level='stimulus_name')
            rows = plot_df.query(agent.query).query(
                f"'{current_stim_name}' in stimulus_name")  # xs(current_stim_name, level='stimulus_name', drop_level=False)
            plot_df.loc[rows.index, 'group'] = group_name
            if current_stim_name == 'control':
                palette_dict[group_name] = COLOR_ANNOT
            else:
                palette_dict[group_name] = agent.color
            counter += 1
    # Ensure correct ordering
    stim_order[-2:] = stim_order[-2:][::-1]  # Trick to swap last two values of stim order: get second control last
    # stim_order = stim_order[::-1]  # Reverse order to have larvae on top
    plot_df['group'] = pd.Categorical(plot_df['group'], categories=stim_order, ordered=True)

    return plot_df, palette_dict, ticks, ticklabels, ref_line


def plot_bin_stats_stripplot_ax(ax, mean_ind_df, stat_df, agents, stim_values):

    # Extract stimulus settings
    stim_name = stim_values['stim_name']
    column_name = stim_values['column_name']
    bin_label = stim_values['bin_label_avg']
    plot_df, palette_dict, ticks, ticklabels, ref_line = _prepare_bin_stats_plot(
        mean_ind_df, stim_values, agents,
    )

    # Stripplot ###########################################################
    strip = sns.stripplot(
        data=plot_df,
        x='group', y=column_name,  # Use 'group' to show all four categories
        hue='group', palette=palette_dict, alpha=ALPHA, size=MARKER_SIZE,
        marker=MARKER_HOLLOW,
        dodge=False, legend=False,
        ax=ax
    )

    # Add statistics
    # # Compare stim and control
    p_value_agent1 = p_value_to_stars(stat_df.query(
        f"stim0 == '{stim_name}' and stim1 == 'control' and agent0 == '{agents[1].name}' and agent1 == '{agents[1].name}'"
    )['M_p_value'].values[0])
    # # Compare stim between agents
    p_value_stim = p_value_to_stars(stat_df.query(
        f"stim0 == '{stim_name}' and stim1 == '{stim_name}' and agent0 == '{agents[0].name}' and agent1 == '{agents[1].name}'"
    )['M_p_value'].values[0])
    # # Compare stim and control
    p_value_agent0 = p_value_to_stars(stat_df.query(
        f"stim0 == '{stim_name}' and stim1 == 'control' and agent0 == '{agents[0].name}' and agent1 == '{agents[0].name}'"
    )['M_p_value'].values[0])

    # x-coordinates in dataspace, y-coordinates in axes space
    add_stats(ax, 0, 1, ANNOT_Y, p_value_agent0)
    add_stats(ax, 1, 2, ANNOT_Y_HIGH, p_value_stim)
    add_stats(ax, 2, 3, ANNOT_Y, p_value_agent1)

    # Format
    set_axlines(ax, axhlines=ref_line)
    hide_spines(ax, ['top', 'right', 'bottom'])
    # Format property axis
    ax.set_ylabel(bin_label)
    set_ticks(ax, y_ticks=ticks, y_ticklabels=ticklabels)
    set_bounds(ax, y=(ticks[0], ticks[-1]))
    set_lims(ax, y=(ticks[0], ticks[-1]))
    # Format age axis
    set_labels(ax, x='')
    set_ticks(ax, x_ticks=[])
    set_lims(ax, x=[-0.5, 3.5])  # Ensure all dots are visible


def plot_bin_stats(
        mean_ind_df, stat_df, agents, stim_dict,
        ax_x_cm=4, ax_y_cm=3,
        jitter=1,
):
    agent_str = ' vs '.join([agent.name for agent in agents])
    fig = create_figure(fig_big_width, 3 * (ax_y_cm + pad_y_cm))
    fig.suptitle(agent_str)

    # Loop over stimuli, plot agents in same plot
    for i, stim_values in enumerate(stim_dict.values()):

        # Stripplot ###########################################################
        # Add axes
        j = 2  # assign row
        l, b, w, h = (
            pad_x_cm + i * ax_x_cm,
            pad_y_cm + j * ax_y_cm,
            ax_x_cm - pad_x_cm,
            ax_y_cm - pad_y_cm
        )
        ax = add_axes(fig, l, b, w, h)

        plot_bin_stats_stripplot_ax(ax, mean_ind_df, stat_df, agents, stim_values)

        # Histograms ##########################################################
        # Extract stimulus settings
        stim_name = stim_values['stim_name']
        column_name = stim_values['column_name']
        bin_label = stim_values['bin_label_avg']
        plot_df, palette_dict, ticks, ticklabels, ref_line = _prepare_bin_stats_plot(
            mean_ind_df, stim_values, agents,
        )

        for k, agent in enumerate(agents):
            # Add axes
            j = 0  # assign row
            l, b, w, h = (
                pad_x_cm + i * ax_x_cm,
                pad_y_cm + (j + k) * ax_y_cm,
                ax_x_cm - pad_x_cm,
                ax_y_cm - pad_y_cm
            )
            ax = add_axes(fig, l, b, w, h)
            ax.set_title(stim_name.replace('_', ' '))  # Add stimulus name for quick reference

            # Histogram
            sns.histplot(
                data=plot_df.query(agent.query),
                y=column_name,
                hue='group', palette=palette_dict, edgecolor=None, alpha=0.6,
                bins=np.linspace(ticks[0], ticks[-1], 21),
                ax=ax, legend=False,
            )

            # Add statistics within histogram
            p_value = p_value_to_stars(stat_df.query(f"stim0 == '{stim_name}' and stim1 == 'control' and agent0 == '{agent.name}' and agent1 == '{agent.name}'")['M_p_value'].values[0])
            # ax.text(0.5, 1, p_value, transform=ax.transAxes, ha='center', va='bottom')
            ax.text(1, 0.5, p_value, transform=ax.transAxes, ha='left', va='center')

            # Format
            hide_spines(ax, ['top', 'right', 'bottom'])
            # Format property axis
            ax.set_ylabel(bin_label)
            set_ticks(ax, y_ticks=ticks, y_ticklabels=ticklabels)
            set_bounds(ax, y=(ticks[0], ticks[-1]))
            # if i != 0:
            #     # Remove y-axis
            #     ax.set_ylabel('')
            #     set_ticks(ax, y_ticks=[])
            if k == 0:
                # Format x-axis
                set_labels(ax, x='nr. of fish')
                set_ticks(ax, x_ticks=[0, 10, 20, 30])
                show_spines(ax, ['bottom'])
            else:
                set_labels(ax, x='')
                set_ticks(ax, x_ticks=[])
    return fig


def plot_midline_stats(
        mean_ind_df, agents, stim_dict,
        ax_x_cm=4, ax_y_cm=3,
        x_ticks=None
):
    if x_ticks is None:
        x_ticks = [0, 1, 2, 3]

    agent_str = ' vs '.join([agent.name for agent in agents])
    fig = create_figure(fig_big_width, (ax_y_cm + pad_y_cm))
    fig.suptitle(agent_str)

    # Loop over stimuli, plot agents in same plot
    for i, stim_values in enumerate(stim_dict.values()):
        # Extract stimulus settings
        stim_name = stim_values['stim_name']
        column_name = stim_values['column_name']
        bin_name = stim_values['bin_name']
        bin_label = stim_values['bin_label_avg']
        ref_line = stim_values.get('ref_line', None)
        ticks = stim_values['ticks']
        ticklabels = stim_values['ticklabels']
        if bin_name == 'azimuth_bin':
            # To properly represent the azimuth positions, we use the weighted
            # histogram of the azimuthal values based on a cosine projection.
            ticks = [-3, 0, 3]
            ticklabels = ticks
            ref_line = 0

        # Add axes
        j = 0  # assign row
        l, b, w, h = (
            pad_x_cm + i * ax_x_cm,
            pad_y_cm + j * ax_y_cm,
            ax_x_cm - pad_x_cm,
            ax_y_cm - pad_y_cm
        )
        ax = add_axes(fig, l, b, w, h)

        # Prepare data for plotting ###########################################
        try:
            plot_df = mean_ind_df.xs(stim_name, level='stimulus_name').reset_index()
        except KeyError as e:
            # Print warning
            print(f"\033[93m\tplot_midline_stats(): {stim_name}:\033[0m {e}")
            continue

        # Correlate data within age group
        res_dict = {}
        for agent in agents:
            agent_df = plot_df.query(agent.query)

            # Fit a linear regression and get correlation
            slope, intercept, r_value, p_value, _ = linregress(agent_df['midline_length'], agent_df[column_name])
            res_dict[agent.label] = (r_value, p_value)

            # Plot regression line
            x_vals = np.linspace(agent_df['midline_length'].min(), agent_df['midline_length'].max(), 100)
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, color=COLOR_ANNOT, linestyle='--', label=f'{agent.name} fit')

        # Create axis title with R and p values
        title_text = "\n".join([f"{name}: R={r:.2f}, p={p:.3g}" for name, (r, p) in res_dict.items()])
        ax.set_title(f"{stim_name}\n{title_text}")

        # Scatter plot ########################################################
        sns.scatterplot(
            data=plot_df,
            x='midline_length', y=column_name,
            hue='fish_age', palette=AGE_PALETTE_DICT, alpha=1, size=MARKER_SIZE,
            marker='o', legend=False,
            ax=ax
        )

        # Format
        # set_axlines(ax, axhlines=ref_line)
        hide_spines(ax, ['top', 'right'])
        set_spine_position(ax)
        # Format property axis
        ax.set_ylabel(bin_label)
        set_ticks(ax, y_ticks=ticks, y_ticklabels=ticklabels)
        set_bounds(ax, y=(ticks[0], ticks[-1]))
        set_lims(ax, y=(ticks[0], ticks[-1]))
        # Format age axis
        set_labels(ax, x='Midline length (cm)')
        set_ticks(ax, x_ticks=x_ticks)
        set_lims(ax, x=(x_ticks[0], x_ticks[-1]))

    return fig


def plot_stimulus_ax(ax, stim_name, nbins=256, r_max=6, c_min=0, c_mid=300, c_max=310, vmin=0, vmax=350):
    px, xs, ys = get_stim_arena_locked(stim_name, nbins, r_max, c_min, c_mid, c_max)
    ax.imshow(px, extent=[-6, 6, -6, 6], cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    hide_all_spines_and_ticks(ax)


def plot_stimuli(stim_dict, ax_x_cm=2, ax_y_cm=2, x_offset=0.2):
    fig = create_figure(fig_height=pad_y_cm + ax_y_cm)
    j = 0   # assign row
    k = 0   # assign column
    # Loop over stimuli
    for i, stim_values in enumerate(stim_dict.values()):
        # Extract stimulus settings
        stim_name = stim_values['stim_name']

        # Add axes
        l, b, w, h = (
            pad_x_cm + i * 2 * ax_x_cm + k * ax_x_cm + (-x_offset if k == 1 else x_offset),
            pad_y_cm + j * ax_y_cm,
            ax_x_cm - pad_x_cm / 2,
            ax_y_cm - pad_y_cm / 2,
        )
        ax = add_axes(fig, l, b, w, h)

        plot_stimulus_ax(ax, stim_name)

    return fig

