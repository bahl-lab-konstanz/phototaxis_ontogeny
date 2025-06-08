# Standard library imports
import datetime
import os
import random
from pathlib import Path

# Third party library imports
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp, mannwhitneyu, wilcoxon, shapiro


# #############################################################################
# Swim data
# #############################################################################
def get_n_fish(df: pd.DataFrame, agents, stim_names=None):
    """Get number of fish in DataFrame"""
    # Ensure agents is iterable
    if not isinstance(agents, (list, tuple)):
        agents = [agents]

    n_fish_dict = {}

    print("Number of fish:")
    for agent in agents:
        if isinstance(agent, type(None)):
            continue

        agent_df = df.query(agent.query)

        if agent_df.empty:
            print(f"\033[93m\tNo data for {agent.label}\033[0m")
            continue

        if isinstance(stim_names, type(None)):
            _stim_names = [agent_df.index.unique('stimulus_name')[0]]  # Get first stimulus name
        else:
            _stim_names = stim_names

        for stim_name in _stim_names:
            try:
                stim_df = agent_df.xs(stim_name, level='stimulus_name')
                n_agent = (stim_df
                           .groupby([
                                'experiment_ID', 'fish_or_agent_name', 'setup_index',
                                'fish_genotype', 'fish_age', 'experiment_repeat',
                                'folder_name',
                            ])
                           .ngroups)
                ages = stim_df.index.unique('fish_age').to_list()
                print(f"\tn={n_agent:3d}: {agent.label.ljust(10)} {ages}dpf ({stim_name})")
                n_fish_dict[agent.name] = n_agent
            except KeyError:
                print(f"\t\033[93mn=   0: {agent.label.ljust(10)} (no data for {stim_name})\033[0m")
                n_fish_dict[agent.name] = np.nan
    return n_fish_dict


def load_bout_df(path_to_main_data_folder, path_name, agents):
    print(f"{datetime.datetime.now():%H:%M:%S} Loading combined bout data ", end='')
    return load_df(agents, path_to_main_data_folder, path_name, 'combined_data.hdf5', 'all_bout_data_pandas')


def load_event_df(path_to_main_data_folder, path_name, agents):
    print(f"{datetime.datetime.now():%H:%M:%S} Loading event data ", end='')
    return load_df(agents, path_to_main_data_folder, path_name, 'analysed_data.hdf5', 'all_bout_data_pandas_event')


def load_tracking_df(path_to_main_data_folder, path_name, agents, require_all_data=True):
    print(f"{datetime.datetime.now():%H:%M:%S} Loading tracking data ", end='')
    return load_df(agents, path_to_main_data_folder, path_name, 'analysed_data.hdf5', 'all_freely_swimming_tracking_data_pandas_event',
                   require_all_data=require_all_data)


def load_median_df(path_to_main_data_folder, path_name, agents):
    print(f"{datetime.datetime.now():%H:%M:%S} Loading median data ", end='')
    return load_df(agents, path_to_main_data_folder, path_name, 'analysed_data.hdf5', 'all_bout_data_pandas_median')


def load_rolled_df(path_to_main_data_folder, path_name, agents):
    print(f"{datetime.datetime.now():%H:%M:%S} Loading rolled data ", end='')
    return load_df(agents, path_to_main_data_folder, path_name, 'analysed_data.hdf5', 'all_bout_data_pandas_rolled')


def load_df(agents, path_to_main_data_folder, path_name, file_name, key, require_all_data=False, verbose=True):
    """Load and concat all data"""
    # Ensure agents is iterable
    if not isinstance(agents, (list, tuple)):
        agents = [agents]

    event_df_list = []
    for agent in agents:
        if isinstance(agent, type(None)):
            continue

        if verbose:
            print(f"{agent.name}... ", end='')
        try:
            if path_name == 'brightness_choice':
                df = _load_df_brightness_choice(path_to_main_data_folder, agent.name, file_name, key)
            else:
                df = pd.read_hdf(
                    path_to_main_data_folder
                    .joinpath(path_name, agent.name, file_name),
                    key=key
                )
        except FileNotFoundError as e:
            print(f"\033[93m{e}\033[0m")

            if require_all_data:
                # Return plain empty dataframe
                return pd.DataFrame()

            continue

        # Ensure that all indices and columns are up to date
        current_index_names = df.index.names
        if 'fish_or_agent_name' not in current_index_names:
            if verbose:
                print(f"| \033[93mRenaming 'fish_index' to 'fish_or_agent_name' for {agent.name} df\033[0m", end='')
            # Rename index level name
            df.index = df.index.rename('fish_or_agent_name', level='fish_index')

        # Before appending, we reset the index to avoid issues with different index name orders
        df_reset = df.reset_index()
        event_df_list.append(df_reset)

    if not len(event_df_list):
        # Cannot concatenate empty list
        return pd.DataFrame()

    event_df = pd.concat(event_df_list)
    # Set index to ensure the same index name order over all agents
    event_df.set_index(current_index_names, inplace=True)
    if verbose:
        print("| \033[92mdone\033[0m")
    return event_df


def _load_df_brightness_choice(path_to_main_data_folder, agent_name, file_name, key):
    if agent_name == 'juvie':
        return pd.read_hdf(
            path_to_main_data_folder.joinpath('brightness_choice_extended', agent_name, 'analysed_data.hdf5'),
            key=key)
    else:
        return pd.read_hdf(
            path_to_main_data_folder.joinpath('brightness_choice', agent_name, file_name),
            key=key
        )


def get_median_df(event_df, bin_name, groupby_labels=None):
    if isinstance(groupby_labels, type(None)):
        groupby_labels = [
            'stimulus_name',
            'fish_genotype', 'fish_age',
            'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
            'arena_index', 'setup_index', 'folder_name',
        ]

    group = event_df.groupby(groupby_labels + [bin_name], observed=True)
    median_df = group.median()
    # Remove time columns since they are not meaningful anymore
    median_df.drop(columns=['time', 'time_absolute'], inplace=True)

    # Compute number of turns and ratio of left turns
    percentage_turn_df = (group['left_events'].sum() + group['right_events'].sum()) / group['left_events'].size() * 100
    percentage_left_df = group['left_events'].sum() / (group['left_events'].sum() + group['right_events'].sum()) * 100
    # combine dataframes
    median_df = pd.concat([
        median_df,
        percentage_turn_df.rename('percentage_turns'),
        percentage_left_df.rename('percentage_left'),
    ], axis=1)
    return median_df


def get_median_df_time(
        trial_df: pd.DataFrame,
        resampling_window: pd.Timedelta,
        groupby_labels: list = None,
) -> pd.DataFrame:
    if isinstance(groupby_labels, type(None)):
        groupby_labels = [
            'stimulus_name',
            'fish_genotype', 'fish_age',
            'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
            'arena_index', 'setup_index', 'folder_name',
        ]

    print(f"{datetime.datetime.now():%H:%M:%S} Compute median df...", end='')

    # Create a local copy
    _df = trial_df.copy()

    # Prepare dataframe
    # important! rolling requires time_datetime to be monotonic
    _df['time_datetime'] = pd.to_datetime(_df['time'], unit='s')
    _df.sort_values(by='time_datetime', inplace=True)
    # # prepare index
    _df.reset_index(inplace=True)
    _df.set_index(['time_datetime'] + groupby_labels, inplace=True)

    # Resample using groupby
    group = _df.groupby([pd.Grouper(freq=resampling_window, level=0)] + groupby_labels)
    # Compute median
    median_df = group.median()
    # Compute number of turns and ratio of left turns
    percentage_turn_df = (group['left_events'].sum() + group['right_events'].sum()) / group['left_events'].size() * 100
    percentage_left_df = group['left_events'].sum() / (group['left_events'].sum() + group['right_events'].sum()) * 100
    # Compute number of left/right/straight events a sum
    median_df.drop(columns=['left_events', 'right_events', 'straight_events'], inplace=True)
    left_events_df = group['left_events'].sum().astype(int)
    right_events_df = group['right_events'].sum().astype(int)
    straight_events_df = group['straight_events'].sum().astype(int)

    # combine dataframes
    median_df = pd.concat([
        median_df,
        percentage_turn_df.rename('percentage_turns'),
        percentage_left_df.rename('percentage_left'),
        left_events_df.rename('left_events'),
        right_events_df.rename('right_events'),
        straight_events_df.rename('straight_events'),
    ], axis=1)

    # Set time as start of each resampling bin
    median_df.reset_index('time_datetime', inplace=True)
    median_df['time'] = median_df['time_datetime'].astype('int64') / int(1e9)
    print(' \033[92mdone\033[0m')

    return median_df


# #############################################################################
# Statistics
# #############################################################################
def p_value_to_stars(p_value: float):
    if p_value is not None and not np.isnan(p_value):
        # Convert p-value to string
        if p_value >= 0.05:
            return 'n.s.'
        elif 0.05 > p_value >= 0.01:
            return '*'  # r'$\ast$'
        elif 0.01 > p_value >= 0.001:
            return '**'  # r'$\ast\ast$'
        elif p_value < 0.001:
            return '***' # r'$\ast\ast\ast$'
        else:
            return 'Invalid'
    return None


def get_stats_two_groups(name0, name1, group0, group1, n_boot=10_000):
    stat_str = ''
    stat_dict = {}

    # Rank tests
    # # Compare age groups: Mann-Whitney U test
    m_stat, m_p_value = mannwhitneyu(group0, group1, alternative='two-sided')  # data is not normally distributed
    stat_str += f"\tMann-Whitney U test\n"
    stat_str += f"\t\t{p_value_to_stars(m_p_value)}\tp={m_p_value: .5f}\tstat={m_stat: .2f}\t{name0} vs. {name1}\n"
    stat_dict['Mann-Whitney U test'] = {'p': m_p_value, 'stat': m_stat}
    # # Compare to 0: one-sample Wilcoxon test
    res0 = wilcoxon(group0, zero_method='wilcox', alternative='two-sided')
    res1 = wilcoxon(group0, zero_method='wilcox', alternative='two-sided')
    w_stat0, w_p_value0 = res0
    w_stat1, w_p_value1 = res1
    stat_str += f"\tWilcoxon test\n"
    stat_str += f"\t\t{p_value_to_stars(w_p_value0)}\tp={w_p_value0: .5f}\tstat={w_stat0: .2f}\t{name0} vs. 0\n"
    stat_str += f"\t\t{p_value_to_stars(w_p_value1)}\tp={w_p_value1: .5f}\tstat={w_stat1: .2f}\t{name1} vs. 0\n"
    stat_dict['Wilcoxon test'] = {
        'p0': w_p_value0, 'stat0': w_stat0,
        'p1': w_p_value1, 'stat1': w_stat1,
    }
    # Bootstrapping
    # # Compare age groups: bootstrapping
    boot_diffs = np.array([
        np.mean(np.random.choice(group0, len(group0), replace=True)) -
        np.mean(np.random.choice(group1, len(group1), replace=True))
        for _ in range(n_boot)
    ])
    # # # Compute two-tailed p-value
    b_p_value = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))
    # # Compare to 0: bootstrapping
    boot_means0 = np.random.choice(group0, (10000, len(group0)), replace=True).mean(axis=1)
    boot_means1 = np.random.choice(group1, (10000, len(group1)), replace=True).mean(axis=1)
    ci_lower0, ci_upper0 = np.percentile(boot_means0, [2.5, 97.5])
    ci_lower1, ci_upper1 = np.percentile(boot_means1, [2.5, 97.5])
    # # # Compute two-tailed p-value: proportion of samples at least as extreme as zero
    b_p_value0 = 2 * min(
        np.mean(boot_means0 >= 0),  # Right tail
        np.mean(boot_means0 <= 0)  # Left tail
    )
    b_p_value1 = 2 * min(
        np.mean(boot_means1 >= 0),  # Right tail
        np.mean(boot_means1 <= 0)  # Left tail
    )
    stat_str += f"\tBootstrapping\n"
    stat_str += f"\t\t{p_value_to_stars(b_p_value)}\tp={b_p_value: .5f}\t{name0} vs. {name1}\n"
    stat_str += f"\t\t{p_value_to_stars(b_p_value0)}\tp={b_p_value0: .5f}\t95% CI: [{ci_lower0:.2f}, {ci_upper0:.2f}]\t{name0} vs 0\n"
    stat_str += f"\t\t{p_value_to_stars(b_p_value1)}\tp={b_p_value1: .5f}\t95% CI: [{ci_lower1:.2f}, {ci_upper1:.2f}]\t{name1} vs 0\n"
    stat_dict['Bootstrapping'] = {
        'p': b_p_value,
        'p0': b_p_value0, 'ci0': [ci_lower0, ci_upper0],
        'p1': b_p_value1, 'ci1': [ci_lower1, ci_upper1],
    }

    # T-test
    # # Check normality
    try:
        res_shapiro_0, p_shapiro_0 = shapiro(group0)
        res_shapiro_1, p_shapiro_1 = shapiro(group1)
    except Warning as e:
        # Do not print SmallSampleWarning
        res_shapiro_0, p_shapiro_0 = np.nan, np.nan
        res_shapiro_1, p_shapiro_1 = np.nan, np.nan
    stat_str += f"\tShapiro test\n"
    stat_str += f"\t\tp={p_shapiro_0: .5f}\t{name0}\n"
    stat_str += f"\t\tp={p_shapiro_1: .5f}\t{name1}\n"
    stat_dict['Shapiro'] = {
        'p0': p_shapiro_0, 'stat0': res_shapiro_0,
        'p1': p_shapiro_1, 'stat1': res_shapiro_1,
    }
    # Compare age groups: t-test
    t_stat, t_p_value = ttest_ind(group0, group1, equal_var=False)  # Welch’s t-test
    # Compare to 0: t-test
    t_stat0, t_p_value0 = ttest_1samp(group0, 0)
    t_stat1, t_p_value1 = ttest_1samp(group1, 0)
    stat_str += f"\tT-test\n"
    stat_str += f"\t\t{p_value_to_stars(t_p_value)}\tp={t_p_value: .5f}\tstat={t_stat: .2f}\t{name0} vs. {name1}\n"
    stat_str += f"\t\t{p_value_to_stars(t_p_value0)}\tp={t_p_value0: .5f}\tstat={t_stat0: .2f}\t{name0} vs. 0\n"
    stat_str += f"\t\t{p_value_to_stars(t_p_value1)}\tp={t_p_value1: .5f}\tstat={t_stat1: .2f}\t{name1} vs 0\n"
    stat_dict['T-test'] = {
        'p': t_p_value, 'stat': t_stat,
        'p0': t_p_value0, 'stat0': t_stat0,
        'p1': t_p_value1, 'stat1': t_stat1,
    }
    return stat_str, stat_dict


def compute_log_likelihood(y_true, y_pred):
    residuals = y_true - y_pred
    sigma2 = np.var(residuals, ddof=1)  # Unbiased estimate
    n = len(y_true)
    log_likelihood = -n / 2 * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)
    return log_likelihood


def compute_AIC(y_true, y_pred, k):
    logL = compute_log_likelihood(y_true, y_pred)
    return 2 * k - 2 * logL


def compute_BIC(y_true, y_pred, k):
    logL = compute_log_likelihood(y_true, y_pred)
    n_datapoints = len(y_true)
    return k * np.log(n_datapoints) - 2 * logL


# #############################################################################
# General functions
# #############################################################################
def log_model(x, a, b):
    return a + b * np.log(x)  # a + b * ln(x)


def my_double_linear(x, a_pos, a_neg, b):
    return np.where(x >= 0, a_pos * x + b, a_neg * x + b)


# #############################################################################
# Stimulus data
# #############################################################################
def get_b_values(ts, t_ns, b_left_ns, b_right_ns):
    """Calculate stimulus, based on retrieved sampling rate"""
    # Assert length of brightness values is equal
    assert len(b_left_ns) == len(b_right_ns)
    t_ns = t_ns[:len(b_left_ns)]  # Ensure same length

    # Find indices ensuring step change happens *at* `t_ns`
    indices = np.searchsorted(t_ns, ts, side='right') - 1

    # Map indices to brightness values
    b_left = b_left_ns[indices]
    b_right = b_right_ns[indices]
    return b_left, b_right

