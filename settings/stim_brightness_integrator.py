
# Imports
import pandas as pd

# Paths
experiment_name = 'integrator'
path_name = experiment_name

# Prepare data settings
do_tracking = False
do_event = True
do_bootstrap = True
do_median_df = True
do_rolling_df = False

# Stimulus settings
stim_names = []  # defined in loop below
t_set = 20  # seconds
t_intervals = [0, 1, 2, 5, 10]  # seconds (stim name converts to integers)
t_test = 30  # seconds

# AnalysisStart
resampling_window = pd.Timedelta(1, unit='s')   # to compute median_df
rolling_window = pd.Timedelta(1, unit='s')    # to compute median_df and rolled_df

label_dict = None
split_dict = {}
flip_dict = {}
stim_dict = {}
for t_interval in t_intervals:
    # Dark set, bright interval ###############################################
    left_stim_name = f'{t_interval:02.0f}s_LbrightRbright_LbrightRdark'
    right_stim_name = f'{t_interval:02.0f}s_LbrightRbright_LdarkRbright'
    stim_names.append(right_stim_name)

    # Shift time to align with start of the test period
    split_dict[left_stim_name] = {
        't_start': 0, 't_end': t_set + t_interval + t_test,
        't_shift': - t_set - t_interval,
        'stim_query': left_stim_name
    }
    split_dict[right_stim_name] = {
        't_start': 0, 't_end': t_set + t_interval + t_test,
        't_shift': - t_set - t_interval,
        'stim_query': right_stim_name
    }

    flip_dict[left_stim_name] = {
        'flip': True,
        'new_stim_name': right_stim_name,
    }
    flip_dict[right_stim_name] = {
        'flip': False,
        'new_stim_name': right_stim_name,
    }

    stim_dict[right_stim_name] = {
        'bs_left': [0, 0, 300, 0, 0],
        'bs_right': [0, 0, 300, 300, 0],
        'ts': [-30, -t_set - t_interval, -t_interval, 0, t_test],
    }

    # Bright set, dark interval ###############################################
    left_stim_name = f'{t_interval:02.0f}s_LdarkRdark_LbrightRdark'
    right_stim_name = f'{t_interval:02.0f}s_LdarkRdark_LdarkRbright'
    stim_names.append(right_stim_name)

    split_dict[left_stim_name] = {
        't_start': 0, 't_end': t_set + t_interval + t_test,
        't_shift': - t_set - t_interval,
        'stim_query': left_stim_name
    }
    split_dict[right_stim_name] = {
        't_start': 0, 't_end': t_set + t_interval + t_test,
        't_shift': - t_set - t_interval,
        'stim_query': right_stim_name
    }

    flip_dict[left_stim_name] = {
        'flip': True,
        'new_stim_name': right_stim_name,
    }
    flip_dict[right_stim_name] = {
        'flip': False,
        'new_stim_name': right_stim_name,
    }

    stim_dict[right_stim_name] = {
        'bs_left': [300, 300, 0, 0, 0],
        'bs_right': [300, 300, 0, 300, 0],
        'ts': [-30, -t_set - t_interval, -t_interval, 0, t_test],
    }

