
# Imports
import pandas as pd

# Paths
experiment_name = 'circadian_rhythm'
path_name = experiment_name

# Prepare data settings
do_tracking = False
do_event = True
do_bootstrap = False
do_median_df = True
do_rolling_df = False

# Stimulus settings
stim_names = []  # defined in loop below
lux_values = [0, 60, 311]

# AnalysisStart
resampling_window = pd.Timedelta(1, unit='s')   # to compute median_df
rolling_window = pd.Timedelta(1, unit='s')    # to compute median_df and rolled_df

label_dict = None
split_dict = None
flip_dict = {}
stim_dict = {}
for lux_value in lux_values:
    str_lux_value = f'{lux_value:.2f}'.replace('.', '_')
    left_stim_name = f'A02_adapt_{str_lux_value}_left_{311}_right_{0}'
    right_stim_name = f'A02_adapt_{str_lux_value}_left_{0}_right_{311}'
    stim_names.append(right_stim_name)

    flip_dict[left_stim_name] = {'flip': True, 'new_stim_name': right_stim_name, }
    flip_dict[right_stim_name] = {'flip': False, 'new_stim_name': right_stim_name, }

    # stim_dict[left_stim_name] = {}
    stim_dict[right_stim_name] = {
        'stim_label': f'{lux_value} lux',
    }

