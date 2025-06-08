
# Imports
import pandas as pd
from settings.general_settings import resampling_window, rolling_window

# Paths
experiment_name = 'temporal'
path_name = experiment_name

# Prepare data settings
do_tracking = False
do_event = True
do_bootstrap = True
do_median_df = True
do_rolling_df = False

# Stimulus settings
stim_names = None
# Stimulus settings
fixed_dur = 10  # seconds
transition_durations = [0, 5, 10, 20]  # s: abrupt, fast, medium and slow
change_slopes = [1 / transition_duration if transition_duration > 0 else 0.5 for transition_duration in transition_durations]  # lux/s: abrupt, fast, medium and slow
change_slope_labels = ['300 lux/s', '120 lux/s', '60 lux/s', '30 lux/s']
label_values = [-300, -120,  -60,  -30,   30,   60,  120,  300]
label_queries = [f'label_value == {label_value}' for label_value in label_values]

# AnalysisStart
split_dict = {}
flip_dict = {}
label_dict = {}
counter = 0
for transition_duration_1, slope_label_1 in zip(transition_durations, change_slope_labels):
    for transition_duration_2, slope_label_2 in zip(transition_durations, change_slope_labels):
        old_stim_name = f'{fixed_dur / 2:03.0f}s_0lux_{transition_duration_1:03.0f}s_up_{fixed_dur:03.0f}s_300lux_{transition_duration_2:03.0f}s_down_{fixed_dur / 2:03.0f}s_0lux'

        # Increasing part
        t_start = fixed_dur/2
        t_end = t_start + transition_duration_1

        if transition_duration_1 == 0:
            t_end += 2  # Include two seconds from the start of the change

        split_dict[f'{transition_duration_1:03.0f}s_increase_{old_stim_name}'] = {
            't_start': t_start,
            't_end': t_end,
            't_shift': -1 * t_start,
            'new_stim_name': f'{slope_label_1}',
            'stim_query': old_stim_name,
        }
        flip_dict[f'{slope_label_1}'] = {
            'flip': 'halve fish',
        }

        # Decreasing part
        t_start = fixed_dur/2 + fixed_dur + transition_duration_1
        t_end = t_start + transition_duration_2
        if transition_duration_2 == 0:
            t_end += 2  # Include two seconds from the start of the change

        split_dict[f'{transition_duration_2:03.0f}s_decrease_{old_stim_name}'] = {
            't_start': t_start,
            't_end': t_end,
            't_shift': -1 * t_start,
            'new_stim_name': f'-{slope_label_2}',
            'stim_query': old_stim_name,
        }
        flip_dict[f'-{slope_label_2}'] = {
            'flip': 'halve fish',
        }

        # Label dictionary: label_value is the slope
        label_dict[f'{slope_label_1}'] = int(slope_label_1.split(' ')[0])
        label_dict[f'-{slope_label_2}'] = -1 * int(slope_label_2.split(' ')[0])




