
# Imports
import numpy as np
import pandas as pd

# Paths
experiment_name = 'brightness_choice_simple'
path_name = experiment_name

# Prepare data settings
do_tracking = False
do_event = True
do_bootstrap = False
do_median_df = True
do_rolling_df = False

# Stimulus values
b_left_ns = np.asarray([
    300, 300,   10, 300,   10,   10, 300,   10, 300,
])
b_right_ns = np.asarray([
    300,   10, 300, 300,   10, 300,   10,   10, 300,
])
t_ns = np.asarray([
    0,  30,  60,  90, 120, 150, 180, 210, 240, # 270,
])

# Stimulus settings
stim_names = ['simple']
time_ticks = t_ns
time_lim = [0, 240]

# AnalysisStart
resampling_window = pd.Timedelta(1, unit='s')   # to compute median_df
rolling_window = pd.Timedelta(1, unit='s')    # to compute median_df and rolled_df
split_dict = None
flip_dict = {
    'stim10': {
        'flip': False,
        'new_stim_name': 'simple',
    },
    'stim11': {
        'flip': True,
        'new_stim_name': 'simple',
    },
}
label_dict = None
