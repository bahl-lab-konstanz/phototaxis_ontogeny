from settings.general_settings import resampling_window, rolling_window

# Paths
experiment_name = 'figS4D'
path_name = 'figS4D'

# Prepare data settings
do_tracking = True
do_event = True
do_bootstrap = False
do_median_df = True
do_rolling_df = False

# # AnalysisStart
split_dict = None
flip_dict = {
    'grey_0': {'flip': False, 'new_stim_name': 'control'},  # 0  # self.c_mid
    'grey_1': {'flip': 'y-axis', 'new_stim_name': 'control'},  # 0  # self.c_mid
    # Azimuth  self.c_min to self.c_mid
    'azimuth_left_bright_right_dark': {'flip': 'y-axis', 'new_stim_name': 'azimuth_left_dark_right_bright'},    # 1
    'azimuth_left_dark_right_bright': {'flip': False, 'new_stim_name': 'azimuth_left_dark_right_bright'},       # 2
    'azimuth_left_bright_right_dark_virtual_yes': {'flip': 'y-axis', 'new_stim_name': 'azimuth_left_dark_right_bright_virtual_yes'},    # 3
    'azimuth_left_dark_right_bright_virtual_yes': {'flip': False, 'new_stim_name': 'azimuth_left_dark_right_bright_virtual_yes'},       # 4
    'azimuth_left_bright_right_dark_avg': {'flip': 'y-axis', 'new_stim_name': 'azimuth_left_dark_right_bright_avg'},    # 13
    'azimuth_left_dark_right_bright_avg': {'flip': False, 'new_stim_name': 'azimuth_left_dark_right_bright_avg'},       # 14

   }
label_dict = None
stim_names = [
    'control',
    'azimuth_left_dark_right_bright',
    'azimuth_left_dark_right_bright_avg',
    'azimuth_left_dark_right_bright_virtual_yes',
]
