from settings.general_settings import resampling_window, rolling_window

# Paths
experiment_name = 'arena_locked'
path_name = experiment_name

# Prepare data settings
do_tracking = True
do_event = True
do_bootstrap = False
do_median_df = True
do_rolling_df = False

# # AnalysisStart
split_dict = None
flip_dict = {
    # Note that there was a bug in the stimulus file, feeding fish x, y, angle
    #  as 0, 0, 0, to virtual stimuli.
    #  Real virtual stimuli are renamed to include '_yes' in the name.
    'grey_0': {'flip': False, 'new_stim_name': 'control'},  # 0  # self.c_mid
    'grey_1': {'flip': 'y-axis', 'new_stim_name': 'control'},  # 0  # self.c_mid
    # Azimuth  self.c_min to self.c_mid
    'azimuth_left_bright_right_dark': {'flip': 'y-axis', 'new_stim_name': 'azimuth_left_dark_right_bright'},  # 1
    'azimuth_left_dark_right_bright': {'flip': False, 'new_stim_name': 'azimuth_left_dark_right_bright'},  # 2
    'azimuth_left_bright_right_dark_virtual': {'flip': 'y-axis', 'new_stim_name': '10 lux'},  # 3
    'azimuth_left_dark_right_bright_virtual': {'flip': False, 'new_stim_name': '10 lux'},  # 4
    'azimuth_left_bright_right_dark_virtual_yes': {'flip': 'y-axis', 'new_stim_name': 'azimuth_left_dark_right_bright_virtual_yes'},  # 3
    'azimuth_left_dark_right_bright_virtual_yes': {'flip': False, 'new_stim_name': 'azimuth_left_dark_right_bright_virtual_yes'},  # 4
    # Radius
    'center_dark_outside_bright_0': {'flip': False, 'new_stim_name': 'center_dark_outside_bright'},  # 5 self.c_min to self.c_max (to keep animals away from edge)
    'center_dark_outside_bright_1': {'flip': 'y-axis', 'new_stim_name': 'center_dark_outside_bright'},  # 5 self.c_min to self.c_max (to keep animals away from edge)
    'center_bright_outside_dark_0': {'flip': False, 'new_stim_name': 'center_bright_outside_dark'},  # 6 self.c_min to self.c_mid (to keep animals away from edge)
    'center_bright_outside_dark_1': {'flip': 'y-axis', 'new_stim_name': 'center_bright_outside_dark'},  # 6 self.c_min to self.c_mid (to keep animals away from edge)
    # 'center_dark_outside_bright_virtual_0': {'flip': False, 'new_stim_name': '10 lux'},  # 7 self.c_min to self.c_max (to keep animals away from edge)
    # 'center_dark_outside_bright_virtual_1': {'flip': 'y-axis', 'new_stim_name': '10 lux'},  # 7 self.c_min to self.c_max (to keep animals away from edge)
    # 'center_bright_outside_dark_virtual_0': {'flip': False, 'new_stim_name': '600 lux'},  # 8 self.c_min to self.c_mid (to keep animals away from edge)
    # 'center_bright_outside_dark_virtual_1': {'flip': 'y-axis', 'new_stim_name': '600 lux'},  # 8 self.c_min to self.c_mid (to keep animals away from edge)
    # 'center_dark_outside_bright_virtual_yes_0': {'flip': False, 'new_stim_name': 'center_dark_outside_bright_virtual'},  # 7 self.c_min to self.c_max (to keep animals away from edge)
    # 'center_dark_outside_bright_virtual_yes_1': {'flip': 'y-axis', 'new_stim_name': 'center_dark_outside_bright_virtual'},  # 7 self.c_min to self.c_max (to keep animals away from edge)
    # 'center_bright_outside_dark_virtual_yes_0': {'flip': False, 'new_stim_name': 'center_bright_outside_dark_virtual'},  # 8 self.c_min to self.c_mid (to keep animals away from edge)
    # 'center_bright_outside_dark_virtual_yes_1': {'flip': 'y-axis', 'new_stim_name': 'center_bright_outside_dark_virtual'},  # 8 self.c_min to self.c_mid (to keep animals away from edge)
    # Splitview  self.c_min and self.c_mid to make either side as attractive as possible
    'splitview_left_bright_right_dark': {'flip': 'y-axis', 'new_stim_name': 'splitview_left_dark_right_bright'},  # 9
    'splitview_left_dark_right_bright': {'flip': False, 'new_stim_name': 'splitview_left_dark_right_bright'},  # 10
    # 'splitview_left_bright_right_dark_virtual': {'flip': 'y-axis', 'new_stim_name': 'splitview_left_dark_right_bright_virtual'},  # 11
    # 'splitview_left_dark_right_bright_virtual': {'flip': False, 'new_stim_name': 'splitview_left_dark_right_bright_virtual'},  # 12
}
label_dict = None
stim_names = [
    'splitview_left_dark_right_bright',
    'azimuth_left_dark_right_bright',
    'center_dark_outside_bright',
    'center_bright_outside_dark',
    'control',
    'azimuth_left_dark_right_bright_virtual_yes',
]
