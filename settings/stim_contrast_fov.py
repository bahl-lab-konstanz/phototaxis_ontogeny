# Paths
experiment_name = 'contrast_fov_5'
path_name = experiment_name

# Prepare data settings
do_tracking = False
do_event = True
do_bootstrap = False
do_median_df = True
do_rolling_df = False

# Stimulus settings
radii = [0, 0.5, 1, 2, 5, 10]

# AnalysisStart
stim_names = []
flip_dict = {}
split_dict = {}
label_dict = {}

b_max = 300
b_min = 0
for radius in radii:
    for sign in [+1, -1]:
        new_stim_name = sign * radius

        # Use split_dict to exclude the first 10 seconds of each trial and get rid of stim_counter
        # TODO: we do not cut out the first 10 seconds here
        new_stim_name_right = f'{new_stim_name:02.1f}_right'
        new_stim_name_left = f'{new_stim_name:02.1f}_left'
        stim_query_right = f'left_{b_min:03.0f}lux_right_{b_max:03.0f}lux_radius{sign * radius:02.1f}cm'
        stim_query_left = f'left_{b_max:03.0f}lux_right_{b_min:03.0f}lux_radius{sign * radius:02.1f}cm'

        split_dict[new_stim_name_right] = {
            't_start': 0, 't_end': 40, 't_shift': 0,
            'stim_query': stim_query_right,
            'new_stim_name': new_stim_name_right,
        }
        split_dict[new_stim_name_left] = {
            't_start': 0, 't_end': 40, 't_shift': 0,
            'stim_query': stim_query_left,
            'new_stim_name': new_stim_name_left,
        }

        # Flip such that bright is on the right side ##########################
        # Flip dictionary: bright is already on the right side
        flip_dict[new_stim_name_right] = {
            'flip': False,
            'new_stim_name': new_stim_name,
        }
        # Flip dictionary: flip such that bright is on the right side
        flip_dict[new_stim_name_left] = {
            'flip': True,
            # Sign of radius is flipped too
            'new_stim_name': -1 * new_stim_name,
        }

        # Label dictionary: label_value is the signed radius
        label_dict[new_stim_name] = sign * radius

        # Store new stim names
        if new_stim_name not in stim_names:
            stim_names.append(new_stim_name)

