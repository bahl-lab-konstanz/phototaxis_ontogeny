# Paths
experiment_name = 'homogeneous_radius'
path_name = experiment_name

# Prepare data settings
do_tracking = False
do_event = True
do_bootstrap = False
do_median_df = True
do_rolling_df = False

# Stimulus settings
stim_names = None
brightnesses = [150]  # lux
radii = [0, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 10]

# AnalysisStart
# Remove first 10 seconds, for all stimuli
split_dict = {
    '010150': {
        't_start': 10, 't_end': 40,
        't_shift': -10,
        'stim_suffix': '',
    },
}
flip_dict = {}
label_dict = {}
label_values, label_queries = [], []  # Will be filled in the loop below
for brightness in brightnesses:
    for radius in radii:
        label_radius = radius if radius != 0 else 0.001  # Avoid 0 radius to keep the sign

        # Inside radius is bright
        str_adapt_i = f'{brightness:04.0f}lux_inside_{radius:04.2f}cm'
        left_stim_name = f'{str_adapt_i}_0'
        right_stim_name = f'{str_adapt_i}_1'
        new_stim_name = label_radius
        flip_dict[left_stim_name] = {'flip': False, 'new_stim_name': new_stim_name}
        flip_dict[right_stim_name] = {'flip': True, 'new_stim_name': new_stim_name}
        label_dict[new_stim_name] = label_radius
        label_queries.append(f'label_value == {label_radius}')
        label_values.append(label_radius)

        # Outside radius is bright
        str_adapt_i = f'{brightness:04.0f}lux_outside_{radius:04.2f}cm'
        left_stim_name = f'{str_adapt_i}_0'
        right_stim_name = f'{str_adapt_i}_1'
        new_stim_name = -1 * label_radius
        flip_dict[left_stim_name] = {'flip': False, 'new_stim_name': new_stim_name}
        flip_dict[right_stim_name] = {'flip': True, 'new_stim_name': new_stim_name}
        label_dict[new_stim_name] = -1 * label_radius
        label_queries.append(f'label_value == {-1 * label_radius}')
        label_values.append(-1 * label_radius)



