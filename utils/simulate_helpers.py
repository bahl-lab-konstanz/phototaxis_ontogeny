
import datetime
import h5py
import json
import numpy as np
import pandas as pd
import time

from .simulate_agent import MyAgent


# #############################################################################
# Simulation functions
# #############################################################################
def task_run_agent(args):
    """Runs a simulation for a single agent and stores results.

    Args:
        args (list): List of arguments required to initialize and run the agent simulation.

    Returns:
        None. Results are saved to HDF5 files.
    """
    index_names = [  # In same order as experimental data
        'stimulus_name', 'fish_genotype', 'experiment_ID', 'fish_or_agent_name',
        'experiment_repeat', 'arena_index', 'setup_index', 'folder_name',
        'fish_age', 'trial',
        'trial_count_since_experiment_start', 'trial_time_since_experiment_start',
        'tracking_type'
    ]

    # Prepare agent ###########################################################
    # Unpack arguments
    (
        main_seed, path_to_raw_data_folder,
        do_fish_lock, n_trials, stim_names, stim_arrays, xs, ys,
        ts, dt, r_view,
        agent_index, agent_age, agent_genotype_name, agent_genotype_dict, folder_name,
        path_to_input_folder, progress_dict
    ) = args
    agent_seed = main_seed + agent_index

    # Initialize agent
    agent = MyAgent(agent_seed, agent_index, agent_age, agent_genotype_dict, path_to_input_folder)
    progress_dict[agent_index] = f"\tAgent{agent_index: 3d} (seed={agent_seed: 3d}) | started"

    # Create hdf5 file for this agent
    path_to_hdf5_file = path_to_raw_data_folder.joinpath(folder_name, f'{folder_name}.hdf5')
    path_to_hdf5_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path_to_hdf5_file, 'w') as f_hdf5:
        f_hdf5.attrs['experiment_ID'] = agent_index
        f_hdf5.attrs['fish_age'] = agent_age
        f_hdf5.attrs['fish_genotype'] = agent_genotype_name
        f_hdf5.attrs['folder_name'] = folder_name

    # Initialize data storage for this agent
    tracking_df_list = []
    bout_df_list = []

    # Loop over stimuli and trials ############################################
    for stim_num, (stim_name, stim_array) in enumerate(zip(stim_names, stim_arrays)):
        for trial_index in range(n_trials):
            print(f"\tAgent{agent_index: 3d} (seed={agent_seed: 3d}) | stim {stim_name} | trial {trial_index}")
            # Start trial simulation ##########################################
            # Reset state variables, new random position and orientation
            agent.start_agent()

            # Initialize data storage for this trial
            trial_tracking_list = []
            trial_bout_list = []

            # Loop over time steps
            for frame_num, t in enumerate(ts):
                # Get current position and orientation
                x, y, theta = agent.get_current_coordinates()  # cm, cm, deg

                # Get current eye brightness
                left_lux, right_lux = agent.get_eye_brightness(
                    stim_name, do_fish_lock, stim_array, frame_num,
                    xs, ys, x, y, theta, r_view
                )

                # Update agent
                tracking_data, bout_data = agent.update_agent(t, left_lux, right_lux, dt)

                # Store data, including trial index values
                trial_tracking_list.append(tracking_data)
                if bout_data:
                    trial_bout_list.append(bout_data)

            # Create dataframes for this trial ################################
            # # Default index values (these are the same for all stimuli, agents, and trials)
            setup_index, arena_index, experiment_repeat, trial_time_since_experiment_start, fish_or_agent_name = 0, 0, 0, 0, 0
            tracking_type = 'freely_swimming_agent'
            index_data = [
                stim_name, agent_genotype_name, agent_index, fish_or_agent_name,
                experiment_repeat, arena_index, setup_index, folder_name,
                agent_age, trial_index,
                trial_index, trial_time_since_experiment_start, tracking_type
            ]
            # Create a MultiIndex directly from index_data and index_names
            tracking_multiindex = pd.MultiIndex.from_tuples([index_data] * len(trial_tracking_list), names=index_names)
            bout_multiindex = pd.MultiIndex.from_tuples([index_data] * len(trial_bout_list), names=index_names)
            # Create the DataFrames
            trial_tracking_df = pd.DataFrame(trial_tracking_list, columns=agent.tracking_columns, index=tracking_multiindex)
            trial_bout_df = pd.DataFrame(trial_bout_list, columns=agent.bout_columns, index=bout_multiindex)
            # Append to list
            tracking_df_list.append(trial_tracking_df)
            bout_df_list.append(trial_bout_df)

    # Combine DataFrames for this agent
    tracking_df = pd.concat(tracking_df_list)
    bout_df = pd.concat(bout_df_list)

    # Convert cm to arena units to match experimental data (later this is multiplied by 6 again)
    tracking_df[['x_position', 'y_position', 'accumulated_path']] /= 6
    bout_df[[
        'start_x_position', 'start_y_position', 'start_accumulated_path',
        'end_x_position', 'end_y_position', 'end_accumulated_path',
        'x_position_change', 'y_position_change', 'distance_change', 'average_speed',
    ]] /= 6

    # Store data to hdf5 file
    tracking_df.to_hdf(path_to_hdf5_file, key='repeat00/all_freely_swimming_tracking_data_pandas', mode='a')
    bout_df.to_hdf(path_to_hdf5_file, key='repeat00/all_bout_data_pandas', mode='a')
    # Store meta data
    store_meta_data(path_to_hdf5_file, args, agent)

    # Update progress_dict
    progress_dict[agent_index] = f"\tAgent{agent_index: 3d} (seed={agent_seed: 3d}) | \033[92mdone\033[0m"


def store_meta_data(path_to_hdf5_file, args, agent):
    """Stores experiment and agent metadata in the HDF5 file.

    Args:
        path_to_hdf5_file (Path): Path to the HDF5 file.
        args (list): Arguments used for the simulation.
        agent (MyAgent): The agent instance.

    Returns:
        None.
    """
    # Unpack arguments
    (
        main_seed, path_to_raw_data_folder,
        do_fish_lock, n_trials, stim_names, stim_arrays, xs, ys,
        ts, dt, r_view,
        agent_index, agent_age, agent_genotype_name, agent_genotype_dict, folder_name,
        path_to_input_folder, progress_dict
    ) = args
    agent_seed = main_seed + agent_index

    # Open the HDF5 file in append mode
    with h5py.File(path_to_hdf5_file, 'a') as f_hdf:
        # Create a group for experiment configuration
        config_group = f_hdf.require_group('experiment_configuration')

        # Add metadata to the 'experiment_configuration' group
        config_group.attrs['main_seed'] = main_seed
        config_group.attrs['agent_seed'] = agent_seed
        config_group.attrs['path_to_raw_data_folder'] = path_to_raw_data_folder.as_posix()
        config_group.attrs['path_to_input_folder'] = path_to_input_folder.as_posix()
        config_group.attrs['do_fish_lock'] = do_fish_lock
        config_group.attrs['n_trials'] = n_trials
        config_group.attrs['stim_names'] = ','.join(stim_names)  # Or use JSON
        config_group.attrs['dt'] = dt
        config_group.attrs['r_view'] = r_view
        config_group.attrs['agent_index'] = agent_index
        config_group.attrs['fish_genotype'] = agent_genotype_name
        config_group.attrs['fish_age'] = agent_age
        config_group.attrs['folder_name'] = folder_name

        # Add agent dictionaries as JSON strings for flexibility
        # Convert the dictionaries before storing
        meta_popt_dict_serializable = json.dumps(convert_ndarray_to_list(agent.meta_popt_dict))
        agent_genotype_dict_serializable = json.dumps(convert_ndarray_to_list(agent_genotype_dict))
        # Store the converted dictionaries
        config_group.attrs['agent_genotype_dict'] = agent_genotype_dict_serializable
        config_group.attrs['meta_popt_dict'] = meta_popt_dict_serializable


# Function to convert numpy arrays in dictionaries to lists
def convert_ndarray_to_list(d):
    """Converts numpy arrays in a dictionary to lists for JSON serialization."""
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# #############################################################################
# Prepare stimuli
# #############################################################################
def load_stimulus(experiment_name, dt, r_max=6, nbins=256, c_min=1, c_mid=300, c_max=600, ):
    """Loads stimulus settings and arrays for a given experiment.

    Args:
        experiment_name (str): Name of the experiment.
        dt (float): Time step size.
        r_max (float, optional): Arena radius. Defaults to 6.
        nbins (int, optional): Number of bins for arena grid. Defaults to 256.
        c_min (int, optional): Minimum brightness in lux. Defaults to 1.
        c_mid (int, optional): Mid brightness in lux. Defaults to 300.
        c_max (int, optional): Maximum brightness in lux. Defaults to 600.

    Returns:
        tuple: (do_fish_lock, n_trials, stim_names, stim_arrays, xs, ys, ts)
            do_fish_lock (bool): If True, stimulus is fish-locked.
            n_trials (int): Number of trials per stimulus.
            stim_names (list): List of stimulus names.
            stim_arrays (list): List of stimulus arrays.
            xs (np.ndarray): X-coordinates of arena grid.
            ys (np.ndarray): Y-coordinates of arena grid.
            ts (np.ndarray): Time steps.
    """
    # Retrieve stimulus settings based on experiment_name
    if experiment_name == 'arena_locked':
        do_fish_lock = False
        t_max = 600     # s
        n_trials = 2    # number of trials per stimulus, per agent
        stim_names = [
            'control',
            'splitview_left_dark_right_bright',
            'azimuth_left_dark_right_bright',
            'azimuth_left_dark_right_bright_virtual_yes',
            'center_dark_outside_bright',
            'center_bright_outside_dark',
        ]
    elif experiment_name == 'arena_locked_quick':
        do_fish_lock = False
        t_max = 600     # s
        n_trials = 2    # number of trials per stimulus, per agent
        stim_names = [
            'control',
            'splitview_left_dark_right_bright',
            'azimuth_left_dark_right_bright',
            'center_dark_outside_bright',
            'center_bright_outside_dark',
        ]
    elif experiment_name == 'arena_locked_flip':
        do_fish_lock = False
        t_max = 600     # s
        n_trials = 1    # number of trials per stimulus, per agent
        stim_names = [
            'grey_0', 'grey_1',
            'splitview_left_dark_right_bright', 'splitview_left_bright_right_dark',
            'azimuth_left_dark_right_bright', 'azimuth_left_bright_right_dark',
            'azimuth_left_dark_right_bright_virtual_yes', 'azimuth_left_bright_right_dark_virtual_yes',
            'center_dark_outside_bright_0', 'center_dark_outside_bright_1',
            'center_bright_outside_dark_0', 'center_bright_outside_dark_1',
        ]
    elif experiment_name == 'homogeneous':
        do_fish_lock = True
        t_max = 40      # s
        n_trials = 20   # number of trials per stimulus, per agent
        brightnesses = np.concatenate([
            [10],
            np.arange(50, 601, 50),
            # np.arange(600, 1001, 100)
        ])  # lux
        stim_names = [f's00_homo{brightness:04.0f}lux' for brightness in brightnesses]
    elif experiment_name == 'brightness_choice_simple':
        do_fish_lock = True
        t_max = 270     # s
        n_trials = 20   # number of trials per stimulus, per agent
        stim_names = ['simple']
    elif experiment_name == 'contrast':
        do_fish_lock = True
        t_max = 40      # s
        n_trials = 20   # number of trials per stimulus, per agent
        ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        average_brightness = 150  # lux
        stim_names = []
        stim_counter = 0
        for ratio in ratios:
            b_max = ratio * average_brightness * 2
            b_min = (1 - ratio) * average_brightness * 2
            stim_names.append(f'left_{b_min:3.0f}lux_right_{b_max:3.0f}lux_{stim_counter}')
            stim_counter += 2  # to match experimental stimulus
    elif experiment_name == 'temporal':
        do_fish_lock = True
        t_max = 40      # s
        n_trials = 20   # number of trials per stimulus, per agent
        transition_durations = [0, 5, 10, 20]  # s: abrupt, fast, medium and slow
        fixed_dur = 10  # s
        stim_names = []
        for transition_duration_1 in transition_durations:
            for transition_duration_2 in transition_durations:
                stim_names.append(
                    f'{fixed_dur / 2:03.0f}s_0lux_{transition_duration_1:03.0f}s_up_'
                    f'{fixed_dur:03.0f}s_300lux_{transition_duration_2:03.0f}s_down_'
                    f'{fixed_dur / 2:03.0f}s_0lux')
    else:
        raise NotImplementedError(f'experiment_name = {experiment_name}')

    # Create stimulus arrays
    ts = np.arange(0, t_max + dt/2, dt)  # same for each stim_name within experiment_name
    stim_arrays = []
    xs, ys = None, None  # will only be overwritten for arena-locked stimuli
    for stim_name in stim_names:
        if do_fish_lock:
            stim_array = get_stim_fish_locked(stim_name, ts)  # (len(ts), 2)
            stim_arrays.append(stim_array)
        else:
            stim_array, xs, ys = get_stim_arena_locked(stim_name, nbins, r_max, c_min, c_mid, c_max)    # (nbins, nbins), (nbins, nbins), (nbins, nbins)
            stim_arrays.append(stim_array)

    return do_fish_lock, n_trials, stim_names, stim_arrays, xs, ys, ts


def get_stim_arena_locked(stim_name, nbins, r_max, c_min, c_mid, c_max):
    """Get brightness level for each x and y bin"""
    # Prepare arena-locked stimuli
    bins = np.linspace(-r_max, r_max, nbins)
    xs, ys = np.meshgrid(bins, bins)
    radius = np.sqrt(xs**2 + ys**2)
    angle = np.arctan2(ys, xs)  # radians: -pi to pi
    px = np.ones_like(radius) * np.nan  # must be an array of floats

    # Arena locked stimuli ####################################################
    if stim_name == 'control' or stim_name == 'grey_0' or stim_name == 'grey_1':
        px[radius <= r_max] = c_mid
    elif stim_name == 'splitview_left_dark_right_bright' or stim_name == 'splitview_left_dark_right_bright_virtual':
        px[xs < 0] = c_min
        px[xs >= 0] = c_mid
        px[radius > r_max] = np.nan
    elif stim_name == 'splitview_left_bright_right_dark' or stim_name == 'splitview_left_bright_right_dark_virtual':
        px[xs < 0] = c_mid
        px[xs >= 0] = c_min
        px[radius > r_max] = np.nan
    elif stim_name == 'azimuth_left_dark_right_bright' or stim_name == 'azimuth_left_dark_right_bright_virtual' or stim_name == 'azimuth_left_dark_right_bright_virtual_yes':
        px = (c_mid - c_min) * (np.pi - np.abs(angle)) / np.pi + c_min
        px[radius > r_max] = np.nan
    elif stim_name == 'azimuth_left_bright_right_dark' or stim_name == 'azimuth_left_bright_right_dark_virtual' or stim_name == 'azimuth_left_bright_right_dark_virtual_yes':
        px = (c_mid - c_min) * (np.abs(angle)) / np.pi + c_min
        px[radius > r_max] = np.nan
    elif stim_name == 'center_dark_outside_bright' or stim_name == 'center_dark_outside_bright_0' or stim_name == 'center_dark_outside_bright_1' or stim_name == 'center_dark_outside_bright_virtual':
        px = (c_max - c_min) * radius / r_max + c_min
        px[radius > r_max] = np.nan
    elif stim_name == 'center_bright_outside_dark' or stim_name == 'center_bright_outside_dark_0' or stim_name == 'center_bright_outside_dark_1' or stim_name == 'center_bright_outside_dark_virtual':
        px = (c_mid - c_min) * (r_max - radius) / r_max + c_min
        px[radius > r_max] = np.nan
    else:
        raise UserWarning(f"\033[91m{stim_name} not recognised\033[0m")

    return px, xs, ys   # (nbins, nbins), (nbins, nbins), (nbins, nbins)


def get_stim_fish_locked(stim_name, ts):
    """Generates fish-locked stimulus arrays for a given stimulus name and time steps.

    This function creates left and right eye brightness arrays over time for fish-locked
    stimuli, based on the provided stimulus name and time vector.

    Args:
        stim_name (str): Name of the stimulus.
        ts (np.ndarray): Array of time steps.

    Returns:
        np.ndarray: Array of shape (len(ts), 2) with left and right eye brightness values.

    Raises:
        NotImplementedError: If the stimulus type is not implemented.
    """
    # Prepare fish-locked stimuli: define values over time
    b_left = np.ones_like(ts) * np.nan
    b_right = np.ones_like(ts) * np.nan

    if stim_name == 'control':
        b_left[:] = 300
        b_right[:] = 300
    elif stim_name == 'brightness_choice':
        b_left[ts <= 30] = 10
        b_right[ts <= 30] = 300
        b_left[ts >= 30] = 300
        b_right[ts >= 30] = 10
    elif stim_name == 'simple':
        b_left[:] = 300
        b_right[:] = 300
        b_left[(60 <= ts) & (ts < 90)] = 10
        b_left[(120 <= ts) & (ts < 180)] = 10
        b_left[(210 <= ts) & (ts < 240)] = 10
        b_right[(30 <= ts) & (ts < 60)] = 10
        b_right[(120 <= ts) & (ts < 150)] = 10
        b_right[(180 <= ts) & (ts < 240)] = 10
    elif 's00_homo' in stim_name:
        # Homogeneous #########################################################
        brightness = int(stim_name.replace('s00_homo', '').replace('lux', ''))  # lux
        b_left = np.ones_like(ts) * brightness
        b_right = np.ones_like(ts) * brightness
    elif 'left_' in stim_name:
        # Contrast ############################################################
        left_brightness = int(stim_name.split('lux_right')[0].replace('left_', ''))  # lux
        right_brightness = int(stim_name.split('lux_right')[-1].split('lux_')[0].replace('_', ''))  # lux
        b_left = np.ones_like(ts) * left_brightness
        b_right = np.ones_like(ts) * right_brightness
    elif 'up' in stim_name:
        # Temporal ############################################################
        # transition_1 = int(stim_name.split('0lux_'))  # TODO
        # Start at 0 lux
        b_left = np.ones_like(ts) * 0
        b_right = np.ones_like(ts) * 0
        raise NotImplementedError

    return np.stack([b_left, b_right], axis=-1)  # (len(ts), 2)


# #############################################################################
# Helper functions
# #############################################################################
def print_progress(progress_dict):
    """Prints simulation progress for all agents at regular intervals.

    Continuously prints the status of each agent from the progress dictionary
    every 10 seconds until all agents are marked as done.

    Args:
        progress_dict (dict): Dictionary mapping agent indices to status strings.

    Returns:
        None
    """
    while True:
        print(f"{datetime.datetime.now():%H:%M:%S} Status:")
        for agent_index, status in progress_dict.items():
            print(f"{status}")
        time.sleep(10)  # Update every 10 seconds
        if all("done" in status for status in progress_dict.values()):
            break


def convert_agent_genotype(agent_genotype_name, age):
    """Converts agent genotype name and age to a formatted agent name.

    Args:
        agent_genotype_name (str): Genotype name.
        age (int): Age of the agent.

    Returns:
        tuple: (agent_genotype_name, agent_name)
            agent_genotype_name (str): The original genotype name.
            agent_name (str): Genotype name with age appended, e.g. 'genotype_05dpf'.
    """
    agent_name = agent_genotype_name + f"_{age:02d}dpf"
    return agent_genotype_name, agent_name

    # key_map = {
    #     'percentage_turns': 'pt',
    #     'percentage_left': 'pl',
    #     'turn_angle': 'a',
    #     'total_duration': 't',
    #     'total_distance': 's'
    # }
    #
    # value_map = {
    #     'blind': 'B',
    #     'spatial_temporal': 'ST',
    #     'st_full_model': 'ST_full',
    #     'st_pleft_model': 'ST_pleft',
    #     'azimuth_virtual': 'AV',
    #     'azimuth': 'A',
    #     'contrast': 'C',
    #     'temporal': 'T',
    #     'st_A': 'ST_A',
    #     'st_C': 'ST_C',
    #     'st_D': 'st_D',
    #     'st_D_C': 'st_D_C',
    #     'st_A_D_C': 'st_A_D_C',
    #     'st_A_AD_D_C_DA': 'st_full',
    #     'st_a_d_c_hom': 'st_A_D_C_hom',
    #     'st_d_c_hom': 'st_D_C_hom',
    # }
    #
    # parts = [f"{key_map[k]}{value_map[v]}" for k, v in agent_genotype_dict.items()]
    #
    # agent_genotype_name = "model_" + "_".join(parts)
    # agent_name = agent_genotype_name + f"_{age:02d}dpf"
    # return agent_genotype_name, agent_name

