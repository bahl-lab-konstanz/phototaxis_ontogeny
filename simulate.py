"""Script to run simulations for a given experiment and agent.

This script sets up and runs agent-based simulations for specified experimental
conditions and agent genotypes. It supports multiprocessing for parallel
simulations and sends notifications upon completion.

Note:
    - User must specify simulation, stimulus, and agent settings.
    - Slack notifications require valid configuration.
"""

import datetime
import itertools
import multiprocessing
import numpy as np

from settings.general_settings import *
from utils.slack_helper import send_slack_message
from utils.simulate_helpers import load_stimulus, task_run_agent, convert_agent_genotype
from utils.prepare_data_wrapper import prepare_data_wrapper

if __name__ == '__main__':
    # #########################################################################
    # User input
    # #########################################################################
    # Simulation settings
    main_seed = 42  # seed
    dt = 1 / 60  # s, time step size
    agent_IDs = np.arange(0, 96)  # agent IDs to simulate

    # # For debugging:
    # n_processes = 1
    # agent_IDs = [0]

    # Stimulus settings
    from settings.stim_arena_locked import *
    input_data_date = '250520'

    # Your Slack name here
    slack_receiver = 'Max Capelle'

    # Agent settings
    agent_ages = [5, 27]  # dpf
    agent_genotype_dicts = {
        'A_DC_A_A_A_wCx10': {
            'percentage_turns': 'azimuth_virtual',
            'percentage_left': 'st_d_c_x10',
            'turn_angle': 'azimuth_virtual',
            'total_duration': 'azimuth_virtual',
            'total_distance': 'azimuth_virtual',
        },
        'A_DC_A_A_A_wCx5': {
            'percentage_turns': 'azimuth_virtual',
            'percentage_left': 'st_d_c_x5',
            'turn_angle': 'azimuth_virtual',
            'total_duration': 'azimuth_virtual',
            'total_distance': 'azimuth_virtual',
        },
        'A_DC_A_A_A_wCx20': {
            'percentage_turns': 'azimuth_virtual',
            'percentage_left': 'st_d_c_x20',
            'turn_angle': 'azimuth_virtual',
            'total_duration': 'azimuth_virtual',
            'total_distance': 'azimuth_virtual',
        },
        'A_DC_A_A_A': {
            'percentage_turns': 'azimuth_virtual',
            'percentage_left': 'st_d_c',
            'turn_angle': 'azimuth_virtual',
            'total_duration': 'azimuth_virtual',
            'total_distance': 'azimuth_virtual',
        },
    }
    r_view = 2  # cm, range of view (radius)

    for (agent_key, agent_genotype_dict), agent_age in itertools.product(agent_genotype_dicts.items(), agent_ages):
        if agent_age == 5:
            age_category = 'larva'
        else:
            age_category = 'juvie'

        # Set agent name based on model and age
        agent_genotype_name, agent_name = convert_agent_genotype(agent_key, agent_age)

        # Paths
        path_to_input_folder = path_to_sim_folder.joinpath('input_data', input_data_date)
        path_to_raw_data_folder = path_to_sim_folder.joinpath('raw_data', agent_name, path_name)
        path_to_raw_data_folder.mkdir(parents=True, exist_ok=True)
        if not path_to_input_folder.exists():
            raise UserWarning(f'path_to_input_folder does not exist: {path_to_input_folder}')

        timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")
        n_agents = len(agent_IDs)

        # #############################################################################
        # Prepare simulation
        # #############################################################################
        # Load stimulus settings
        do_fish_lock, n_trials, stim_names, stim_arrays, xs, ys, ts = load_stimulus(experiment_name, dt)

        # Run simulation ##############################################################
        # Print settings
        print(
            f"\033[1mSimulation\033[0m"
            f"\n\texperiment:       {experiment_name}"
            f"\n\tn_processes:      {n_processes}"
            f"\n\tsimulation time:  {np.max(ts) * n_trials * len(stim_names):.0f}s per agent"
            f"\n\tn_agents:         {len(agent_IDs)}"
            f"\n\tn_trials:         {n_trials}"
            f"\n\tto:               {path_to_raw_data_folder.joinpath(timestamp + '_agent**')}"
            f"\n\tagent name:       {agent_name if agent_name else 'None'}"
        )

        with multiprocessing.Manager() as manager:
            # Create arguments ############################################################
            progress_dict = manager.dict({})

            # Create argument for each agent
            args = []
            for agent_index in agent_IDs:
                folder_name = f'{timestamp}_agent{agent_index:03d}'

                args.append([
                    main_seed, path_to_raw_data_folder,
                    do_fish_lock, n_trials, stim_names, stim_arrays, xs, ys,
                    ts, dt, r_view,
                    agent_index, agent_age, agent_genotype_name, agent_genotype_dict, folder_name,
                    path_to_input_folder, progress_dict
                ])

            # Create a pool and start the tasks ###########################################
            with multiprocessing.Pool(processes=n_processes) as pool:
                print(f"{datetime.datetime.now():%H:%M:%S} Start simulation")
                pool.map(task_run_agent, args)

                # Close and join the pool
                pool.close()
                pool.join()

        # Send slack message when done
        print(
            "Simulation \033[92mdone\033[0m\n"
            f"\t{experiment_name}/{path_name} {agent_name} \033[94mn={n_agents}\033[0m"
        )
        send_slack_message(to=slack_receiver,
                           message=f'simulation done for {experiment_name}/{path_name} {agent_name}: n={n_agents}')

        # #####################################################################
        # Prepare data directly after simulation of this genotype
        # #####################################################################
        prepare_data_wrapper(
            age_category='simulations', agent_name=agent_name,
            experiment_name=path_name,
            flip_dict=flip_dict, split_dict=split_dict, label_dict=label_dict,
            included_stim_names=None,  # stim_names are different before and after flipping
            rolling_window=rolling_window, resampling_window=resampling_window,
            do_tracking=do_tracking, do_event=do_event,
            slack_receiver=slack_receiver
        )
