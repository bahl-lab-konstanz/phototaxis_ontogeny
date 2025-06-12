

# Local library imports
from settings.general_settings import *
from .preprocess_utils import *
from .analysis_start import AnalysisStart
from .combine_data import CombineData
from .prescreening import Prescreening
from .sanity_check import SanityCheck
from .slack_helper import send_slack_message


def prepare_data_wrapper(
    age_category, agent_name,
    experiment_name, flip_dict, split_dict, label_dict,
    included_stim_names,  # TODO: do we actually need this?
    rolling_window=None, resampling_window=None,
    do_preprocess=False,
    do_tracking=False, do_event=True,
    do_bootstrap=False, do_median_df=True, do_rolling_df=False,
    slack_receiver=None
):

    # Path to store prepared data
    path_to_stim_folder = path_to_main_data_folder.joinpath(experiment_name)

    # Path to experiment data
    if age_category == 'larva':
        path_to_experiment_folders = path_to_larva_server_folder.joinpath(experiment_name)
        path_to_agent_folder = path_to_stim_folder.joinpath('larva')
    elif age_category == 'juvie':
        path_to_experiment_folders = path_to_juvie_server_folder.joinpath(experiment_name)
        path_to_agent_folder = path_to_stim_folder.joinpath('juvie')
    elif age_category == 'agents':
        path_to_experiment_folders = path_to_agents_server_folder.joinpath(agent_name, experiment_name)
        path_to_agent_folder = path_to_stim_folder.joinpath(agent_name)
    elif age_category == 'simulations':
        path_to_experiment_folders = path_to_sim_folder.joinpath('raw_data', agent_name, experiment_name)
        path_to_agent_folder = path_to_stim_folder.joinpath(agent_name)
    else:
        raise NotImplementedError(f"age_category '{age_category}'")

    path_to_input_file = path_to_agent_folder.joinpath('combined_data.hdf5')
    path_to_analysed_file = path_to_agent_folder.joinpath('analysed_data.hdf5')
    path_to_agent_folder.mkdir(exist_ok=True, parents=True)

    # Check paths
    if not path_to_experiment_folders.exists():
        print(f"\033[91mpath_to_experiment_folders cannot be found:\033[0m\n\t{path_to_experiment_folders}")

    # Age category specific settings
    if age_category == 'larva':
        max_contour_area = 12_000  # pixels
        max_average_speed = 1.5  # cm/s
        drop_trial_if_fraction_of_abs_orientation_change = 0.10
        opts_dict = opts_dict_larva
    elif age_category == 'juvie':
        max_contour_area = 16_000  # pixels
        max_average_speed = 3  # cm/s
        drop_trial_if_fraction_of_abs_orientation_change = 0.05
        opts_dict = opts_dict_juvie
    elif age_category == 'agents':
        max_contour_area = None
        max_average_speed = 3  # cm/s
        drop_trial_if_fraction_of_abs_orientation_change = 0.10
        opts_dict = opts_dict_agent
    elif age_category == 'simulations':
        max_contour_area = None
        max_average_speed = 3
        drop_trial_if_fraction_of_abs_orientation_change = 0.10
        opts_dict = opts_dict_agent
    else:
        raise NotImplementedError(f"age_category '{age_category}'")

    # Print settings
    print(
        f"\033[1mPrepare data\033[0m"
        f"\n\texperiment:   {experiment_name}"
        f"\n\ttracking:     {do_tracking}"
        f"\n\tevent:        {do_event}"
        f"\n\tto:           {path_to_agent_folder}"
        f"\n\tage category: {age_category}"
        f"\n\tagent name:   {agent_name if agent_name else 'None'}"
        f"\n\tpreprocess:   {do_preprocess}"
    )

    # #############################################################################
    # Preprocess data
    # #############################################################################
    if do_preprocess:
        # Preprocess data
        prepd = PreprocessAllData(
            path_to_experiment_folders=path_to_experiment_folders,
            opts_dict=opts_dict
        )
        prepd.run()

    # #############################################################################
    # Start Preparation
    # #############################################################################
    # Prescreening ################################################################
    if age_category == 'larva':
        prescreening = Prescreening(
            path_to_experiment_folders=path_to_experiment_folders,
            path_to_analysed_folder=path_to_agent_folder,
            min_percentage_grating_left=0.5,
        )
        # TODO: change after debugging
        included_experiment_IDs, excluded_experiment_IDs = prescreening.run()
        # excluded_experiment_IDs = None
    else:
        excluded_experiment_IDs = None

    # Combine Data ################################################################
    combineData = CombineData(
        path_to_experiment_folders=path_to_experiment_folders,
        path_to_combined_data_file=path_to_input_file,
        combine_bout_data=do_event,
        combine_tracking_data=do_tracking,
        prescreening=False,
        excluded_experiment_IDs=excluded_experiment_IDs,
        overwrite_output_file=False
    )
    # TODO: change after debugging
    combined_event_df = combineData.run()
    # combined_event_df = pd.read_hdf(path_to_input_file, key="all_bout_data_pandas")

    # Sanity Check ################################################################
    if do_event:
        sanity_check = SanityCheck(
            path_to_combined_data_file=path_to_input_file,
            max_interbout_interval=10,  # Set to None if this should be ignored
            max_contour_area=max_contour_area,  # Set to None if this should be ignored
            max_average_speed=max_average_speed,  # Set to None if this should be ignored
            max_abs_orientation_change=150,  # Set to None if this should be ignored
            max_radius=None,  # fraction of arena radius, None here because we exclude the outer 1 cm later in analysis_start
            drop_trial_if_fraction_of_abs_orientation_change=drop_trial_if_fraction_of_abs_orientation_change,
            plot_statistics=True,
            do_overwrite=True,
        )
        # TODO: change after debugging
        event_df = sanity_check.run()
        # event_df = pd.read_hdf(path_to_input_file, key="all_bout_data_sanity_checked_pandas")

    # Analysis Start ##############################################################
    analysis_start = AnalysisStart(
        path_to_analysed_file=path_to_analysed_file,
    )

    if do_event:
        # Create event_df
        trial_event_df = analysis_start.get_trial_df(
            event_df,
            flip_dict=flip_dict,
            split_dict=split_dict,
            label_dict=label_dict,
            included_stim_names=included_stim_names,
            data_type='bout',
        )
        # Compute n-number
        n = trial_event_df.groupby([
            'experiment_ID', 'fish_or_agent_name', 'setup_index',
            'fish_genotype', 'fish_age', 'experiment_repeat',
        ]).ngroups

        # Aggregate over fish #################################################
        if do_bootstrap:
            print(f'Bootstrapping')

            # Get unique experiment_IDs
            unique_ids = trial_event_df.index.unique('experiment_ID')
            n_output = len(unique_ids)
            n_input = 16  # int(n_output/3)  # Number of experiment_IDs to sample

            # Create a list to store the bootstrapped data
            bootstrapped_df_list = []
            for new_id in range(n_output):
                # Randomly sample `n_input` experiment_IDs with replacement
                sampled_ids = rng.choice(unique_ids, size=n_input, replace=True)

                # Select data for the sampled experiment_IDs
                sampled_data = trial_event_df.loc[
                    trial_event_df.index.get_level_values('experiment_ID').isin(sampled_ids)]

                # Assign new index values to sampled data
                sampled_data = sampled_data.reset_index()
                # # Assign the new experiment_ID
                sampled_data['experiment_ID'] = new_id
                # # Assign first value
                sampled_data['fish_genotype'] = sampled_data['fish_genotype'].unique()[0]
                sampled_data['fish_age'] = sampled_data['fish_age'].unique()[0]
                sampled_data['fish_or_agent_name'] = sampled_data['fish_or_agent_name'].unique()[0]
                sampled_data['folder_name'] = sampled_data['folder_name'].unique()[0]
                # # Assign 0 (these are not relevant for the bootstrapped data)
                sampled_data['experiment_repeat'] = 0
                sampled_data['arena_index'] = 0
                sampled_data['setup_index'] = 0

                bootstrapped_df_list.append(sampled_data)

            # Combine all sampled data
            bootstrapped_df = pd.concat(bootstrapped_df_list)
            bootstrapped_df = bootstrapped_df.set_index(trial_event_df.index.names)

            trial_event_df = bootstrapped_df  # Overwrite the original event_df

        if do_median_df:
            median_df = analysis_start.get_median_df(
                trial_df=trial_event_df,
                resampling_window=resampling_window,
            )

        if do_rolling_df:
            rolled_df = analysis_start.roll_over_trials(
                trial_df=trial_event_df,
                rolling_window=rolling_window,
                resampling_window=resampling_window,
            )

        # # Create stim_df
        # stim_df = analysis_start.average_over_fish(fish_df=fish_df)

    if do_tracking:
        # Create trial_tracking_df
        tracking_df = pd.read_hdf(path_to_input_file, key="all_freely_swimming_tracking_data_pandas")
        trial_tracking_df = analysis_start.get_trial_df(
            tracking_df,
            flip_dict=flip_dict,
            split_dict=split_dict,
            label_dict=label_dict,
            included_stim_names=included_stim_names,
            data_type='tracking',
        )
        # Compute n-number
        n = trial_tracking_df.groupby([
            'experiment_ID', 'fish_or_agent_name', 'setup_index',
            'fish_genotype', 'fish_age', 'experiment_repeat',
        ]).ngroups

    # #############################################################################
    # Preparation done! Happy. Jeeejjj
    # #############################################################################
    print(
        "Prepare data \033[92mdone\033[0m\n"
        f"\t{experiment_name} {age_category} {agent_name} \033[94mn={n}\033[0m"
    )
    if slack_receiver:
        send_slack_message(to=slack_receiver, message=f'prepare_data done for {experiment_name} {age_category} {agent_name}: n={n}')


if __name__ == '__main__':
    # #########################################################################
    # User input
    # #########################################################################
    # Experiment settings
    from settings.stim_arena_locked import *

    # Fish/agent settings
    age_category = 'juvie'  # 'larva' or 'juvie' or 'agents' or 'simulations'
    agent_name = ''  # agent name or ''
    do_preprocess = False  # True or False

    # Your Slack name here
    slack_receiver = 'Max Capelle'

    prepare_data_wrapper(
        age_category=age_category, agent_name=agent_name,
        experiment_name=experiment_name,
        flip_dict=flip_dict, split_dict=split_dict, label_dict=label_dict,
        included_stim_names=stim_names,
        rolling_window=rolling_window, resampling_window=resampling_window,
        do_tracking=do_tracking, do_event=do_event, do_bootstrap=do_bootstrap, do_preprocess=do_preprocess,
        slack_receiver=slack_receiver
    )