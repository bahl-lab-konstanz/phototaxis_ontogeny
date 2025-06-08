# Standard library imports
import datetime

# Third party library imports
import h5py
import pandas as pd

# Local library imports
from shared_modules_across_frameworks.modules.general_utilities import compress_hdf5_container
from analysis_helpers.analysis.multifish_modules.preprocess_data import PreprocessData

# #############################################################################
# Preprocessing settings
# #############################################################################
opts_dict = dict({})
opts_dict["head_embedded_real_fish"] = dict({})
opts_dict["freely_swimming_intelligent_agent"] = dict({})
opts_dict["freely_swimming_real_fish"] = dict({})
opts_dict["tail_real_fish"] = dict({})
opts_dict["head_real_fish"] = dict({})
opts_dict["stimulus"] = dict({})
opts_dict["general"] = dict({})

opts_dict["general"]["compression"] = 9

# Align stimuli based on the beginning of the stimulus + (end is beginning + stimulus duration)
opts_dict["general"]["verbose"] = 0
opts_dict["general"]["recompute_windowed_variances"] = 0
opts_dict["general"]["stimulus_alignment_strategy"] = 0
opts_dict["general"]["stimulus_alignment_start_delta_t"] = 0  # You can use -10 to draw into the previous trial, for example if this was gray anyway
opts_dict["general"]["stimulus_alignment_end_delta_t"] = 0  # You canuse +10 to draw into the next trial, if this was all gray, for example
opts_dict["stimulus"]["interpolation_dt"] = 1 / 60

# Real fish
opts_dict["freely_swimming_real_fish"]["interpolation_dt"] = 1 / 90
opts_dict["freely_swimming_real_fish"]["swim_event_detection_variance_window"] = 0.05
opts_dict["freely_swimming_real_fish"]["bout_detection_start_threshold"] = 5  # 7.5 for juvies
opts_dict["freely_swimming_real_fish"]["bout_detection_time_required_above_start_threshold_to_be_valid_event"] = 0.020
opts_dict["freely_swimming_real_fish"]["bout_detection_end_threshold"] = 1.5  # 5 for juvies
opts_dict["freely_swimming_real_fish"]["bout_detection_time_required_below_end_threshold_to_be_valid_event"] = 0.050
opts_dict["freely_swimming_real_fish"]["bout_detection_maximal_length_of_event"] = 4
opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t0_relative_to_event_start"] = -0.0600
opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t1_relative_to_event_start"] = -0.0300
opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t2_relative_to_event_start"] = -0.0600
opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t3_relative_to_event_start"] = -0.0300
opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t0_relative_to_event_end"] = 0.0100
opts_dict["freely_swimming_real_fish"]["swim_event_feature_extraction_t1_relative_to_event_end"] = 0.0500

# Agents
opts_dict["freely_swimming_intelligent_agent"]["interpolation_dt"] = 1 / 60
opts_dict["freely_swimming_intelligent_agent"]["swim_event_detection_variance_window"] = 0.05
opts_dict["freely_swimming_intelligent_agent"]["bout_detection_start_threshold"] = 0.01
opts_dict["freely_swimming_intelligent_agent"]["bout_detection_time_required_above_start_threshold_to_be_valid_event"] = 0.020
opts_dict["freely_swimming_intelligent_agent"]["bout_detection_end_threshold"] = 0.005
opts_dict["freely_swimming_intelligent_agent"]["bout_detection_time_required_below_end_threshold_to_be_valid_event"] = 0.050
opts_dict["freely_swimming_intelligent_agent"]["bout_detection_maximal_length_of_event"] = 4
opts_dict["freely_swimming_intelligent_agent"]["swim_event_feature_extraction_t0_relative_to_event_start"] = -0.0600
opts_dict["freely_swimming_intelligent_agent"]["swim_event_feature_extraction_t1_relative_to_event_start"] = -0.0300
opts_dict["freely_swimming_intelligent_agent"]["swim_event_feature_extraction_t2_relative_to_event_start"] = -0.0600
opts_dict["freely_swimming_intelligent_agent"]["swim_event_feature_extraction_t3_relative_to_event_start"] = -0.0300
opts_dict["freely_swimming_intelligent_agent"]["swim_event_feature_extraction_t0_relative_to_event_end"] = 0.0100
opts_dict["freely_swimming_intelligent_agent"]["swim_event_feature_extraction_t1_relative_to_event_end"] = 0.0500

# Agent specific settings
opts_dict_larva = opts_dict.copy()
opts_dict_juvie = opts_dict.copy()
opts_dict_agent = opts_dict.copy()
opts_dict_larva["freely_swimming_real_fish"]["bout_detection_start_threshold"] = 5
opts_dict_larva["freely_swimming_real_fish"]["bout_detection_end_threshold"] = 1.5
opts_dict_juvie["freely_swimming_real_fish"]["bout_detection_start_threshold"] = 7.5
opts_dict_juvie["freely_swimming_real_fish"]["bout_detection_end_threshold"] = 5


class PreprocessAllData:
    def __init__(self, path_to_experiment_folders, opts_dict, glob_str="**/[!._]*.hdf5"):
        self.path_to_experiment_folders = path_to_experiment_folders
        self.opts_dict = opts_dict
        self.glob_str = glob_str

    def run(self):
        print("Start PreprocessAllData with the following settings: ")
        print(f"\tpath_to_experiment_folders: {self.path_to_experiment_folders}")
        print(f"\tglob_str: {self.glob_str}")

        # Find all experiment folders
        data_file_paths = list(sorted(self.path_to_experiment_folders.glob(self.glob_str)))
        n_data_files = len(data_file_paths)

        # Loop over all data files
        for file_num, data_file_path in enumerate(data_file_paths):
            print(f"{file_num + 1: 4d}/{n_data_files} \t{datetime.datetime.now().strftime('%H:%M:%S')} {data_file_path.stem} ")

            # Open the hdf5 file to find all repeats
            with h5py.File(data_file_path, 'r') as f_hdf5:
                repeats = [key for key in f_hdf5.keys() if 'repeat' in key]
                if len(repeats) == 0:
                    print("\033[91m" + "No repeats found in file:" + "\033[0m", str(data_file_path))
                    continue

            # Loop over repeats
            for dset_prefix in repeats:
                experiment_repeat = int(dset_prefix.split('repeat')[-1])
                print(f"\t\t{dset_prefix} ", end='')

                prepd = PreprocessData(dset_prefix=dset_prefix,
                                       experiment_repeat=experiment_repeat,
                                       container_file_path=data_file_path,
                                       opts_dict=self.opts_dict)
                prepd.run()

            compress_hdf5_container(data_file_path)
            print("\033[92m" + "done" + "\033[0m")


def get_experiment_IDs(path_to_experiment_folders, glob_str="**/[!._]*.hdf5"):
    # Find all experiment folders
    data_file_paths = sorted(path_to_experiment_folders.glob(glob_str))

    # Loop over all data files
    exp_ID_dict = {}
    for file_num, data_file_path in enumerate(data_file_paths):

        # Open the hdf5 file and extract meta data
        with h5py.File(data_file_path, 'r') as f_hdf5:
            setup_index = f_hdf5.attrs["setup_index"]
            arena_index = f_hdf5.attrs["arena_index"]
            experiment_ID = f_hdf5.attrs["experiment_ID"]
            fish_genotype = f_hdf5.attrs["fish_genotype"]
            fish_age = f_hdf5.attrs["fish_age"]
            folder_name = f_hdf5.attrs["folder_name"]

        print(f"folder_name: {folder_name}, experiment_ID: {experiment_ID: 5d}, fish_genotype: {fish_genotype}, fish_age: {fish_age}")

        exp_ID_dict[experiment_ID] = {
            'folder_name': folder_name,
            'setup_index': setup_index,
            'arena_index': arena_index,
            'fish_genotype': fish_genotype,
            'fish_age': fish_age
        }

    return exp_ID_dict


def rewrite_experiment_IDs(path_to_experiment_folders, glob_str="**/[!._]*.hdf5", add_exp_ID=0):
    # Find all experiment folders
    data_file_paths = sorted(path_to_experiment_folders.glob(glob_str))

    # Loop over all data files
    for file_num, data_file_path in enumerate(data_file_paths):

        # Open the hdf5 file and update meta data
        with h5py.File(data_file_path, 'r+') as f_hdf5:
            experiment_ID = f_hdf5.attrs["experiment_ID"]
            folder_name = f_hdf5.attrs["folder_name"]

            # Update experiment_ID
            f_hdf5.attrs["experiment_ID"] = experiment_ID + add_exp_ID

        print(f"folder_name: {folder_name}, experiment_ID: {experiment_ID: 5d} -> {experiment_ID + add_exp_ID: 5d}")


if __name__ == "__main__":
    from settings.general_settings import *

    # Experiment settings
    experiment_name = "arena_locked_preprocessing"
    age_category = 'larvae'  # 'larvae' or 'juveniles' or 'agents'
    agent_name = ''  # agent name or ''

    # Path to experiment data
    if age_category == 'larvae':
        path_to_experiment_folders = path_to_larva_server_folder.joinpath(experiment_name)
    elif age_category == 'juveniles':
        path_to_experiment_folders = path_to_juvie_server_folder.joinpath(experiment_name)
    elif age_category == 'agents':
        path_to_experiment_folders = path_to_agents_server_folder.joinpath(agent_name).joinpath(experiment_name)
    else:
        raise NotImplementedError(f"age_category '{age_category}'")

    # # Rewrite experiment IDs
    # exp_ID_dict = get_experiment_IDs(
    #     path_to_experiment_folders,
    #     glob_str="2024-05-23_14-57*/[!._]*.hdf5",
    # )
    # rewrite_experiment_IDs(
    #     path_to_experiment_folders,
    #     glob_str="2024-05-23_14-57*/[!._]*.hdf5",
    #     add_exp_ID=1,
    # )

    # Preprocess data
    prepd = PreprocessAllData(
        path_to_experiment_folders=path_to_experiment_folders,
        opts_dict=opts_dict,
        glob_str="2025-02-26_11-51-04_setup*_arena*/[!._]*.hdf5",
    )
    prepd.run()