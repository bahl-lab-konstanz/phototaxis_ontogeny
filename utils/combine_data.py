import h5py
import pandas as pd
from pathlib import Path
from datetime import datetime


class CombineData:
    def __init__(self, path_to_experiment_folders,
                 path_to_combined_data_file,
                 combine_bout_data=False,
                 combine_tracking_data=True,
                 prescreening=False,
                 excluded_data_file_names=[],
                 excluded_experiment_IDs=None,
                 overwrite_output_file=False,
                 compression=9):
        """
        Initialize the CombineData class for combining experimental data files.

        Args:
            path_to_experiment_folders (str or Path): Path to the folder containing
                experiment data files.
            path_to_combined_data_file (str or Path): Path to the output combined
                data file.
            combine_bout_data (bool, optional): Whether to combine bout data.
                Defaults to False.
            combine_tracking_data (bool, optional): Whether to combine tracking
                data. Defaults to True.
            prescreening (bool, optional): Whether to use prescreening data files.
                Defaults to False.
            excluded_data_file_names (list, optional): List of data file names to
                exclude from combining. Defaults to empty list.
            excluded_experiment_IDs (list or None, optional): List of experiment
                IDs to exclude. Defaults to None.
            overwrite_output_file (bool, optional): Whether to overwrite the
                existing combined data file. Defaults to False.
            compression (int, optional): Compression level for HDF5 output.
                Defaults to 9.

        """
        self.path_to_experiment_folders = path_to_experiment_folders
        self.path_to_combined_data_file = path_to_combined_data_file
        self.combine_bout_data = combine_bout_data
        self.combine_tracking_data = combine_tracking_data
        self.prescreening = prescreening
        self.excluded_data_file_names = excluded_data_file_names
        self.excluded_experiment_IDs = excluded_experiment_IDs
        self.overwrite_output_file = overwrite_output_file
        self.compression = compression

    def run(self):
        """
        Combine data from multiple experiment files into a single HDF5 file.

        This method searches for experiment data files, loads and concatenates
        bout and/or tracking data, and saves the combined data to the specified
        output file. It skips files that have already been combined or are
        excluded by name or experiment ID. The method also manages file
        attributes to track which files have been processed.

        Returns:
            pd.DataFrame: The combined bout data DataFrame if available, otherwise
                None or a warning.
        """
        print("Start CombineData")
        path_to_experiment_folders = Path(self.path_to_experiment_folders)
        path_to_combined_data_file = Path(self.path_to_combined_data_file)
        print(f"\t{path_to_combined_data_file}")

        if not path_to_experiment_folders.exists():
            return UserWarning(f'path_to_experiment_folders does not exist: {path_to_experiment_folders}')

        if self.prescreening:
            input_files_search_string = "**/*_prescreening.hdf5"
        else:
            input_files_search_string = "**/*[!_prescreening].hdf5"

        # Create the folder in which the combined datafile should reside
        path_to_combined_data_file.parent.mkdir(parents=True, exist_ok=True)

        # Create empty dataframes
        # combined_df_bout_data = pd.DataFrame()
        # combined_df_freely_swimming_tracking_data = pd.DataFrame()
        # combined_df_head_embedded_tracking_data = pd.DataFrame()
        combined_df_bout_data_list = []
        combined_df_freely_swimming_tracking_data_list = []
        combined_df_head_embedded_tracking_data_list = []

        # Load old dataframe if existing and select *new* folder paths
        previously_combined_data_files = []
        if not self.overwrite_output_file and path_to_combined_data_file.exists():

            with h5py.File(path_to_combined_data_file, 'r') as f_hdf5:
                if "all_bout_data_pandas" in f_hdf5.keys():
                    df = pd.read_hdf(path_to_combined_data_file, "all_bout_data_pandas")
                    combined_df_bout_data_list.append(df)

                if "all_freely_swimming_tracking_data_pandas" in f_hdf5.keys():
                    df = pd.read_hdf(path_to_combined_data_file, "all_freely_swimming_tracking_data_pandas")
                    combined_df_freely_swimming_tracking_data_list.append(df)

                if "all_head_embedded_tracking_data_pandas" in f_hdf5.keys():
                    df = pd.read_hdf(path_to_combined_data_file, "all_head_embedded_tracking_data_pandas")
                    combined_df_head_embedded_tracking_data_list.append(df)

                # There is a list in the combined datafile that shows what has already been combined, so one does not need to do this again
                if "combined_data_file_names" in f_hdf5.attrs:
                    # Load the dataset of filename. Make sure they are converted into the right string format.
                    previously_combined_data_files = list(f_hdf5.attrs["combined_data_file_names"])

                    print(f"\tPreviously combined data files that will be ignored ({len(previously_combined_data_files)}):", previously_combined_data_files)

        # Make a list of new data file path.
        data_file_paths = []
        for data_file_path in path_to_experiment_folders.glob(input_files_search_string):
            # Ignore all hdf5 files in the parent directory. Because here one would like to store the combined hdf5
            if data_file_path.parent == path_to_experiment_folders:
                continue

            if '._' == data_file_path.stem[:2]:
                # Ignore 'hidden' files
                continue

            if data_file_path.name not in previously_combined_data_files and \
                    data_file_path.name not in self.excluded_data_file_names:
                data_file_paths.append(data_file_path)

        # Check if any new input files are found
        if len(data_file_paths) == 0:
            print(f"\t\033[94mNo new data files found\033[0m")
            combined_df_bout_data = pd.read_hdf(path_to_combined_data_file, key="all_bout_data_pandas")
            return combined_df_bout_data

        # Loop over new HDF5 folder_paths
        for file_num, data_file_path in enumerate(data_file_paths.copy()):
            # Try to open file
            try:
                with h5py.File(data_file_path, 'r') as f_hdf5:
                    pass
            except Exception as e:
                print(f"{file_num + 1: 4d}/{len(data_file_paths)} \t{datetime.now().strftime('%H:%M:%S')} {data_file_path.stem} ", end='')
                print(f"\033[93m{e}\033[0m")
                # Remember this datafile, so it will not be used again
                previously_combined_data_files.append(data_file_path.name)
                continue

            with h5py.File(data_file_path, 'r+') as f_hdf5:
                # Skip data files that don't have an experiment_ID
                try:
                    experiment_ID = f_hdf5.attrs["experiment_ID"]
                except KeyError:
                    experiment_ID = data_file_path.stem
                    f_hdf5.attrs["experiment_ID"] = experiment_ID
                    print("\033[93m! No experiment_ID found \033[0min", data_file_path,
                          f"\n\t{experiment_ID} added to hdf5 file")

                # Also skip data files based on experiment_ID information
                if not isinstance(self.excluded_experiment_IDs, type(None)) and experiment_ID in self.excluded_experiment_IDs:
                    print(f"\t{file_num + 1: 3d}/{len(data_file_paths)} {datetime.now().strftime('%H:%M:%S')} {data_file_path.stem}{experiment_ID: 5.0f} | ", end='')
                    print(f"\033[93mexcluded\033[0m | experiment_ID {experiment_ID}")
                    # Remove this datafile from the list of datafiles to be combined
                    data_file_paths.remove(data_file_path)
                    # Remember this datafile, so it will not be used again
                    previously_combined_data_files.append(data_file_path.name)
                    continue

                # This datafile is ready to be combined
                # For the future, still remember this datafile, so it will not be used again
                previously_combined_data_files.append(data_file_path.name)

                dset_prefixes = [key for key in f_hdf5.keys() if 'experiment' not in key]
                if not len(dset_prefixes):
                    print(f"\t{file_num + 1: 3d}/{len(data_file_paths)} {datetime.now().strftime('%H:%M:%S')} {data_file_path.stem}{experiment_ID: 5.0f} | ", end='')
                    print(f'\033[93No dset_prefixes found!\033[0m')

                dset_keys = dict()
                for dset_prefix in dset_prefixes:
                    dset_keys[dset_prefix] = list(f_hdf5[dset_prefix].keys())

            # Leave f_hdf5 wrapper and load data using pandas
            for dset_prefix in dset_prefixes:
                print(f"\t{file_num + 1: 3d}/{len(data_file_paths)} {datetime.now().strftime('%H:%M:%S')} {data_file_path.stem}{experiment_ID: 5.0f} | {dset_prefix} | ", end='')
                if self.combine_bout_data:
                    if "all_bout_data_pandas" in dset_keys[dset_prefix]:
                    # if "all_bout_data_pandas" in f_hdf5[dset_prefix].keys():
                        try:
                            df = pd.read_hdf(data_file_path, key=f'{dset_prefix}/all_bout_data_pandas')
                        except UnicodeDecodeError as e:
                            print(f'\033[93m{e}\033[0m')
                            continue

                        # Store end_result_info as column values
                        index_names = [idx for idx in df.index.names if 'end_result_info' in idx]
                        df.reset_index(index_names, inplace=True)

                        # Rename "fish_index" to "fish_or_agent_name" if present
                        if "fish_index" in df.index.names:
                            print("'fish_index' to 'fish_or_agent_name' | ",  end='')
                            df.index.rename({"fish_index": "fish_or_agent_name"}, inplace=True)

                        combined_df_bout_data_list.append(df)
                    else:
                        print(f'\033[93mall_bout_data_pandas not found for {data_file_path}\033[0m ')

                if self.combine_tracking_data:
                    if "all_freely_swimming_tracking_data_pandas" in dset_keys[dset_prefix]:
                    # if "all_freely_swimming_tracking_data_pandas" in f_hdf5[dset_prefix].keys():
                        df = pd.read_hdf(data_file_path, key=f'{dset_prefix}/all_freely_swimming_tracking_data_pandas')

                        # Rename "fish_index" to "fish_or_agent_name" if present
                        if "fish_index" in df.index.names:
                            print("'fish_index' to 'fish_or_agent_name' | ",  end='')
                            df.index.rename({"fish_index": "fish_or_agent_name"}, inplace=True)

                        combined_df_freely_swimming_tracking_data_list.append(df)

                    if "all_head_embedded_tracking_data_pandas" in dset_keys[dset_prefix]:
                    # if "all_head_embedded_tracking_data_pandas" in f_hdf5[dset_prefix].keys():
                        df = pd.read_hdf(data_file_path, key=f'{dset_prefix}/all_head_embedded_tracking_data_pandas')
                        combined_df_head_embedded_tracking_data = pd.concat([combined_df_head_embedded_tracking_data, df])

                print('\033[92mdone\033[0m')

        if len(combined_df_bout_data_list) > 0 and len(data_file_paths) > 0:
            print(f"\t{datetime.now().strftime('%H:%M:%S')} Storing all_bout_data_pandas...", end='')
            combined_df_bout_data = pd.concat(combined_df_bout_data_list)
            combined_df_bout_data.sort_index(inplace=True)
            combined_df_bout_data.to_hdf(path_to_combined_data_file,
                                         key="all_bout_data_pandas",
                                         complevel=self.compression)
            print('\033[92mdone\033[0m')

        if len(combined_df_freely_swimming_tracking_data_list) > 0 and len(data_file_paths) > 0:
            print(f"\t{datetime.now().strftime('%H:%M:%S')} Storing all_freely_swimming_tracking_data_pandas...", end='')
            combined_df_freely_swimming_tracking_data = pd.concat(combined_df_freely_swimming_tracking_data_list)
            combined_df_freely_swimming_tracking_data.sort_index(inplace=True)
            combined_df_freely_swimming_tracking_data.to_hdf(path_to_combined_data_file,
                                                             key="all_freely_swimming_tracking_data_pandas",
                                                             complevel=self.compression)
            print('\033[92mdone\033[0m')

        if len(combined_df_head_embedded_tracking_data_list) > 0 and len(data_file_paths) > 0:
            print(f"\t{datetime.now().strftime('%H:%M:%S')} Storing all_head_embedded_tracking_data_pandas...", end='')
            combined_df_head_embedded_tracking_data = pd.concat(combined_df_head_embedded_tracking_data_list)
            combined_df_head_embedded_tracking_data.sort_index(inplace=True)
            combined_df_head_embedded_tracking_data.to_hdf(path_to_combined_data_file,
                                                           key="all_head_embedded_tracking_data_pandas",
                                                           complevel=self.compression)
            print('\033[92mdone\033[0m')

        # Save a list of the folder names, so later one can skip those
        with h5py.File(path_to_combined_data_file, "a") as f_hdf5:
            f_hdf5.attrs["combined_data_file_names"] = previously_combined_data_files

        print(f"\tCombineData \033[92mdone\033[0m")

        return combined_df_bout_data
