
import os
from pathlib import Path

import pandas as pd
import random

from utils.general_utils import *
from settings.prop_settings import *

# #############################################################################
# User settings
# #############################################################################
# Plot for presentations or print
do_presentation = False

# Specify the main folder name
main_folder_name = 'capelle_et_al'

# Default settings ############################################################
# Set seeds random number generator
seed = 42
rng = np.random.default_rng(seed)
random.seed(seed)

# Analysis settings
turn_threshold = 10  # deg
resampling_window = pd.Timedelta(1, unit='s')   # to compute median_df
rolling_window = pd.Timedelta(1, unit='s')      # to compute median_df and rolled_df


# #############################################################################
# Paths
# #############################################################################
if os.name == 'posix':
    # We are likely on my mac
    n_processes = 4

    path_to_main_folder = Path("/Users/mc/NC/Max Capelle (AG Bahl)/Analysis/freely-swimming", main_folder_name)
    path_to_main_data_folder = path_to_main_folder.joinpath('data')

    path_to_server_folder = Path(f'/Volumes/ag-bahl_behavior_data/ZT6 Multi-fish behavior setup')
    path_to_larva_server_folder = Path(f'/Volumes/ag-bahl_behavior_data/ZT6 Multi-fish behavior setup/Freely swimming larvae/Max/raw_data')
    path_to_agents_server_folder = Path(f'/Volumes/ag-bahl_behavior_data/ZT6 Multi-fish behavior setup/Freely swimming agents/Max/raw_data')
    path_to_juvie_server_folder = Path(f'/Volumes/ag-bahl_behavior_data1/ZT6 Multi-fish behavior setup/Freely swimming juveniles/Max/raw_data')

    # Path to simulation folder
    path_to_sim_folder = path_to_main_folder.joinpath("simulations")

elif os.name == 'nt':
    # We are likely on a behavior setup
    n_processes = 16

    # Get setup number
    setup_nr = os.environ.get('setup')

    # Find path_to_server_folder for larva and agents
    if Path(r'Z:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Max').exists():
        path_to_server_folder = Path(r'Z:\ZT6 Multi-fish behavior setup')
    elif Path(r'Z:\Freely swimming larvae\Max\raw_data').exists():
        path_to_server_folder = Path(r'Z:')
    elif Path(r'V:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Max').exists():
        path_to_server_folder = Path(r'V:\ZT6 Multi-fish behavior setup')
    elif Path(r'V:\Freely swimming larvae\Max\raw_data').exists():
        path_to_server_folder = Path(r'V:')
    elif Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Max').exists():
        path_to_server_folder = Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Max')
    elif Path(r'Y:\Freely swimming larvae\Max').exists():
        path_to_server_folder = Path(r'Y:')
        n_processes = 12  # Ephys setup has less cores
    else:
        print("\033[93mCould not find path_to_server_folder\033[0m")

    # Find path_to_juvie_server_folder for juveniles
    if Path(r'V:\Freely swimming juveniles\Max\raw_data').exists():
        path_to_juvie_server_folder = Path(r'V:\Freely swimming juveniles\Max\raw_data')
    elif Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data').exists():
        path_to_juvie_server_folder = Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    elif Path(r'U:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data').exists():
        path_to_juvie_server_folder = Path(r'U:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    elif Path(r'Z:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max').exists():
        path_to_juvie_server_folder = Path(r'Z:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    else:
        print("\033[93mCould not find path_to_juvie_server_folder\033[0m")

    # In path_to_server_folder are path_to_main_folder and path_to_output_folder, path_to_larva_server_folder and path_to_agents_server_folder
    path_to_main_folder = path_to_server_folder.joinpath(r'Freely swimming larvae\Max\analysed_data', main_folder_name)
    path_to_main_data_folder = path_to_main_folder.joinpath('data')

    path_to_larva_server_folder = path_to_server_folder.joinpath(r'Freely swimming larvae\Max\raw_data')
    path_to_agents_server_folder = path_to_server_folder.joinpath(r'Freely swimming agents\Max\raw_data')

    # Path to simulation folder on desktop
    path_to_sim_folder = Path(r'C:\Users\ag-bahl\Desktop\Simulated data\Max')

else:
    raise ValueError("OS not recognized")

# Create path to main figure folder
if do_presentation:
    path_to_main_fig_folder = path_to_main_folder.joinpath('black')
    path_to_main_fig_folder.mkdir(exist_ok=True)
else:
    path_to_main_fig_folder = path_to_main_folder.joinpath('white')
    path_to_main_fig_folder.mkdir(exist_ok=True)

