
import random

from utils.general_utils import *
from settings.prop_settings import *

# #############################################################################
# User settings
# #############################################################################
# Plot for presentations or print
document_type = 'paper'  # 'thesis', 'paper', or 'presentation'

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
    path_to_larva_server_folder = Path(f'/Volumes/ag-bahl_behavior_data/ZT6 Multi-fish behavior setup/Freely swimming larvae/Max/raw_data')
    path_to_agents_server_folder = Path(f'/Volumes/ag-bahl_behavior_data/ZT6 Multi-fish behavior setup/Freely swimming agents/Max/raw_data')
    path_to_juvie_server_folder = Path(f'/Volumes/ag-bahl_behavior_data1/ZT6 Multi-fish behavior setup/Freely swimming juveniles/Max/raw_data')

    # Path to simulation folder
    path_to_sim_folder = path_to_main_folder.joinpath("simulations")

elif os.name == 'nt':
    # We are likely on a behavior setup
    n_processes = 16

    # Get setup number
    try:
        setup_index = os.environ['SETUP_INDEX']
    except KeyError:
        raise ValueError("Could not get setup number from environment variable 'SETUP_INDEX'")

    # Find paths to server based on setup_index
    if setup_index == '0':  # Setup 0
        path_to_larva_server_folder = Path(r'X:\ZT6 Multi-fish behavior setup')
        path_to_juvie_server_folder = Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    elif setup_index == '1':  # Setup 1
        path_to_server_folder = Path(r'V:\ZT6 Multi-fish behavior setup')
        path_to_juvie_server_folder = Path(r'U:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    elif setup_index == '2':  # Setup 2
        path_to_server_folder = Path(r'Z:\ZT6 Multi-fish behavior setup')
        path_to_juvie_server_folder = Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    elif setup_index == '3':  # Setup 3
        path_to_server_folder = Path(r'X:\ZT6 Multi-fish behavior setup')
        path_to_juvie_server_folder = Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    elif setup_index == '4':  # Setup 4
        path_to_server_folder = Path(r'X:\ZT6 Multi-fish behavior setup')
        path_to_juvie_server_folder = Path(r'Z:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    elif setup_index == '5':  # Setup 5
        path_to_server_folder = Path(r'Z:\ZT6 Multi-fish behavior setup')
        path_to_juvie_server_folder = Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    else:
        raise ValueError("\033[93mSETUP_INDEX not recognized\033[0m")

    # if Path(r'Z:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Max').exists():
    #     path_to_server_folder = Path(r'Z:\ZT6 Multi-fish behavior setup')
    # elif Path(r'Z:\Freely swimming larvae\Max\raw_data').exists():
    #     path_to_server_folder = Path(r'Z:')
    # elif Path(r'V:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Max').exists():
    #     path_to_server_folder = Path(r'V:\ZT6 Multi-fish behavior setup')
    # elif Path(r'V:\Freely swimming larvae\Max\raw_data').exists():
    #     path_to_server_folder = Path(r'V:')
    # elif Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Max').exists():
    #     path_to_server_folder = Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming larvae\Max')
    # elif Path(r'Y:\Freely swimming larvae\Max').exists():
    #     path_to_server_folder = Path(r'Y:')
    #     n_processes = 12  # Ephys setup has less cores
    # else:
    #     print("\033[93mCould not find path_to_server_folder\033[0m")
    #
    # # Find path_to_juvie_server_folder for juveniles
    # if Path(r'V:\Freely swimming juveniles\Max\raw_data').exists():
    #     path_to_juvie_server_folder = Path(r'V:\Freely swimming juveniles\Max\raw_data')
    # elif Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data').exists():
    #     path_to_juvie_server_folder = Path(r'Y:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    # elif Path(r'U:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data').exists():
    #     path_to_juvie_server_folder = Path(r'U:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    # elif Path(r'Z:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max').exists():
    #     path_to_juvie_server_folder = Path(r'Z:\ZT6 Multi-fish behavior setup\Freely swimming juveniles\Max\raw_data')
    # else:
    #     print("\033[93mCould not find path_to_juvie_server_folder\033[0m")

    # Check if path_to_server_folder exists
    if not path_to_server_folder.exists():
        raise ValueError(f'path_to_server_folder does not exist: {path_to_server_folder}')

    # Define paths relative to path_to_server_folder
    path_to_main_folder = path_to_server_folder.joinpath(r'Freely swimming larvae\Max\analysed_data', main_folder_name)
    path_to_main_data_folder = path_to_main_folder.joinpath('data')
    path_to_larva_server_folder = path_to_server_folder.joinpath(r'Freely swimming larvae\Max\raw_data')
    path_to_agents_server_folder = path_to_server_folder.joinpath(r'Freely swimming agents\Max\raw_data')

    # Path to simulation folder is on desktop
    path_to_sim_folder = Path(r'C:\Users\ag-bahl\Desktop\Simulated data\Max')

else:
    raise ValueError("OS not recognized")

# Define path to main figure folder
path_to_main_fig_folder = path_to_main_folder.joinpath(document_type)

# Print results
if __name__ == '__main__':
    print(f"Loaded general_settings.py\n"
          f"\tdocument_type:      \t{document_type}\n"
          f"\tmain_folder_name:   \t{main_folder_name}\n"
          f"\tpath_to_main_folder:\t{path_to_main_folder}\n")
