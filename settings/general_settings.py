
import numpy as np
import pandas as pd
import random
from utils.general_utils import *

# #############################################################################
# User settings
# #############################################################################
# Data (intput)
path_to_main_data_folder = Path('all_data')  # TODO: add path to folder containing all data folders.
# Figures (output)
path_to_main_fig_folder = Path('fig_components')  # TODO: change to actual path to store figures.
# Simulations (output)
path_to_sim_folder = path_to_main_data_folder.joinpath("simulations")

# Simulation settings
n_processes = 4  # TODO: update to available cores for simulation.

# Default settings ############################################################
# Set seeds random number generator
seed = 42
rng = np.random.default_rng(seed)
random.seed(seed)

# Analysis settings
turn_threshold = 10  # deg
resampling_window = pd.Timedelta(1, unit='s')   # to compute median_df
rolling_window = pd.Timedelta(1, unit='s')      # to compute median_df and rolled_df

# Visualisation settings ######################################################
document_type = 'paper'  # 'paper' (white background) or 'presentation' (black background)

