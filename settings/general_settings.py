
import numpy as np
import pandas as pd
import random
from utils.general_utils import *

# #############################################################################
# User settings
# #############################################################################
# Paths #######################################################################
# Data (intput)
path_to_main_data_folder = Path('/Users/mc/NC/Capelle et al. 2025/ready_to_submit_26_march/all_data')
# Figures (output)
path_to_main_fig_folder = Path('/Users/mc/NC/Capelle et al. 2025/ready_to_submit_26_march/fig_components')
# Simulations (output)
path_to_sim_folder = path_to_main_data_folder.joinpath("simulations")

# Default settings ############################################################
# Set seeds random number generator
seed = 42
rng = np.random.default_rng(seed)
random.seed(seed)

# Analysis settings
turn_threshold = 10  # deg
resampling_window = pd.Timedelta(1, unit='s')   # to compute median_df
rolling_window = pd.Timedelta(1, unit='s')      # to compute median_df and rolled_df

# Simulation settings
n_processes = 4

# Visualisation settings ######################################################
document_type = 'paper'  # 'paper' or 'presentation'

