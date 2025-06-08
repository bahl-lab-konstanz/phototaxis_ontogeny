
# Local library imports
from utils.general_utils import load_event_df, get_n_fish, get_median_df
from settings.agent_settings import Larva, Juvie
from settings.general_settings import path_to_main_fig_folder, path_to_main_data_folder

from fig2_helpers import *

# #############################################################################
# User settings
# #############################################################################
# Import stimulus settings
from settings.stim_contrast_fov import *

# Plot settings
alpha = 0.2

# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath('figS4_FOV')
path_to_fig_folder.mkdir(exist_ok=True, parents=True)
key_base = experiment_name

# Agents
agents = [Larva(), Juvie()]
agents_str = '_and_'.join([agent.name for agent in agents])

# Properties
prop_classes = [
    PercentageTurns(),
    TurnAngle(),
    TotalDuration(),
    Distance(),
    # PercentageLeft(),
]

# Stimulus settings
bin_name = 'distance_bin'
label = 'Offset (cm)'
distances = [-5, -2, -0.5, 0, 0.5, 2, 5]
distance_bins = [-10, -2.5, -0.75, -0.25, 0.25, 0.75, 2.5, 10]
ticks = [-5, -2, 0, 2, 5]

# #############################################################################
# Load and prepare data
# #############################################################################
# Load data
event_df = load_event_df(path_to_main_data_folder, path_name, agents)
n_fish_dict = get_n_fish(event_df, agents, stim_names=[-5.0])

# Remove wall interactions
event_df = event_df.query('radius <= 5').copy()  # cm

# Specify bins: distance ######################################################
event_df[bin_name] = pd.cut(event_df.index.get_level_values('stimulus_name'), bins=distance_bins, labels=distances)

# #############################################################################
# Swim properties: plot median with bin
# #############################################################################
# Compute median ##########################################################
phase_event_df = event_df.loc[event_df['time'] > 10]
median_df = get_median_df(phase_event_df, bin_name)

fig = plot_median(
    median_df, None,
    agents, prop_classes,
    bin_name, None,
    label, ticks, ticks,
)
savefig(fig, path_to_fig_folder.joinpath('offset_median.pdf'), close_fig=True)
