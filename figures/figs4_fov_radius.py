
# # Local library imports
from utils.general_utils import load_event_df, get_n_fish, get_median_df
from settings.agent_settings import Larva, Juvie
from settings.general_settings import path_to_main_fig_folder, path_to_main_data_folder

from fig2_helpers import *

# #############################################################################
# User settings
# #############################################################################
# Import stimulus settings
from settings.stim_homogeneous_radius import *

# Plot settings
alpha = 0.2

# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath('figS4_fov')
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
bin_name = 'radius_bin'
label = 'Radius (cm)'
radii = [0.001, 0.1, 0.2, 0.3, 0.5, 1, 2, 5]
radius_bins = [0, 0.05, 0.15, 0.25, 0.4, 0.75, 1.5, 3, 6]
ticks = [0, 2, 5]

# #############################################################################
# Load and prepare data
# #############################################################################
# Load data
event_df = load_event_df(path_to_main_data_folder, path_name, agents)
n_fish_dict = get_n_fish(event_df, agents)

# Remove wall interactions
event_df = event_df.query('radius <= 5').copy()  # cm

# Specify bins: radii #########################################################
event_df[bin_name] = pd.cut(abs(event_df.index.get_level_values('stimulus_name')), bins=radius_bins, labels=radii)

# Compute median ##############################################################
median_df = get_median_df(event_df, bin_name)

# #############################################################################
# Plot midline distribution
# #############################################################################
fig, ax = plot_midline_length(event_df)
savefig(fig, path_to_fig_folder.joinpath('radius_midline_length_distribution.pdf'), close_fig=True)

# #############################################################################
# Swim properties: plot median with bin
# #############################################################################
stim_queries = ['inside_', 'outside_']

for j, stim_query in enumerate(stim_queries):
    if stim_query == 'inside_':
        stim_median_df = median_df.query(f'stimulus_name > 0')
    else:
        stim_median_df = median_df.query(f'stimulus_name < 0')

    fig = plot_median(
        stim_median_df, None,
        agents, prop_classes,
        bin_name, None,
        label, ticks, ticks,
    )
    savefig(fig, path_to_fig_folder.joinpath(f'radius_median_{stim_query}.pdf'), close_fig=True)
