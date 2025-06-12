# Imports
from scipy.stats import ks_2samp

from utils.plot_utils import *
from settings.agent_settings import Larva, Juvie
from settings.plot_settings import *
from settings.general_settings import *


# #############################################################################
# User settings
# #############################################################################
# Import stimulus settings
from settings.stim_arena_locked import *

# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath('figS2_wall_interactions')
path_to_fig_folder.mkdir(exist_ok=True)

# Agents
agents = [Larva(), Juvie()]

# Stimuli to plot, in correct order
stim_dict = {
    'control': {},
}

# Visualisation settings
start_radius = 0  # cm
dr = 1
r_bins = np.arange(start_radius, 6, dr)
delta_phi_bins = np.linspace(-180, 180, 31)
delta_phi_bin_centers = (delta_phi_bins[1:] + delta_phi_bins[:-1]) / 2

nrows = len(agents)
ncols = len(r_bins)

# #############################################################################
# Load and prepare data
# #############################################################################
# Load data
event_df = load_event_df(path_to_main_data_folder, path_name, agents)

# Keep only data for Virtual circular gradient stimulus
stim_name = 'azimuth_left_dark_right_bright_virtual_yes'
event_df = event_df.xs(stim_name, level='stimulus_name', drop_level=False)

# Keep only data for control stimulus
# event_df = event_df.xs('control', level='stimulus_name', drop_level=False)

n_fish_dict = get_n_fish(event_df, agents, stim_names=[stim_name])

# Ensure we have data all the way to the border of the arena
if event_df['radius'].max() < 6:
    raise ValueError(f"Data does not extend to the border of the arena (6 cm). "
                     f"Current maximum radius: {event_df['radius'].max():.2f} cm")


# #############################################################################
# Orientation change distribution as function of radius
# #############################################################################
prop_class = OrientationChange()
bins = np.arange(-100, 101, 5)
bin_centers = (bins[:-1] + bins[1:]) / 2
bins_straight = np.arange(-turn_threshold, turn_threshold + 0.01, 5)

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, sharex=True, sharey=True,
    figsize=(12 * cm, nrows * 3 * cm, ), constrained_layout=True,
)
for row_num, (axs, agent) in enumerate(zip(axes, agents)):
    # Get agent data
    agent_df = event_df.query(agent.query)
    for col_num, (ax, r) in enumerate(zip(axs, r_bins)):
        r_start, r_end = r, r + dr

        # Histogram: all swims
        sns.histplot(
            data=agent_df.query(f'{r_start} <= radius < {r_end}').reset_index(), x=prop_class.prop_name, bins=bins,
            color=agent.color, alpha=0.7, linestyle='none',
            stat='density',
            ax=ax,
        )

        # Make straight swims lighter
        ax.axvspan(-turn_threshold, turn_threshold, color=COLOR_AGENT_MARKER, alpha=0.3, label='Straight swims')

        # Format
        if row_num == 0:
            ax.set_title(rf'${r_start} \leq r < {r_end}$ cm')

        set_labels(ax, y='Prob. density')

    # Format
    hide_spines(axes)
    set_spine_position(axes, spines='left')
    set_labels(axes, x='')
    set_ticks(axes, x_ticks=np.arange(-100, 101, 100))
    set_lims_and_bounds(axes, x=[-100, 100])
fig.supxlabel(f'{prop_class.label} ({prop_class.unit})')
savefig(fig, path_to_fig_folder.joinpath('orientation_change_dist_with_radius.pdf'), close_fig=False)


