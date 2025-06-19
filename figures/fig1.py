

# Local library imports
from fig1_helpers import *
from settings.agent_settings import *
from settings.general_settings import *
from utils.general_utils import load_tracking_df

# #############################################################################
# User settings
# #############################################################################
# Paths
experiment_name = 'arena_locked'
path_name = experiment_name
path_to_fig_folder = path_to_main_fig_folder.joinpath(f'fig1_{experiment_name}')
path_to_stats_txt = path_to_fig_folder.joinpath('main/stats.txt')


# Specify agents (comment out either Larva/Juvie or LarvaAgent/JuvieAgent)
agents = [Larva(), Juvie()]
vmin, vmax = ref_vmin, ref_vmax
vmin_diff, vmax_diff = -0.8, 0.8
example_IDs = [100, 433]  # Larva, Juvie

agents = [LarvaAgent(), JuvieAgent()]
vmin, vmax = test_vmin, test_vmax
vmin_diff, vmax_diff = -0.4, 0.4
example_IDs = [[44, 44, 92, 92], [62, 93, 87, 55]]   # Larval agent, Juvenile agent

agents_str = '_'.join([agent.name for agent in agents])

# Plot settings
fig_width_cm = 18.4     # cm
fig_height_cm = 18    # cm
n_cols = 13
pad = 0.1  # cm, both in x and y direction
small_grid_x = fig_width_cm / n_cols
small_grid_y = small_grid_x  # square grid

# #############################################################################
# Load and prepare data
# #############################################################################
full_tracking_df = load_tracking_df(path_to_main_data_folder, path_name, agents)

# Remove values too close to the wall
tracking_df = full_tracking_df.loc[full_tracking_df['radius'] <= 5].copy()

# Compute swim properties
tracking_df, median_ind_df, std_ind_df, n_frames = compute_bins(tracking_df)
x_df, x_ind_df = compute_swim_properties_tracking(tracking_df, n_frames, ['x_bin'])
radius_df, radius_ind_df = compute_swim_properties_tracking(tracking_df, n_frames, ['radius_bin'])
azimuth_df, azimuth_ind_df = compute_swim_properties_tracking(tracking_df, n_frames, ['azimuth_bin'])

# #############################################################################
# Create separate figure for trajectories (store as png)
# #############################################################################
fig = create_figure(fig_width=fig_width_cm, fig_height=2*small_grid_y)

j = 1
# Loop over agents
for k, (agent, exp_ID) in enumerate(zip(agents, example_IDs)):

    # Ensure we have an exp_ID for each stimulus
    if isinstance(exp_ID, int):
        exp_IDs = [exp_ID] * 4
    else:
        exp_IDs = exp_ID

    # Loop over stimuli
    for i, (stim_values, exp_ID) in enumerate(zip(stim_dict.values(), exp_IDs)):
        stim_name = stim_values['stim_name']
        print(f"Plotting trajectories: {agent.name} {stim_name} {exp_ID}", end='\r')
        exp_df = full_tracking_df.xs(exp_ID, level='experiment_ID').query(agent.query)

        if i >= 4:
            break

        # Add axes
        l, b, w, h = (
            small_grid_x * (3*i + k + 1),
            small_grid_y * j,
            small_grid_x - pad,
            small_grid_y - pad,
        )
        ax = add_axes(fig, l, b, w, h)

        # Plot trajectory
        plot_single_trajectory(ax, exp_df, stim_name, agent)

# Add scalebar in separate ax
l, b, w, h = (
    small_grid_x * (3 * i + k + 2),
    small_grid_y * j,
    small_grid_x - pad,
    small_grid_y - pad,
)
ax = add_axes(fig, l, b, w, h)

set_lims(ax, [-6, 6], [-6, 6])
hide_all_spines_and_ticks(ax)
set_aspect(ax, 'equal')
add_scalebar(ax, size=1, label='1 cm', loc='lower center')

# Save figure #################################################################
print("Saving figure: ", end='')
savefig(fig, path_to_fig_folder.joinpath('main', f'main_{agents_str}_trajs.png'), close_fig=True)
print("\033[92mdone\033[0m")

# #############################################################################
# Create figure
# #############################################################################
fig = create_figure(fig_width=fig_width_cm, fig_height=fig_height_cm)

# Illustrate stimuli ##########################################################
j = 7
# Loop over stimuli
for i, stim_values in enumerate(stim_dict.values()):
    if i >= 4:
        break

    # Add axes
    l, b, w, h = (
        small_grid_x * (3*i + 2),
        small_grid_y * j,
        small_grid_x - pad,
        small_grid_y - pad,
    )
    ax = add_axes(fig, l, b, w, h)

    # Extract stimulus settings
    stim_name = stim_values['stim_name']
    plot_stimulus_ax(ax, stim_name)

# Plot 2D density hexbins #####################################################
# Loop over agents
for k, agent in enumerate(agents):
    agent_tracking_df = tracking_df.query(agent.query)
    stim_names = agent_tracking_df.index.unique('stimulus_name')

    # Compute control density histogram for 2d density difference
    print("Computing control density histogram: ", end='')
    control_array = compute_2d_density_control_ax(agent_tracking_df)
    print("\033[92mdone\033[0m")

    # Loop over stimuli
    for i, stim_values in enumerate(stim_dict.values()):
        if i >= 4:
            break
        stim_name = stim_values['stim_name']
        print(f"Plotting 2D density bins: {agent.name} {stim_name}", end='\r')

        # Get data for this stimulus
        stim_df = agent_tracking_df.xs(stim_name, level='stimulus_name').copy()
        n_fish_stim = stim_df.index.unique('experiment_ID').size

        # Plot 2D density hexbins #############################################
        j = 5
        # Add axes
        l, b, w, h = (
            small_grid_x * (3*i + k + 1),
            small_grid_y * j,
            small_grid_x - pad,
            small_grid_y - pad,
        )
        ax = add_axes(fig, l, b, w, h)

        # Plot 2D density
        cbar = plot_2d_density_ax(ax, stim_df, n_fish_stim, agent, vmin=vmin, vmax=vmax)

        # Store colorbar separately
        savefig(cbar, path_to_fig_folder.joinpath('main', f'E_colorbar_{agent.name}.pdf'), close_fig=True)

        # Plot 2D density difference hexbins ##################################
        j = 4
        # Add axes
        l, b, w, h = (
            small_grid_x * (3 * i + k + 1),
            small_grid_y * j,
            small_grid_x - pad,
            small_grid_y - pad,
        )
        ax = add_axes(fig, l, b, w, h)

        # Plot 2D density difference
        cbar = plot_2d_density_diff_ax(ax, stim_df, n_fish_stim, control_array, vmin_diff, vmax_diff)

        # Store colorbar separately
        savefig(cbar, path_to_fig_folder.joinpath('main', f'F_colorbar_{agent.name}.pdf'), close_fig=True)


# Add scalebar in separate ax
j = 5
i = 3
l, b, w, h = (
    small_grid_x * (3 * i + k + 2),
    small_grid_y * j,
    small_grid_x - pad,
    small_grid_y - pad,
)
ax = add_axes(fig, l, b, w, h)

set_lims(ax, [-6, 6], [-6, 6])
hide_all_spines_and_ticks(ax)
set_aspect(ax, 'equal')
add_scalebar(ax, size=1, label='1 cm', loc='lower center')

print(f"Plotting 2D density bins: \033[92mdone\033[0m")

# Plot 1D density histograms ##################################################
do_subtract = 'control'
ref_stim_name = None

# Loop over stimuli
for i, stim_values in enumerate(stim_dict.values()):
    if i >= 4:
        break
    stim_name = stim_values['stim_name']
    print(f"Plotting 1D density bins: {stim_name}", end='\r')

    j = 3
    # Add axes
    l, b, w, h = (
        small_grid_x * (3 * i + 1),
        small_grid_y * j,
        2*small_grid_x - pad,
        small_grid_y - pad,
    )
    ax = add_axes(fig, l, b, w, h)

    plot_1d_density_ax(
        ax, agents, stim_name, stim_values, do_subtract,
        x_df, radius_df, azimuth_df,
    )

print(f"Plotting 1D density bins: \033[92mdone\033[0m")

# Plot statistics based on mean within fish ###################################
print("Computing statistics:", end='')
bin_stats_list, stat_df, stat_str = get_bin_stats(median_ind_df, agents, stim_dict)
# Store statistics txt file
with open(path_to_stats_txt, 'a+') as output:
    output.write(stat_str)
print(f"\033[92mdone\033[0m")

# Loop over stimuli
for i, stim_values in enumerate(stim_dict.values()):
    if i >= 4:
        break
    stim_name = stim_values['stim_name']
    print(f"Plotting statistics: {stim_name}", end='\r')

    # Add axes
    j = 1
    l, b, w, h = (
        small_grid_x * (3 * i + 1),
        small_grid_y * j,
        2*small_grid_x - pad,
        small_grid_y - pad,
    )
    ax = add_axes(fig, l, b, w, h)

    plot_bin_stats_stripplot_ax(ax, median_ind_df, stat_df, agents, stim_values)

print(f"Plotting statistics: \033[92mdone\033[0m")

# Save figure #################################################################
print("Saving figure: ", end='')
savefig(fig, path_to_fig_folder.joinpath('main', f'main_{agents_str}.pdf'), close_fig=True)
print("\033[92mdone\033[0m")


#
# agent_df = full_tracking_df.query(agent.query)
# grouped = agent_df.groupby('experiment_ID')
#
# for stim_name in [
# 'splitview_left_dark_right_bright',
# 'azimuth_left_dark_right_bright',
# 'center_bright_outside_dark',
# 'center_dark_outside_bright',
# ]:
#
#     fig = create_figure(fig_width=fig_width_cm, fig_height=fig_height_cm)
#     for k, (exp_ID, exp_df) in enumerate(grouped):
#         print(f"Plotting trajectories: {agent.name} {exp_ID}")
#
#         i = k % n_cols
#         j = k // n_cols
#
#         if i * j > n_cols ** 2:
#             break
#
#         # Add axes
#         l, b, w, h = (
#             small_grid_x * i,
#             small_grid_y * j,
#             small_grid_x - pad,
#             small_grid_y - pad,
#         )
#         ax = add_axes(fig, l, b, w, h)
#
#         # Plot trajectory
#         plot_single_trajectory(ax, exp_df, stim_name, agent)
#         ax.text(0.5, 0.5, exp_ID)
#
#     fig.suptitle(f'{agent.name} {stim_name}')
