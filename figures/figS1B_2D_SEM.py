import pandas as pd

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
path_name = 'figS1/S1B'
path_to_fig_folder = path_to_main_fig_folder.joinpath(path_name)
path_to_stats_txt = path_to_fig_folder.joinpath('stats.txt')

# Specify agents (comment out either Larva/Juvie or LarvaAgent/JuvieAgent)
agents = [Larva(), Juvie()]
agents_str = '_'.join([agent.name for agent in agents])

# #############################################################################
# Load and prepare data
# #############################################################################
full_tracking_df = load_tracking_df(path_to_main_data_folder, path_name, agents)

# Remove values too close to the wall
tracking_df = full_tracking_df.loc[full_tracking_df['radius'] <= 5].copy()

# Compute swim properties
tracking_df, median_ind_df, std_ind_df, n_frames = compute_bins(tracking_df)

# #############################################################################
# Plot 2D SEM hexbins
# #############################################################################

hb_dict = {}

# Loop over agents
for k, agent in enumerate(agents):
    hb_dict[agent.name] = {}
    agent_tracking_df = tracking_df.query(agent.query)
    stim_names = agent_tracking_df.index.unique('stimulus_name')

    # Loop over stimuli
    for i, stim_values in enumerate(stim_dict.values()):
        stim_name = stim_values['stim_name']
        print(f"\tComputing density histograms: {agent.name} {stim_name}")

        hb_dict[agent.name][stim_name] = {}
        stim_df = agent_tracking_df.xs(stim_name, level='stimulus_name')

        fish_arrays = []
        fig_empty, ax_empty = plt.subplots(1, 1)

        # Loop over experiments
        for j, (exp_ID, fish_df) in enumerate(stim_df.groupby('experiment_ID')):
            # Compute hexbin with density for each individual
            fish_hb = ax_empty.hexbin(
                fish_df['x_position'], fish_df['y_position'],
                gridsize=11, extent=(-6, 6, -6, 6),
                linewidths=0,
                mincnt=0,  # Include all hexagons, even those with zero counts
            )
            fish_array = fish_hb.get_array()
            fish_arrays.append(fish_array)

        # Create numpy array
        fish_arrays = np.array(fish_arrays)
        # Normalise within fish, to compute percentage
        fish_arrays_normed = fish_arrays / np.sum(fish_arrays, axis=1, keepdims=True) * 100

        # The outermost hexagons only include half of the data,
        # so we set their values to nan to avoid misinterpretation
        hex_x, hex_y = fish_hb.get_offsets().T  # Extract (x, y) centers of hexagons
        radii = np.sqrt(hex_x ** 2 + hex_y ** 2)
        fish_arrays_normed[:, radii >= 5] = np.nan

        # Compute mean, std, sem over fish
        n_fish = fish_arrays_normed.shape[0]
        h_mean = np.nanmean(fish_arrays_normed, axis=0)
        h_std = np.nanstd(fish_arrays_normed, axis=0)
        h_sem = h_std / np.sqrt(n_fish)

        # Store values in dictionary
        hb_dict[agent.name][stim_name]['fish_arrays'] = fish_arrays
        hb_dict[agent.name][stim_name]['h_mean'] = h_mean
        hb_dict[agent.name][stim_name]['h_sem'] = h_sem

        # # For check, plot SEM
        # # Update hexbin with values
        # fish_hb.set_array(h_sem)
        # fish_hb.set_clim(0, 0.2)
        # fish_hb.set_cmap('Greys_r')  # must be set again to update figure
        # ax_empty.set_title(f'{agent.name} {stim_name} SEM')

        # Close empty figure
        plt.close(fig_empty)

for color in ['agent', 'grey']:

    # Plot results
    fig = create_figure(fig_width=fig_width_cm, fig_height=4*small_grid_y)
    for k, agent in enumerate(agents):
        for i, stim_values in enumerate(stim_dict.values()):
            if i >= 4:
                break

            stim_name = stim_values['stim_name']
            for j, stat in enumerate(['h_sem', 'h_mean']):
                h_values = hb_dict[agent.name][stim_name][stat]

                # Add axes
                l, b, w, h = (
                    small_grid_x * (3*i + k + 1),
                    small_grid_y * j,
                    small_grid_x - pad,
                    small_grid_y - pad,
                )
                ax = add_axes(fig, l, b, w, h)

                # Set same values for mean and SEM plots
                vmin, vmax = 0, 2
                if stat == 'h_mean':
                    stat_label = 'Mean (%)'
                elif stat == 'h_sem':
                    stat_label = 'SEM (%)'

                if color == 'grey':
                    cmap = 'Greys_r'
                elif color == 'agent':
                    cmap = agent.cmap

                # Plot hexbin
                # # Create empty hexbin to get hexagon centers
                hb = ax.hexbin(
                    fish_df['x_position'], fish_df['y_position'],
                    gridsize=11, extent=(-6, 6, -6, 6),
                    linewidths=0,
                    mincnt=0,  # Include all hexagons, even those with zero counts
                )

                # Update hexbin with values
                hb.set_array(h_values)
                hb.set_clim(vmin, vmax)
                hb.set_cmap(cmap)  # must be set again to update figure

                # Format
                ax.set_aspect('equal')
                hide_all_spines_and_ticks(ax)
                set_ticks(ax, x_ticks=[], y_ticks=[])
                if i == 0 and k == 0:  # Format first row
                    ax.set_ylabel(stat_label)

                # Create colorbar
                ticks = np.linspace(vmin, vmax, 5)
                ticklabels = [f'{tick:.1f}' for tick in ticks]
                cbar = get_colorbar(
                    cmap, ticks=ticks, ticklabels=ticklabels,
                    orientation='vertical',
                    figsize=(ax_x_cm * cm, ax_y_cm * cm)
                )
                # Store colorbar separately
                savefig(cbar, path_to_fig_folder.joinpath(f'S1B_Mean_SEM_{color}_{agent.name}_cbar.pdf'), close_fig=True)

    # Save figure
    savefig(fig, path_to_fig_folder.joinpath(f'S1B_Mean_SEM_{color}.pdf'), close_fig=True)


