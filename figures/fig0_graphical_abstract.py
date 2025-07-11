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
path_to_fig_folder = path_to_main_fig_folder.joinpath(f'fig0_GA')

# Specify agents ##############################################################
agents = [Larva(), Juvie(), ]
# agents = [LarvaAgent(), JuvieAgent()]
# from cmcrameri import cm
# agents[0].cmap = cm.devon_r  # LarvaAgent colormap

# Emphasise effects for charicature graphical abstract
cf_vmin, cf_vmax = 0.002, 0.015     # Fish
# cf_vmin, cf_vmax = 0.01, 0.013    # Agents

# #############################################################################
# Load and prepare data
# #############################################################################
full_tracking_df = load_tracking_df(path_to_main_data_folder, path_name, agents)

# Remove values too close to the wall
tracking_df = full_tracking_df.loc[full_tracking_df['radius'] <= 5].copy()

# Keep only the circular gradient stimulus
stim_tracking_df = tracking_df.xs('azimuth_left_dark_right_bright', level='stimulus_name')

# Prepare donut-shape with inner radius 5 and outer radius 6
n, radii = 50, [5, 25]
theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
xs = np.outer(radii, np.cos(theta))
ys = np.outer(radii, np.sin(theta))
# in order to have a closed area, the circles should be traversed in opposite directions
xs[1, :] = xs[1, ::-1]
ys[1, :] = ys[1, ::-1]

# Plot 2D KSDEs using Matplotlib ##############################################
from scipy.stats import gaussian_kde

gridsize = 50j  # Number of points in the grid, must be a complex number
r_max = 5
xx, yy = np.mgrid[-r_max:r_max:gridsize, -r_max:r_max:gridsize]
grid_coords = np.vstack([xx.ravel(), yy.ravel()])

for agent in agents:
    plot_df = stim_tracking_df.query(agent.query)

    # Extract data
    x = plot_df["x_position"].values
    y = plot_df["y_position"].values

    # Compute KDE
    bw_adjust = 0.2
    kde = gaussian_kde([x, y], bw_method=bw_adjust)

    # Create a grid
    print("Creating grid for KDE...", end=' ')
    zz = kde(grid_coords).reshape(xx.shape)
    print("done.")

    # Plot
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 6))
    cs = ax.contourf(
        xx, yy, zz, levels=1000,
        cmap=agent.cmap, vmin=cf_vmin, vmax=cf_vmax,
    )

    # Block KDE-smoothed values outside arena
    ax.fill(np.ravel(xs), np.ravel(ys), edgecolor='white', facecolor='white')

    # Format
    ax.set_aspect('equal')
    set_lims(ax, x=(-5, 5), y=(-5, 5))
    hide_all_spines_and_ticks(ax)
    set_labels(ax, x='', y='')

    # Store
    savefig(fig, path_to_fig_folder.joinpath(f'cf_{agent.name}_{bw_adjust:1.1f}.png'), close_fig=False)


# Plot 2D KDEs using Seaborne #################################################
for agent in agents:
    plot_df = stim_tracking_df.query(agent.query).reset_index()

    for bw_adjust in [4, 3, 2]:

        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 6))
        kde = sns.kdeplot(
            data=plot_df,
            x="x_position", y="y_position",
            cmap=agent.cmap,
            fill=True,  # Fill the contour
            bw_adjust=bw_adjust, gridsize=50, levels=100,
            common_norm=False,
            ax=ax,
        )

        # Block KDE-smoothed values outside arena
        ax.fill(np.ravel(xs), np.ravel(ys), edgecolor='white', facecolor='white')
        # Format
        ax.set_aspect('equal')
        set_lims(ax, x=(-5, 5), y=(-5, 5))
        hide_all_spines_and_ticks(ax)
        set_labels(ax, x='', y='')
        # Store
        savefig(fig, path_to_fig_folder.joinpath(f'kde_{agent.name}_{bw_adjust:1.1f}.png'), close_fig=False)




