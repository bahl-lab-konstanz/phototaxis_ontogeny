
# Local library imports
from fig1_helpers import *
from settings.agent_settings import *
from settings.general_settings import *
from utils.general_utils import load_tracking_df
from settings.stim_averaging_fov import path_name


# #############################################################################
# User settings
# #############################################################################

# Agents
ref_agents = [Larva()]
ref_agents_str = '_'.join([agent.name for agent in ref_agents])

# Plot settings
stim_name = 'azimuth_left_dark_right_bright_avg'
control_stim_name = 'azimuth_left_dark_right_bright'
stim_color = 'tab:blue'
control_stim_color = 'k'

# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath(path_name)
path_to_fig_folder.mkdir(exist_ok=True)
path_to_stats_txt = path_to_fig_folder.joinpath('stats.txt')

stim_dict_figS4B = {
    'azimuth_left_dark_right_bright_avg': {
        'stim_name': 'azimuth_left_dark_right_bright_avg', 'stim_label': 'Averaged circular gradient',
        'column_name': 'azimuth',
        'bin_name': 'azimuth_bin', 'bin_label': r'Angle (deg)', 'bin_label_avg': f"Average cosine\nweighted angle ()",
        'bins': azimuth_bins, 'bin_centers': azimuth_bin_centers,
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
}

stim_dict_all = {
    'control': {
        'stim_name': 'control', 'stim_label': 'Control',
        'column_name': 'azimuth',
        'bin_name': 'azimuth_bin', 'bin_label': r'Angle (deg)', 'bin_label_avg': f"Average cosine\nweighted angle ()",
        'bins': azimuth_bins, 'bin_centers': azimuth_bin_centers,
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
    'azimuth_left_dark_right_bright': {
        'stim_name': 'azimuth_left_dark_right_bright', 'stim_label': 'Circular gradient',
        'column_name': 'azimuth',
        'bin_name': 'azimuth_bin', 'bin_label': r'Angle (deg)', 'bin_label_avg': f"Average cosine\nweighted angle ()",
        'bins': azimuth_bins, 'bin_centers': azimuth_bin_centers,
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
    'azimuth_left_dark_right_bright_avg': {
        'stim_name': 'azimuth_left_dark_right_bright_avg', 'stim_label': 'Averaged circular gradient',
        'column_name': 'azimuth',
        'bin_name': 'azimuth_bin', 'bin_label': r'Angle (deg)', 'bin_label_avg': f"Average cosine\nweighted angle ()",
        'bins': azimuth_bins, 'bin_centers': azimuth_bin_centers,
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
    'azimuth_left_dark_right_bright_virtual_yes': {
        'stim_name': 'azimuth_left_dark_right_bright_virtual_yes', 'stim_label': 'Virtual circular gradient',
        'column_name': 'azimuth',
        'bin_name': 'azimuth_bin', 'bin_label': r'Angle (deg)', 'bin_label_avg': f"Average cosine\nweighted angle ()",
        'bins': azimuth_bins, 'bin_centers': azimuth_bin_centers,
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
}

# #############################################################################
# Load and prepare data
# #############################################################################
full_ref_tracking_df = load_tracking_df(path_to_main_data_folder, path_name, ref_agents)

# Get number of fish
# get_n_fish(full_ref_tracking_df, ref_agents)

# Remove values too close to the wall
ref_tracking_df = full_ref_tracking_df.loc[full_ref_tracking_df['radius'] <= 5].copy()

# Compute swim properties
ref_tracking_df, ref_median_ind_df, ref_std_ind_df, n_frames = compute_bins(ref_tracking_df)
ref_x_df, ref_x_ind_df = compute_swim_properties_tracking(ref_tracking_df, n_frames, ['x_bin'])
ref_radius_df, ref_radius_ind_df = compute_swim_properties_tracking(ref_tracking_df, n_frames, ['radius_bin'])
ref_azimuth_df, ref_azimuth_ind_df = compute_swim_properties_tracking(ref_tracking_df, n_frames, ['azimuth_bin'])

# #############################################################################
# Plot 1D density plots
# #############################################################################
figs4 = plot_1d_density(ref_agents, stim_dict_figS4B, ref_x_df, ref_radius_df, ref_azimuth_df, ref_stim_name='azimuth_left_dark_right_bright', ref_stim_color=control_stim_color)
savefig(figs4[0], path_to_fig_folder.joinpath(f'1D_{ref_agents_str}.pdf'), close_fig=True)
savefig(figs4[1], path_to_fig_folder.joinpath(f'1D_{ref_agents_str}_control.pdf'), close_fig=True)
savefig(figs4[2], path_to_fig_folder.joinpath(f'1D_{ref_agents_str}_chance.pdf'), close_fig=True)

# Plot all stimuli
figs_all = plot_1d_density(ref_agents, stim_dict_all, ref_x_df, ref_radius_df, ref_azimuth_df)
savefig(figs_all[0], path_to_fig_folder.joinpath(f'1D_all_{ref_agents_str}.pdf'), close_fig=True)
savefig(figs_all[1], path_to_fig_folder.joinpath(f'1D_all_{ref_agents_str}_control.pdf'), close_fig=True)
savefig(figs_all[2], path_to_fig_folder.joinpath(f'1D_all_{ref_agents_str}_chance.pdf'), close_fig=True)

# #############################################################################
# Plot statistics based on mean within fish
# #############################################################################
from fig1_helpers import _prepare_bin_stats_plot

ref_agents = [Larva(), Larva()]  # Duplicate to allow stat functions to work

print("Computing statistics:", end='')
bin_stats_list, stat_df, stat_str = get_bin_stats(
    ref_median_ind_df, ref_agents, stim_dict_all,
    control_stim_name=control_stim_name
)
# Store statistics txt file
with open(path_to_stats_txt, 'a+') as output:
    output.write(stat_str)
print(f"\033[92mdone\033[0m")

# Loop over stimuli
fig = create_figure(fig_width=3*small_grid_x, fig_height=3*small_grid_y)

# Add axes
i, j = 0, 1
l, b, w, h = (
    small_grid_x * (3 * i + 1),
    small_grid_y * j,
    small_grid_x - pad,
    small_grid_y - pad,
)
ax = add_axes(fig, l, b, w, h)


stim_values = stim_dict_all[stim_name]
column_name = stim_values['column_name']
bin_label = stim_values['bin_label_avg']
_, _, ticks, ticklabels, ref_line = _prepare_bin_stats_plot(
    ref_median_ind_df, stim_values, ref_agents,
)

# Set marker type based on first agent
marker = ref_agents[0].marker

plot_df = ref_median_ind_df.reset_index()  # Reset index for easier filtering
plot_df = plot_df[plot_df['stimulus_name'].isin([stim_name, control_stim_name])]  # Keep only the relevant stimuli

palette_dict = {
    stim_name: stim_color,
    control_stim_name: control_stim_color,
}

strip = sns.stripplot(
    data=plot_df,
    x='stimulus_name', y=column_name,  # Use 'group' to show all four categories
    hue='stimulus_name', palette=palette_dict,
    alpha=ALPHA, size=MARKER_SIZE,
    marker=marker,
    dodge=False, legend=False,
    ax=ax
)

# Add statistics
# # Compare stim and control
p_value_agent0 = p_value_to_stars(stat_df.query(
    f"stim0 == '{stim_name}' and stim1 == '{control_stim_name}' and agent0 == '{ref_agents[0].name}' and agent1 == '{ref_agents[0].name}'"
)['M_p_value'].values[0])

# x-coordinates in dataspace, y-coordinates in axes space
add_stats(ax, 0, 1, ANNOT_Y, p_value_agent0)

# Format
set_axlines(ax, axhlines=ref_line)
hide_spines(ax, ['top', 'right', 'bottom'])
# Format property axis
ax.set_ylabel(bin_label)
set_ticks(ax, y_ticks=ticks, y_ticklabels=ticklabels)
set_bounds(ax, y=(ticks[0], ticks[-1]))
set_lims(ax, y=(ticks[0], ticks[-1]))
# Format age axis
set_labels(ax, x='')
set_ticks(ax, x_ticks=[])
set_lims(ax, x=[-0.5, 1.5])  # Ensure all dots are visible

savefig(fig, path_to_fig_folder.joinpath(f'statistics_{ref_agents_str}.pdf'), close_fig=True)

