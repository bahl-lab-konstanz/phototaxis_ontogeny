# Import default settings and functions for figure 2
from fig2_helpers import *

# Imports
import datetime

# # Local library imports
from utils.general_utils import load_tracking_df, load_event_df, get_n_fish, get_median_df
from settings.agent_settings import Larva, Juvie
from settings.general_settings import path_to_main_fig_folder, path_to_main_data_folder, turn_threshold

# #############################################################################
# User settings
# #############################################################################
stim_name = 'azimuth_left_dark_right_bright_virtual_yes'
path_name = 'arena_locked'

stim_dict = {   # Stimuli to plot
    stim_name: {
        'stim_name': stim_name,
        'bin_name': 'azimuth_bin', 'bin_label': 'azimuth (deg)', 'column_name': 'azimuth',
        'ticks': azimuth_ticks, 'ticklabels': azimuth_ticklabels, 'ref_line': 180,
        'df_mean': 'azimuth_df', 'df_ind': 'azimuth_ind_df',  # later retrieved via eval
    },
}

# Agents
agents = [Larva(), Juvie()]
agents_str = '_and_'.join([agent.name for agent in agents])

# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath(f'fig2')
path_to_fig_folder.mkdir(exist_ok=True)
hdf5_file = path_to_fig_folder.joinpath(f'fit_df_{stim_name}.hdf5')
key_base = stim_name

# Create new stat str and file
stat_str = (f'Statistics for figure 2 {stim_name} {agents[0].name} and {agents[1].name}\n'
            f'\t{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
path_to_stat_file = path_to_fig_folder.joinpath(f'stats_{stim_name}.txt')

# #############################################################################
# Load and prepare data
# #############################################################################
# Load data
if path_to_fig_folder.joinpath(f'analysed_data_{stim_name}.hdf5').exists():
    event_df = pd.read_hdf(path_to_fig_folder.joinpath(f'analysed_data_{stim_name}.hdf5'), key='all_bout_data_pandas_event')
else:
    # For faster loading in the future...
    # ... load full dataset, ...
    event_df = load_event_df(path_to_main_data_folder, path_name, agents)
    # ... keep only relevant stimuli, ...
    event_df = event_df.xs(stim_name, level='stimulus_name', drop_level=False)
    # ... and store to figure folder.
    event_df.to_hdf(path_to_fig_folder.joinpath(f'analysed_data_{stim_name}.hdf5'), key='all_bout_data_pandas_event')

# Get number of fish ##########################################################
n_fish_dict = get_n_fish(event_df, agents, stim_names=list(stim_dict.keys()))

# Remove wall interactions ####################################################
event_df = event_df.loc[event_df['radius'] <= 5].copy()  # cm

# Compute azimuth and map to brightness
c_min, c_mid = 10, 300  # lux
event_df['azimuth_rad'] = np.arctan2(event_df['y_position'], event_df['x_position'])  # -pi to pi rad
event_df[col_name] = (c_mid - c_min) * (np.pi - np.abs(event_df['azimuth_rad'])) / np.pi + c_min  # c_min to c_mid lux

event_df[col_name] = map_azimuth_brightness(event_df, c_min=10, c_max=300)

# Set brightness bins, use bin centers as labels
event_df[bin_name] = pd.cut(event_df[col_name], bins=brightness_bins, labels=brightness_bin_centers, include_lowest=True)

# Get abs, to later find median orientation_change
event_df['estimated_orientation_change_abs'] = event_df['estimated_orientation_change'].abs()
# Set all orientation changes straight swims (below 10 degrees) to NaN
event_df['turn_angle'] = event_df['estimated_orientation_change_abs'].where(event_df['estimated_orientation_change_abs'] > 10)

# #############################################################################
# Figure: percentage turns, percentage left, turn angle
# #############################################################################
fig = plot_orientation_change_dist(
    event_df,
    agents, turn_threshold
)
savefig(fig, path_to_fig_folder.joinpath(f'orientation_change_dist', f'{stim_name}.pdf'), close_fig=True)

# #############################################################################
# Retrieve median_df
# #############################################################################
median_df = get_median_df(event_df, bin_name)

# #########################################################################
# Fit model as function of brightness
# #########################################################################
ind_meta_fit_df, mean_ind_meta_fit_df, mean_meta_fit_df = fit_model(
    median_df, agents, prop_classes,
    col_name, bin_name, model_names,
)

ind_meta_fit_df.to_hdf(hdf5_file, key=key_base + '_meta_ind', mode='a')
mean_ind_meta_fit_df.to_hdf(hdf5_file, key=key_base + '_meta_mean_ind', mode='a')
mean_meta_fit_df.to_hdf(hdf5_file, key=key_base + '_meta_mean', mode='a')

# #########################################################################
# Figures
# #########################################################################
# Plot median as function of brightness ###################################
fig = plot_median(
    median_df, mean_ind_meta_fit_df,
    agents, prop_classes,
    bin_name, model_name=None,
    label=label, ticks=brightness_bin_ticks, tick_labels=brightness_bin_tick_labels,
)
fig.suptitle(f'{stim_name} | {col_name}')
savefig(fig, path_to_fig_folder.joinpath('median', f'{stim_name}.pdf'), close_fig=True)

# Include fit
for model_name in model_names:
    fig = plot_median(
        median_df, mean_ind_meta_fit_df,
        agents, prop_classes,
        bin_name, model_name,
        label=label, ticks=brightness_bin_ticks, tick_labels=brightness_bin_tick_labels,
    )
    fig.suptitle(f'{stim_name} | {col_name} | {model_name}')
    savefig(fig, path_to_fig_folder.joinpath('median', f'{stim_name}_{model_name}.pdf'), close_fig=True)

# Plot fitted parameter values ############################################
for model_name in model_names:
    figs, meta_par_names, _stat_str = plot_fitted_pars(
        ind_meta_fit_df, mean_ind_meta_fit_df,
        agents, prop_classes,
        bin_name, model_name,
        label, jitter=0.3,
    )
    stat_str += _stat_str
    for fig, meta_par_name in zip(figs, meta_par_names):
        fig.suptitle(f'{stim_name} | {col_name} | {model_name} | {meta_par_name}')
        savefig(fig, path_to_fig_folder.joinpath('params', f'{stim_name}_{model_name}_{meta_par_name}.pdf'), close_fig=True)

# Print statistics and store to file ##########################################
# print(stat_str)
with open(path_to_stat_file, "w") as text_file:
    text_file.write(stat_str)


# Plot midline distribution ###################################################
fig, ax = plot_midline_length(event_df)
savefig(fig, path_to_fig_folder.joinpath('midline', f'{stim_name}.pdf'), close_fig=True)

# Nr. swims per bin ###########################################################
# Define ticks including 0
ticks = [int(i) for i in np.linspace(0, 300, 3)]
fig = plot_nr_events(
    event_df, col_name, brightness_bins, label,
    ticks, ticks
)
savefig(fig, path_to_fig_folder.joinpath('nr_swims', f'{stim_name}.pdf'), close_fig=True)


# #############################################################################
# Figures based on tracking data
# #############################################################################
# Load data
full_tracking_df = load_tracking_df(path_to_main_data_folder, path_name, agents)

# 1D density plots ############################################################
# from fig1_helpers import stim_dict_fig2, compute_bins, compute_swim_properties_tracking, plot_1d_density
#
# # Remove values too close to the wall
# tracking_df = full_tracking_df.loc[full_tracking_df['radius'] <= 5].copy()
#
# # Compute swim properties
# tracking_df, ref_median_ind_df, ref_std_ind_df, n_frames = compute_bins(tracking_df)
# ref_x_df, ref_x_ind_df = compute_swim_properties_tracking(tracking_df, n_frames, ['x_bin'])
# ref_radius_df, ref_radius_ind_df = compute_swim_properties_tracking(tracking_df, n_frames, ['radius_bin'])
# ref_azimuth_df, ref_azimuth_ind_df = compute_swim_properties_tracking(tracking_df, n_frames, ['azimuth_bin'])
#
# # Plot
# figs = plot_1d_density(agents, stim_dict_fig2, ref_x_df, ref_radius_df, ref_azimuth_df, )
# savefig(figs[0], path_to_fig_folder.joinpath('1D_density', f'1D_{agents_str}.pdf'), close_fig=True)
# savefig(figs[1], path_to_fig_folder.joinpath('1D_density', f'1D_{agents_str}_control.pdf'), close_fig=True)
# savefig(figs[2], path_to_fig_folder.joinpath('1D_density', f'1D_{agents_str}_chance.pdf'), close_fig=True)

# Accumulated Orientation and trajectories ####################################
print("Plotting accumulated orientation and trajectories", end='')
# Ensure values are sorted by time
tracking_df = full_tracking_df.xs('azimuth_left_dark_right_bright_virtual_yes', level='stimulus_name').sort_values('time')

ax0_x_cm = 4    # cm
ax1_x_cm = 1.5  # cm
ax_y_cm = 1.5   # cm
time_window = 5     # seconds
space_window = 4.6  # cm
# Trajectory scatter settings
s = 3  # marker size
alpha = 0.1  # transparency

# Plot example individuals
fig = create_figure(ax0_x_cm + ax1_x_cm + pad_x_cm, 3 * (ax_y_cm + pad_y_cm))

# Plot separately for each agent
for k, (agent, exp_ID, time) in enumerate(zip(
        agents, [101, 794], [294.6, 295.48]
)):
    time_start, time_end = time, time + time_window
    exp_df = (
        tracking_df
        .xs(exp_ID, level='experiment_ID')
        .xs(0, level='trial')
        .query(agent.query)
        .query('@time_start <= time <= @time_end')
    )

    # Accumulated orientation
    ax = add_axes(fig, 0, (2 - k) * (ax_y_cm + pad_y_cm), ax0_x_cm, ax_y_cm)
    ax.plot(exp_df['time'], exp_df['accumulated_orientation'], color=agent.color, linewidth=0.7)
    hide_all_spines_and_ticks(ax)

    # Trajectory
    ax = add_axes(fig, ax0_x_cm, (2 - k) * (ax_y_cm + pad_y_cm), ax1_x_cm, ax_y_cm)
    x, y = -1 * exp_df['x_position'], -1 * exp_df['y_position']  # Rotate swim direction with 180 degrees
    ax.scatter(
        x, y,
        color=agent.color, edgecolors='none',  # edgecolor to 'none' to avoid transparency issues
        s=s, alpha=alpha  # alpha,
    )

    # Ensure equal scales among ages
    x_max = x.max()
    x_min = x_max - space_window
    print(x.max() - x.min())
    set_lims(ax, x=[x_min - 0.1, x_max + 0.1])
    set_aspect(ax, 'equal')
    hide_all_spines_and_ticks(ax)

# Add third row for scale bars
k += 1
ax = add_axes(fig, 0, (2 - k) * (ax_y_cm + pad_y_cm), ax0_x_cm, ax_y_cm)
ax.plot(exp_df['time'], exp_df['accumulated_orientation'], alpha=0)  # alpha=0 to hide line
add_scalebar_horizontal(ax, size=5, label='500 ms')
add_scalebar_vertical(ax, size=25, label='25 deg', loc='upper right')
hide_all_spines_and_ticks(ax)

ax = add_axes(fig, ax0_x_cm, (2 - k) * (ax_y_cm + pad_y_cm), ax1_x_cm, ax_y_cm)
ax.plot(exp_df['x_position'], exp_df['y_position'], alpha=0)  # alpha=0 to hide line
set_aspect(ax, 'equal')
add_scalebar_horizontal(ax, size=1, label='1 cm')
hide_all_spines_and_ticks(ax)

savefig(fig, path_to_fig_folder.joinpath('example', 'example_trajectory.pdf'), close_fig=False)
savefig(fig, path_to_fig_folder.joinpath('example', 'example_trajectory.png'), close_fig=True)

# #############################################################################
# Legend
# #############################################################################
fig = get_legend_fig(agents)
savefig(fig, path_to_fig_folder.joinpath('legend.pdf'), close_fig=True)
