"""
Supplementary figure: Whole-field luminance changes.
"""

# Standard library imports
import datetime

# Local library imports
from fig2_helpers import *
from utils.general_utils import load_event_df, get_n_fish, get_median_df
from settings.agent_settings import Larva, Juvie
from settings.general_settings import path_to_main_fig_folder, path_to_main_data_folder, turn_threshold

# #############################################################################
# User settings
# #############################################################################
# Import stimulus settings
from settings.stim_homogeneous_temporal import *
stim_name = experiment_name
col_name = 'temporal'
bin_name = 'temporal_bin'
model_names = ['double_linear']
values = [-300, -120, -60, -30, 30, 60, 120, 300]
value_bins = [-450, -150, -90, -45, 0, 45, 90, 150, 450]
label = 'Brightness change\n(lux/s)'
ticks = [-300, -150, 0, 150, 300]
tick_labels = [-300, '', 0, '', 300]

# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath('figS3_temporal')
path_to_fig_folder.mkdir(exist_ok=True)
hdf5_file = path_to_fig_folder.joinpath(f'fit_df_{experiment_name}.hdf5')
key_base = experiment_name

# Agents
agents = [Larva(), Juvie()]
agents_str = '_and_'.join([agent.name for agent in agents])

# Create new stat str and file
stat_str = (f'Statistics for {stim_name} {agents[0].name} and {agents[1].name}\n'
            f'\t{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')
path_to_stat_file = path_to_fig_folder.joinpath(f'stats_{stim_name}.txt')

# #############################################################################
# Load and prepare data
# #############################################################################
# Load data
event_df = load_event_df(path_to_main_data_folder, path_name, agents)
n_fish_dict = get_n_fish(event_df, agents)

# Remove wall interactions
event_df = event_df.loc[event_df['radius'] <= 5].copy()  # cm

# Map index fish_age 26 to 27 for easier fitting
event_df.rename(index={26: 27}, level='fish_age', inplace=True)

# Specify bins: temporal ######################################################
event_df[col_name] = event_df.index.get_level_values('stimulus_name')
event_df[col_name] = event_df[col_name].str.extract(r'([-+]?\d+)').astype(int)
event_df[bin_name] = pd.cut(event_df[col_name], bins=value_bins, labels=values)

median_df = get_median_df(event_df, bin_name)

# #########################################################################
# Fit model as function of brightness
# #########################################################################
ind_meta_fit_df, mean_ind_meta_fit_df, mean_meta_fit_df = fit_model(
    median_df, agents, prop_classes,
    col_name, bin_name, model_names,
)
ind_meta_fit_df.to_hdf(hdf5_file, key=key_base + '_meta_ind', mode='a')
# mean_ind_meta_fit_df.to_hdf(hdf5_file, key=key_base + '_meta_mean_ind', mode='a')
mean_meta_fit_df.to_hdf(hdf5_file, key=key_base + '_meta_mean', mode='a')

# #########################################################################
# Figures
# #########################################################################
# Plot median as function of brightness ###################################
for model_name in model_names:
    fig = plot_median(
        median_df, mean_meta_fit_df,
        agents, prop_classes,
        bin_name, model_name,
        label=label, ticks=ticks, tick_labels=tick_labels,
    )
    fig.suptitle(f'{stim_name} | {col_name} | {model_name}')
    savefig(fig, path_to_fig_folder.joinpath('median', f'{stim_name}_{model_name}.pdf'), close_fig=True)

# Plot fitted parameter values ############################################
for model_name in model_names:
    figs, meta_par_names, _stat_str = plot_fitted_pars(
        ind_meta_fit_df, mean_meta_fit_df,
        agents, prop_classes,
        bin_name, model_name,
        'lux/s'
    )
    stat_str += _stat_str
    for fig, meta_par_name in zip(figs, meta_par_names):
        fig.suptitle(f'{stim_name} | {col_name} | {model_name} | {meta_par_name}')
        savefig(fig, path_to_fig_folder.joinpath('params', f'{stim_name}_{model_name}_{meta_par_name}.pdf'), close_fig=True)

# Print statistics and store to file ##########################################
# print(stat_str)
with open(path_to_stat_file, "w") as text_file:
    text_file.write(stat_str)

# #############################################################################
# Supplementary figures
# #############################################################################
# Figure: percentage turns, percentage left, turn angle #######################
fig = plot_orientation_change_dist(
    event_df,  # Select time window,
    agents, turn_threshold
)
savefig(fig, path_to_fig_folder.joinpath(f'orientation_change_dist', f'{stim_name}.pdf'), close_fig=True)

# Plot midline distribution ###################################################
fig, ax = plot_midline_length(event_df)
savefig(fig, path_to_fig_folder.joinpath('midline', f'{stim_name}.pdf'), close_fig=True)

# Nr. swims per bin ###########################################################
fig = plot_nr_events(
    event_df, col_name, values, label,
    ticks, tick_labels
)
savefig(fig, path_to_fig_folder.joinpath('nr_swims', f'{stim_name}.pdf'), close_fig=True)
