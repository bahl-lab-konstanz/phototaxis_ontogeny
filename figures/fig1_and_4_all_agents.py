"""
Create figures for fig 2B, fig S1, and fig S5.
Compile data for fig 4B, fig S7.
"""


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
path_name = 'fig1_and_4'

path_to_figS1_folder = path_to_main_fig_folder.joinpath('figS1')
path_to_figS5_folder = path_to_main_fig_folder.joinpath('figS5')
path_to_fig2B_folder = path_to_main_fig_folder.joinpath('fig2', '2B')
path_to_stats_file = path_to_main_data_folder.joinpath('fig1', f'stats.hdf5')
path_to_stats_txt = path_to_figS1_folder.joinpath('stats.txt')

stats_columns = [
    'ref_agent', 'test_agent', 'stimulus', 'bin',
    'do_subtract_control', 'do_bootstrap', 'i_bootstrap',
    'KL', 'SSD', 'Z-score', 'MSE',
]

# Specify agents ##############################################################
ref_agents = [Larva(), Juvie()]
ref_agents_str = '_'.join([agent.name for agent in ref_agents])
ref_example_IDs = [100, 433]  # Larva, Juvie

# Plot stimuli ################################################################
fig = plot_stimuli(stim_dict)
savefig(fig, path_to_figS1_folder.joinpath('stimuli.pdf'), close_fig=True)

# #############################################################################
# Load and prepare reference data
# #############################################################################
full_ref_tracking_df = load_tracking_df(path_to_main_data_folder, path_name, ref_agents)

if full_ref_tracking_df.empty:
    raise UserWarning(f"Skipping {ref_agents_str}: \033[91mno data\033[0m")

# Remove values too close to the wall
ref_tracking_df = full_ref_tracking_df.loc[full_ref_tracking_df['radius'] <= 5].copy()

# Print number of reference agents
# n_ref_fish = get_n_fish(ref_tracking_df, ref_agents)
stat_str = ''

# Compute swim properties
ref_tracking_df, ref_median_ind_df, ref_std_ind_df, n_frames = compute_bins(ref_tracking_df)
ref_x_df, ref_x_ind_df = compute_swim_properties_tracking(ref_tracking_df, n_frames, ['x_bin'])
ref_radius_df, ref_radius_ind_df = compute_swim_properties_tracking(ref_tracking_df, n_frames, ['radius_bin'])
ref_azimuth_df, ref_azimuth_ind_df = compute_swim_properties_tracking(ref_tracking_df, n_frames, ['azimuth_bin'])

# Store individual prob results as numpy arrays for later use in stats
p_array = get_prob_array(ref_agents, stim_dict, ref_x_ind_df, ref_radius_ind_df, ref_azimuth_ind_df)
store_prob_array(p_array, path_to_stats_file, ref_agents_str)

# #############################################################################
# Statistics
# #############################################################################
# Quantify midline length and brightness preference
fig = plot_midline_stats(ref_median_ind_df, ref_agents, stim_dict, )
savefig(fig, path_to_figS1_folder.joinpath(f'midlines_{ref_agents_str}.pdf'), close_fig=True)

# Compute statistics based on mean within fish ################################
bin_stats_list, stat_df, stat_str = get_bin_stats(ref_median_ind_df, ref_agents, stim_dict)
# Store statistics txt file
with open(path_to_stats_txt, 'a+') as output:
    output.write(stat_str)

# Plot and store
fig = plot_bin_stats(ref_median_ind_df, stat_df, ref_agents, stim_dict)
savefig(fig, path_to_figS1_folder.joinpath('stats', f'stats_{ref_agents_str}.pdf'), close_fig=True)

# Z-score and MSE #############################################################
# Compute z-scores, MSE between reference agents
stat_list = get_agent_stats(
        ref_agents[0], ref_agents[1], stim_dict,
        ref_x_ind_df, ref_radius_ind_df, ref_azimuth_ind_df,
)

# Store stats_df with key for these agents
path_to_stats_file.parent.mkdir(parents=True, exist_ok=True)
stats_df = pd.DataFrame(stat_list, columns=stats_columns)
stats_df.to_hdf(path_to_stats_file, key=ref_agents_str, mode='a')

# #############################################################################
# Plot trajectories, 1D and 2D density
# #############################################################################
# Plot trajectories in 30s windows ############################################
# fig = create_figure(fig_width=4*small_grid_y, fig_height=2*small_grid_y)
# # Loop over agents
# for k, (agent, exp_ID) in enumerate(zip(ref_agents, ref_example_IDs)):
#     # Ensure we have an exp_ID for each stimulus
#     if isinstance(exp_ID, int):
#         exp_IDs = [exp_ID] * 4
#     else:
#         exp_IDs = exp_ID
#
#     # Loop over stimuli
#     for i, (stim_values, exp_ID) in enumerate(zip(stim_dict.values(), exp_IDs)):
#         stim_name = stim_values['stim_name']
#         print(f"Plotting trajectories: {agent.name} {stim_name} {exp_ID}", end='\r')
#         exp_df = full_ref_tracking_df.xs(exp_ID, level='experiment_ID').query(agent.query)
#
#         if i >= 1:
#             break
#
#         # Loop over time windows
#         for j, window in enumerate([3, 7]):
#             # Add axes
#             l, b, w, h = (
#                 small_grid_x * (3*i + k + 1),
#                 small_grid_y * j,
#                 small_grid_x - pad,
#                 small_grid_y - pad,
#             )
#             ax = add_axes(fig, l, b, w, h)
#
#             # Extract trajectory for this time window
#             traj_df = exp_df.query(f'{window*30} <= time < {(window+1)*30}')
#
#             # Plot trajectory
#             plot_single_trajectory(
#                 ax, traj_df,
#                 stim_name,
#                 agent=agent, alpha=0.1, do_plot_stim_line=False)
#
# # Save figure #################################################################
# print("Saving figure: ", end='')
# savefig(fig, path_to_figS1_folder.joinpath(f'main_{ref_agents_str}_trajs_windows.pdf'), close_fig=True)
# print("\033[92mdone\033[0m")

# # Plot trajectories ###########################################################
# plot_all_trajectories(full_ref_tracking_df, ref_agents, stim_dict, path_to_figS1_folder)
# for k, (agent, exp_ID) in enumerate(zip([Larva(), Juvie()], [100, 433])):
#     fig = plot_exp_trajectories(full_ref_tracking_df.xs(exp_ID, level='experiment_ID').query(agent.query), agent, stim_dict, k)
#     savefig(fig, path_to_figS1_folder.joinpath('trajectories', 'examples', f'trajectory_{agent.name}.png'), close_fig=False)
#     savefig(fig, path_to_figS1_folder.joinpath('trajectories', 'examples', f'trajectory_{agent.name}.pdf'), close_fig=True)
#
# Plot 2D density hexbins #####################################################
fig, cbars = plot_2d_density(ref_tracking_df, ref_agents, stim_dict, vmin=ref_vmin, vmax=ref_vmax)
savefig(fig, path_to_figS1_folder.joinpath('2D_density', f'2D_{ref_agents_str}.pdf'), close_fig=True)
savefig(cbars[0], path_to_figS1_folder.joinpath('cbar', f'2D_{ref_agents[0].name}.pdf'), close_fig=True)
savefig(cbars[1], path_to_figS1_folder.joinpath('cbar', f'2D_{ref_agents[1].name}.pdf'), close_fig=True)
# Plot 2D density difference hexbins
fig, cbar = plot_2d_density_diff(ref_tracking_df, ref_agents, stim_dict, vmin=-1, vmax=1)
savefig(fig, path_to_figS1_folder.joinpath('2D_density_dff', f'2D_diff_{ref_agents_str}.pdf'), close_fig=True)
savefig(cbar, path_to_figS1_folder.joinpath('cbar', f'2D_diff_{ref_agents_str}.pdf'), close_fig=True)

# Plot agents separately and together
figs1 = plot_1d_density(ref_agents[0], stim_dict, ref_x_df, ref_radius_df, ref_azimuth_df)
savefig(figs1[0], path_to_figS1_folder.joinpath('1D_density', f'1D_{ref_agents[0].name}.pdf'), close_fig=True)
savefig(figs1[1], path_to_figS1_folder.joinpath('1D_density_control', f'1D_{ref_agents[0].name}.pdf'), close_fig=True)
savefig(figs1[2], path_to_figS1_folder.joinpath('1D_density_chance', f'1D_{ref_agents[0].name}.pdf'), close_fig=True)

figs2 = plot_1d_density(ref_agents[1], stim_dict, ref_x_df, ref_radius_df, ref_azimuth_df, )
savefig(figs2[0], path_to_figS1_folder.joinpath('1D_density', f'1D_{ref_agents[1].name}.pdf'), close_fig=True)
savefig(figs2[1], path_to_figS1_folder.joinpath('1D_density_control', f'1D_{ref_agents[1].name}.pdf'), close_fig=True)
savefig(figs2[2], path_to_figS1_folder.joinpath('1D_density_chance', f'1D_{ref_agents[1].name}.pdf'), close_fig=True)

figs3 = plot_1d_density(ref_agents, stim_dict, ref_x_df, ref_radius_df, ref_azimuth_df, )
savefig(figs3[0], path_to_figS1_folder.joinpath('1D_density', f'1D_{ref_agents_str}.pdf'), close_fig=True)
savefig(figs3[1], path_to_figS1_folder.joinpath('1D_density_control', f'1D_{ref_agents_str}.pdf'), close_fig=True)
savefig(figs3[2], path_to_figS1_folder.joinpath('1D_density_chance', f'1D_{ref_agents_str}.pdf'), close_fig=True)

# Plot agents with fixed x-axis for supplementary figure 5
figs4 = plot_1d_density(ref_agents, stim_dict, ref_x_df, ref_radius_df, ref_azimuth_df, ax_x_cm=3, )
savefig(figs4[0], path_to_figS5_folder.joinpath('1D_density', f'1D_{ref_agents_str}.pdf'), close_fig=True)
savefig(figs4[1], path_to_figS5_folder.joinpath('1D_density_control', f'1D_{ref_agents_str}.pdf'), close_fig=True)
savefig(figs4[2], path_to_figS5_folder.joinpath('1D_density_chance', f'1D_{ref_agents_str}.pdf'), close_fig=True)

# Plot for all bins
figs = plot_1d_density_all_bins(ref_agents[0], stim_dict,  ref_x_df, ref_radius_df, ref_azimuth_df, )
savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_all', f'1D_{ref_agents[0].name}.pdf'), close_fig=True)
savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_all_control', f'1D_{ref_agents[0].name}.pdf'), close_fig=True)
savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_all_chance', f'1D_{ref_agents[0].name}.pdf'), close_fig=True)

figs = plot_1d_density_all_bins(ref_agents[1], stim_dict,  ref_x_df, ref_radius_df, ref_azimuth_df, )
savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_all', f'1D_{ref_agents[1].name}.pdf'), close_fig=True)
savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_all_control', f'1D_{ref_agents[1].name}.pdf'), close_fig=True)
savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_all_chance', f'1D_{ref_agents[1].name}.pdf'), close_fig=True)

figs = plot_1d_density_all_bins(ref_agents, stim_dict,  ref_x_df, ref_radius_df, ref_azimuth_df, )
savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_all', f'1D_{ref_agents_str}.pdf'), close_fig=True)
savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_all_control', f'1D_{ref_agents_str}.pdf'), close_fig=True)
savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_all_chance', f'1D_{ref_agents_str}.pdf'), close_fig=True)

# Include individual lines
figs = plot_1d_density(ref_agents[0], stim_dict, ref_x_df, ref_radius_df, ref_azimuth_df, ref_x_ind_df, ref_radius_ind_df, ref_azimuth_ind_df)
savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_ind', f'1D_{ref_agents[0].name}.pdf'), close_fig=True)
savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control_ind', f'1D_{ref_agents[0].name}.pdf'), close_fig=True)
savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance_ind', f'1D_{ref_agents[0].name}.pdf'), close_fig=True)

figs = plot_1d_density(ref_agents[1], stim_dict, ref_x_df, ref_radius_df, ref_azimuth_df, ref_x_ind_df, ref_radius_ind_df, ref_azimuth_ind_df)
savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_ind', f'1D_{ref_agents[1].name}.pdf'), close_fig=True)
savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control_ind', f'1D_{ref_agents[1].name}.pdf'), close_fig=True)
savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance_ind', f'1D_{ref_agents[1].name}.pdf'), close_fig=True)

# Separate plot for figure S1
fig = plot_1d_density_fig_s1(ref_agents, stim_dict, ref_x_df, ref_radius_df, ref_azimuth_df)
savefig(fig, path_to_figS1_folder.joinpath(f'1D_{ref_agents_str}.pdf'), close_fig=True)

# Separate plot for figure 2B
figs4 = plot_1d_density(ref_agents, stim_dict_fig2, ref_x_df, ref_radius_df, ref_azimuth_df, ref_stim_name='azimuth_left_dark_right_bright')
savefig(figs4[0], path_to_fig2B_folder.joinpath(f'1D_{ref_agents_str}.pdf'), close_fig=True)
savefig(figs4[1], path_to_fig2B_folder.joinpath(f'1D_{ref_agents_str}_control.pdf'), close_fig=True)
savefig(figs4[2], path_to_fig2B_folder.joinpath(f'1D_{ref_agents_str}_chance.pdf'), close_fig=True)

# Separate plot for defence: fig2B but reference stim hidden
figs4 = plot_1d_density(ref_agents, stim_dict_fig2, ref_x_df, ref_radius_df, ref_azimuth_df, ref_stim_name='azimuth_left_dark_right_bright', ref_stim_color=COLOR_BACKGROUND)
savefig(figs4[1], path_to_fig2B_folder.joinpath(f'1D_{ref_agents_str}_control_defence_virtual.pdf'), close_fig=True)
closefigs(figs4)

figs4 = plot_1d_density(ref_agents, stim_dict_fig2, ref_x_df, ref_radius_df, ref_azimuth_df, ref_stim_name='azimuth_left_dark_right_bright_virtual_yes', ref_stim_color=COLOR_BACKGROUND)
savefig(figs4[1], path_to_fig2B_folder.joinpath(f'1D_{ref_agents_str}_control_defence.pdf'), close_fig=True)
closefigs(figs4)

# #########################################################################
# Loop over test agents
# #########################################################################
for agent_genotype, agent_dict in agent_mapping.items():
    larva_agent = LarvaAgent()
    juvie_agent = JuvieAgent()
    if not agent_genotype == 'model_ptAV_plST_aAV_tAV_sAV':
        # Overwrite agent class for this genotype
        larva_agent_name = f'{agent_genotype}_05dpf'
        larva_agent.name = larva_agent_name
        larva_agent.query = f'fish_age <= 5 and fish_genotype == "{agent_genotype.lower()}"'
        larva_agent.folder_name = agent_dict['folder_name']
        juvie_agent_name = f'{agent_genotype}_27dpf'
        juvie_agent.name = juvie_agent_name
        juvie_agent.query = f'fish_age >= 21 and fish_genotype == "{agent_genotype.lower()}"'
        juvie_agent.folder_name = agent_dict['folder_name']

    # Define agent string
    test_agents = [larva_agent, juvie_agent]
    test_agents_str = '_'.join([agent.name for agent in test_agents])
    test_agents_str = test_agents_str[:30]  # Limit length of test_agents_str

    # #########################################################################
    # Load and prepare test data
    # #########################################################################
    full_test_tracking_df0 = load_tracking_df(path_to_main_data_folder, path_name, test_agents[0])
    full_test_tracking_df1 = load_tracking_df(path_to_main_data_folder, path_name, test_agents[1])
    if full_test_tracking_df0.empty:
        print(f"Skipping {test_agents[0]}: \033[91mno data\033[0m")
        continue
    elif full_test_tracking_df1.empty:
        print(f"Skipping {test_agents[1]}: \033[91mno data\033[0m")
        continue

    full_test_tracking_df = pd.concat([full_test_tracking_df0, full_test_tracking_df1])

    # Remove values too close to the wall
    test_tracking_df = full_test_tracking_df.loc[full_test_tracking_df['radius'] <= 5].copy()

    # Keep same amount of trials as real fish
    test_tracking_df = test_tracking_df.loc[test_tracking_df.index.get_level_values('trial') < 2].copy()

    # Compute swim properties
    test_tracking_df, test_median_ind_df, test_std_ind_df, n_frames = compute_bins(test_tracking_df)
    test_x_df, test_x_ind_df = compute_swim_properties_tracking(test_tracking_df, n_frames, ['x_bin'])
    test_radius_df, test_radius_ind_df = compute_swim_properties_tracking(test_tracking_df, n_frames, ['radius_bin'])
    test_azimuth_df, test_azimuth_ind_df = compute_swim_properties_tracking(test_tracking_df, n_frames, ['azimuth_bin'])

    # Store individual prob results as numpy arrays for later use in stats
    p_array = get_prob_array(test_agents, stim_dict, test_x_ind_df, test_radius_ind_df, test_azimuth_ind_df)
    store_prob_array(p_array, path_to_stats_file, test_agents_str)

    # Combine reference and test data #########################################
    x_df = pd.concat([ref_x_df, test_x_df])
    radius_df = pd.concat([ref_radius_df, test_radius_df])
    azimuth_df = pd.concat([ref_azimuth_df, test_azimuth_df])
    x_ind_df = pd.concat([ref_x_ind_df, test_x_ind_df])
    radius_ind_df = pd.concat([ref_radius_ind_df, test_radius_ind_df])
    azimuth_ind_df = pd.concat([ref_azimuth_ind_df, test_azimuth_ind_df])

    # #########################################################################
    # Statistics
    # #########################################################################
    # Compute statistics based on mean within fish ############################
    bin_stats_list, stat_df, stat_str = get_bin_stats(test_median_ind_df, test_agents, stim_dict)
    # Store statistics txt file
    with open(path_to_stats_txt, 'a+') as output:
        output.write(stat_str)

    # Plot and store
    fig = plot_bin_stats(test_median_ind_df, stat_df, test_agents, stim_dict)
    savefig(fig, path_to_figS1_folder.joinpath('stats', f'stats_{test_agents_str}.pdf'), close_fig=True)

    # Z-score and MSE #########################################################
    # Between test_agent0 and its reference fish
    stat_list_0 = get_agent_stats(
            test_agents[0].ref_agent, test_agents[0], stim_dict,
            x_ind_df, radius_ind_df, azimuth_ind_df,
    )

    # Between test_agent1 and its reference fish
    stat_list_1 = get_agent_stats(
            test_agents[1].ref_agent, test_agents[1], stim_dict,
            x_ind_df, radius_ind_df, azimuth_ind_df,
    )

    # Store as hdf with separate keys
    stats_df_0 = pd.DataFrame(stat_list_0, columns=stats_columns)
    stats_df_1 = pd.DataFrame(stat_list_1, columns=stats_columns)
    stats_df_0.to_hdf(path_to_stats_file, key=test_agents[0].name, mode='a')
    stats_df_1.to_hdf(path_to_stats_file, key=test_agents[1].name, mode='a')

    # #########################################################################
    # Plot trajectories, 1D and 2D density
    # #########################################################################
    if agent_genotype == 'model_ptAV_plST_aAV_tAV_sAV' or agent_genotype == 'A_DC_A_A_A_wCx5':  # Illustrate these models
        # Plot trajectories ###################################################
        plot_all_trajectories(full_test_tracking_df, test_agents, stim_dict, path_to_figS1_folder)
        for k, (agent, exp_ID) in enumerate(zip(test_agents, [3, 14])):
            fig = plot_exp_trajectories(full_test_tracking_df.xs(exp_ID, level='experiment_ID').query(agent.query), agent, stim_dict, k)
            savefig(fig, path_to_figS1_folder.joinpath('trajectories', 'examples', f'trajectory_{agent.name}.png'), close_fig=False)
            savefig(fig, path_to_figS1_folder.joinpath('trajectories', 'examples', f'trajectory_{agent.name}.pdf'), close_fig=True)

        # Plot 2D density hexbins #############################################
        fig, cbars = plot_2d_density(test_tracking_df, test_agents, stim_dict, vmin=test_vmin, vmax=test_vmax)
        savefig(fig, path_to_figS1_folder.joinpath('2D_density', f'2D_{test_agents_str}.pdf'), close_fig=True)
        savefig(cbars[0], path_to_figS1_folder.joinpath('cbar', f'2D_{test_agents[0].name}.pdf'), close_fig=True)
        savefig(cbars[1], path_to_figS1_folder.joinpath('cbar', f'2D_{test_agents[1].name}.pdf'), close_fig=True)

        # Plot 2D density difference hexbins ##################################
        fig, cbar = plot_2d_density_diff(test_tracking_df, test_agents, stim_dict)
        savefig(fig, path_to_figS1_folder.joinpath('2D_density_dff', f'2D_diff_{test_agents_str}.pdf'), close_fig=True)
        savefig(cbar, path_to_figS1_folder.joinpath('cbar', f'2D_diff_{test_agents_str}.pdf'), close_fig=True)

    # Plot agents separately and together
    figs = plot_1d_density(test_agents[0], stim_dict, test_x_df, test_radius_df, test_azimuth_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density', f'1D_{test_agents[0].name}.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control', f'1D_{test_agents[0].name}.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance', f'1D_{test_agents[0].name}.pdf'), close_fig=True)

    figs = plot_1d_density(test_agents[1], stim_dict, test_x_df, test_radius_df, test_azimuth_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density', f'1D_{test_agents[1].name}.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control', f'1D_{test_agents[1].name}.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance', f'1D_{test_agents[1].name}.pdf'), close_fig=True)

    figs = plot_1d_density(test_agents, stim_dict, test_x_df, test_radius_df, test_azimuth_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density', f'1D_{test_agents_str}.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control', f'1D_{test_agents_str}.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance', f'1D_{test_agents_str}.pdf'), close_fig=True)

    # Plot agents with reference data
    figs = plot_1d_density(test_agents + ref_agents, stim_dict, x_df, radius_df, azimuth_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density', f'1D_{test_agents_str}_w_ref.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control', f'1D_{test_agents_str}_w_ref.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance', f'1D_{test_agents_str}_w_ref.pdf'), close_fig=True)

    figs = plot_1d_density([test_agents[0], ref_agents[0]], stim_dict, x_df, radius_df, azimuth_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density', f'1D_{test_agents[0].name}_w_ref.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control', f'1D_{test_agents[0].name}_w_ref.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance', f'1D_{test_agents[0].name}_w_ref.pdf'), close_fig=True)

    figs = plot_1d_density([test_agents[1], ref_agents[1], ], stim_dict, x_df, radius_df, azimuth_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density', f'1D_{test_agents[1].name}_w_ref.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control', f'1D_{test_agents[1].name}_w_ref.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance', f'1D_{test_agents[1].name}_w_ref.pdf'), close_fig=True)

    # Plot agents for supplementary figure 5
    figs4 = plot_1d_density(test_agents, stim_dict, test_x_df, test_radius_df, test_azimuth_df, ax_x_cm=3, )
    savefig(figs4[0], path_to_figS5_folder.joinpath('1D_density', f'1D_{test_agents_str}.pdf'), close_fig=True)
    savefig(figs4[1], path_to_figS5_folder.joinpath('1D_density_control', f'1D_{test_agents_str}.pdf'), close_fig=True)
    savefig(figs4[2], path_to_figS5_folder.joinpath('1D_density_chance', f'1D_{test_agents_str}.pdf'), close_fig=True)

    figs5 = plot_1d_density(test_agents + ref_agents, stim_dict, x_df, radius_df, azimuth_df, ax_x_cm=3, )
    savefig(figs5[0], path_to_figS5_folder.joinpath('1D_density', f'1D_{test_agents_str}_w_ref.pdf'), close_fig=True)
    savefig(figs5[1], path_to_figS5_folder.joinpath('1D_density_control', f'1D_{test_agents_str}_w_ref.pdf'), close_fig=True)
    savefig(figs5[2], path_to_figS5_folder.joinpath('1D_density_chance', f'1D_{test_agents_str}_w_ref.pdf'), close_fig=True)

    figs5 = plot_1d_density([test_agents[0], ref_agents[0]], stim_dict, x_df, radius_df, azimuth_df, ax_x_cm=3, )
    savefig(figs5[0], path_to_figS5_folder.joinpath('1D_density', f'1D_{test_agents[0].name}_w_ref.pdf'), close_fig=True)
    savefig(figs5[1], path_to_figS5_folder.joinpath('1D_density_control', f'1D_{test_agents[0].name}_w_ref.pdf'), close_fig=True)
    savefig(figs5[2], path_to_figS5_folder.joinpath('1D_density_chance', f'1D_{test_agents[0].name}_w_ref.pdf'), close_fig=True)

    figs5 = plot_1d_density([test_agents[1], ref_agents[1]], stim_dict, x_df, radius_df, azimuth_df, ax_x_cm=3, )
    savefig(figs5[0], path_to_figS5_folder.joinpath('1D_density', f'1D_{test_agents[1].name}_w_ref.pdf'), close_fig=True)
    savefig(figs5[1], path_to_figS5_folder.joinpath('1D_density_control', f'1D_{test_agents[1].name}_w_ref.pdf'), close_fig=True)
    savefig(figs5[2], path_to_figS5_folder.joinpath('1D_density_chance', f'1D_{test_agents[1].name}_w_ref.pdf'), close_fig=True)

    # Plot for all bins
    figs = plot_1d_density_all_bins(test_agents[0], stim_dict, test_x_df, test_radius_df, test_azimuth_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_all', f'1D_{test_agents[0].name}.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_all_control', f'1D_{test_agents[0].name}.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_all_chance', f'1D_{test_agents[0].name}.pdf'), close_fig=True)

    figs = plot_1d_density_all_bins(test_agents[1], stim_dict, test_x_df, test_radius_df, test_azimuth_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_all', f'1D_{test_agents[1].name}.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_all_control', f'1D_{test_agents[1].name}.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_all_chance', f'1D_{test_agents[1].name}.pdf'), close_fig=True)

    figs = plot_1d_density_all_bins(test_agents, stim_dict, test_x_df, test_radius_df, test_azimuth_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_all', f'1D_{test_agents_str}.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_all_control', f'1D_{test_agents_str}.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_all_chance', f'1D_{test_agents_str}.pdf'), close_fig=True)

    # Include individual lines
    figs = plot_1d_density(test_agents[0], stim_dict, test_x_df, test_radius_df, test_azimuth_df, test_x_ind_df, test_radius_ind_df, test_azimuth_ind_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_ind', f'1D_{test_agents[0].name}.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control_ind', f'1D_{test_agents[0].name}.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance_ind', f'1D_{test_agents[0].name}.pdf'), close_fig=True)

    figs = plot_1d_density(test_agents[1], stim_dict, test_x_df, test_radius_df, test_azimuth_df, test_x_ind_df, test_radius_ind_df, test_azimuth_ind_df)
    savefig(figs[0], path_to_figS1_folder.joinpath('1D_density_ind', f'1D_{test_agents[1].name}.pdf'), close_fig=True)
    savefig(figs[1], path_to_figS1_folder.joinpath('1D_density_control_ind', f'1D_{test_agents[1].name}.pdf'), close_fig=True)
    savefig(figs[2], path_to_figS1_folder.joinpath('1D_density_chance_ind', f'1D_{test_agents[1].name}.pdf'), close_fig=True)

    # Close all figures
    plt.close('all')
