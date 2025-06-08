# Import default settings and functions for figure 2
from fig2_helpers import load_dunn_df, count_turn_pairs, get_streak_length_distributions, cumulative_turn_direction

# Third party library imports
from scipy.stats import ranksums, mannwhitneyu

# Local library imports
from utils.general_utils import load_tracking_df, load_event_df, get_n_fish, get_median_df
from utils.plot_utils import *
from settings.agent_settings import Larva, Juvie
from settings.plot_settings import *
from settings.prop_settings import *
from settings.general_settings import path_to_main_fig_folder, path_to_main_data_folder, turn_threshold


# #############################################################################
# User settings
# #############################################################################
# Import stimulus settings
stim_name = 'azimuth_left_dark_right_bright_virtual_yes'
path_name = 'arena_locked'

# Dunn et al. 2016 settings
totsize = 20                # the maximum number of turns in a sequence to plot
turn_threshold_dunn = 0     # the minimum angle change to be considered a turn (and switch)

# Agents
agents = [Larva(), Juvie()]
agents_str = '_and_'.join([agent.name for agent in agents])

# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath(f'figS2_turnsequences')
path_to_fig_folder.mkdir(exist_ok=True)
path_to_stat_file = path_to_fig_folder.joinpath('stats.txt')
stat_str = ''

# #############################################################################
# Load data
# #############################################################################
full_event_df = load_event_df(path_to_main_data_folder, path_name, agents)
event_df = full_event_df.xs(stim_name, level='stimulus_name', drop_level=False).copy()
control_event_df = full_event_df.xs('control', level='stimulus_name', drop_level=False).copy()

n_fish = get_n_fish(event_df, agents)
n_fish = get_n_fish(control_event_df, agents)

# Load data from Dunn et al. 2016 and convert to our data format
path_to_mat_file = path_to_main_data_folder.joinpath('dunn2016', 'figure1e_sourcedata.mat')
dunn_df = load_dunn_df(path_to_mat_file)

# Remove wall interactions
event_df = event_df.loc[event_df['radius'] <= 5].copy()  # cm
control_event_df = control_event_df.loc[control_event_df['radius'] <= 5].copy()  # cm

# #############################################################################
# Analysis and figures
# #############################################################################
for angle_threshold in [turn_threshold_dunn, turn_threshold]:
    for df, name in zip([event_df, control_event_df, dunn_df], [stim_name, 'control', 'dunn']):
        stat_str += f"Stimulus: {name} | angle threshold: {angle_threshold}\n"

        fig, axes = plt.subplots(
            nrows=4, ncols=2, sharex='row', sharey='row',
            figsize=(2 * (pad_x_cm + ax_x_cm) * cm * 3/4, 3*(pad_y_cm + ax_y_cm) * cm),
        )

        for i, agent in enumerate(agents):
            agent_df = df.query(agent.query)

            if agent_df.empty:
                # Dunn dataframe does not have all age categories
                continue

            # Turn pairs ######################################################
            counted_pair_df, order = count_turn_pairs(agent_df, angle_threshold)

            j = 0
            ax = axes[j, i]

            # Plot barplot: mean and 95% CI over individuals
            sns.barplot(
                data=counted_pair_df,
                x='turn_pair', y='count', hue='kind',
                order=order,
                estimator='mean', errorbar='se',
                palette={'real': agent.color, 'shuffled': 'lightgrey'},
                legend=True, ax=ax
            )

            # Format
            set_labels(ax, x='Turn pair', y='Nr. paired events')
            hide_spines(ax)
            set_spine_position(ax, ['left'])
            if i != 0:
                # Only set step size for last row
                set_ticks(ax, y_step=100)
                # Hide y axis
                ax.set_ylabel('')
                hide_spines(ax, ['left'])
                ax.tick_params(axis='y', size=0)

            # Add legend
            # # Add empty handles for legend of lower rows
            ax.plot([], [], color=agent.color, label=agent.label, linestyle='solid', linewidth=1, )
            ax.plot([], [], color='grey', label='Random', linestyle='dashed', linewidth=1, )
            # # Rename barplot labels
            handles, labels = ax.get_legend_handles_labels()
            labels[:2] = agent.label, 'Shuffled'
            set_legend(ax, handles=handles, labels=labels, loc='lower right', bbox_to_anchor=(1.1, 1.1))

            # Get streak length distribution for this agent ###########################
            streak_hist_df, streak_bincenters = get_streak_length_distributions(agent_df, angle_threshold, do_shuffle=False)
            streak_hist_shuffle_df, _ = get_streak_length_distributions(agent_df, angle_threshold, do_shuffle=True)

            # Histogram
            # Compute mean across fish
            n_fish = streak_hist_df['experiment_ID'].nunique()
            mean = streak_hist_df.groupby(['bins'])['streak_hist'].mean().values
            shuffled_mean = streak_hist_shuffle_df.groupby(['bins'])['streak_hist'].mean().values

            # Compute statistics
            stat, pvalue = ranksums(mean, shuffled_mean)
            stat_str += f"\t{agent.name}:\t\t{p_value_to_stars(pvalue)}\tp={pvalue: .5f}\tstat={stat: .2f}\t(Ranksum test data vs random)\n"
            m_stat, m_p_value = mannwhitneyu(mean, shuffled_mean, alternative='two-sided')  # data is not normally distributed
            stat_str += f"\t{agent.name}:\t\t{p_value_to_stars(m_p_value)}\tp={m_p_value: .5f}\tstat={m_stat: .2f}\t(MWU test data vs random)\n"

            j = 1
            ax = axes[j, i]

            ax.plot(
                streak_bincenters, mean,
                color=agent.color, label=agent.name,
                linestyle='solid', linewidth=1, zorder=+100,
            )
            ax.plot(
                streak_bincenters, shuffled_mean,
                color='grey', label='Random',
                linestyle='dashed', linewidth=1, zorder=-100,
            )
            # Add statistics
            x_center = (streak_bincenters[0] + streak_bincenters[-1]) / 2
            # add_stats(ax, x_center, x_center, ANNOT_Y, p_value_to_stars(m_p_value))
            add_stats(ax, x_center, x_center, ANNOT_Y, f'p={m_p_value:3.2f}')

            set_ticks(ax, x_ticks=[0, 5, 10, 15], y_ticks=np.linspace(0, 1, 5))
            set_bounds(ax, x=[0, 15], y=[0, 1])
            hide_spines(ax)
            set_spine_position(ax)
            set_labels(ax, x='Streak length', y='Prob')
            if i != 0:
                # Hide y axis
                ax.set_ylabel('')
                hide_spines(ax, ['left'])
                ax.tick_params(axis='y', size=0)

            # Cumulative
            # Compute mean across fish
            n_fish = streak_hist_df['experiment_ID'].nunique()
            mean = streak_hist_df.groupby(['bins'])['streak_hist_cumsum'].mean().values
            std = streak_hist_df.groupby(['bins'])['streak_hist_cumsum'].mean().values
            sem = std / np.sqrt(n_fish)

            shuffled_mean = streak_hist_shuffle_df.groupby(['bins'])['streak_hist_cumsum'].mean().values
            shuffled_std = streak_hist_shuffle_df.groupby(['bins'])['streak_hist_cumsum'].mean().values
            shuffled_sem = shuffled_std / np.sqrt(n_fish)

            j = 2
            ax = axes[j, i]

            ax.plot(
                streak_bincenters, mean,
                color=agent.color, label=agent.name,
                linestyle='solid', linewidth=1, zorder=+100,
            )
            ax.plot(
                streak_bincenters, shuffled_mean,
                color='grey', label='Random',
                linestyle='dashed', linewidth=1, zorder=-100,
            )

            # Add statistics
            x_center = (streak_bincenters[0] + streak_bincenters[-1]) / 2
            # add_stats(ax, x_center, x_center, ANNOT_Y, p_value_to_stars(m_p_value))
            add_stats(ax, x_center, x_center, ANNOT_Y, f'p={m_p_value:3.2f}')

            set_ticks(ax, x_ticks=[0, 5, 10, 15], y_ticks=np.linspace(0, 1, 5))
            set_bounds(ax, x=[0, 15], y=[0, 1])
            hide_spines(ax)
            set_spine_position(ax)
            set_labels(ax, x='Streak length', y='Cumulative prob.')
            if i != 0:
                # Hide y axis
                ax.set_ylabel('')
                hide_spines(ax, ['left'])
                ax.tick_params(axis='y', size=0)

            # Compute cumulative turn direction for this agent ########################
            fish_mean, n_fish = cumulative_turn_direction(agent_df, totsize, angle_threshold)
            shuffled_fish_mean, _ = cumulative_turn_direction(agent_df, totsize, angle_threshold, shuffle=True)

            # Compute mean across fish
            mean = np.nanmean(fish_mean, axis=0)
            sem = np.nanstd(fish_mean, axis=0) / np.sqrt(n_fish)
            shuffled_mean = np.nanmean(shuffled_fish_mean, axis=0)
            shuffled_sem = np.nanstd(shuffled_fish_mean, axis=0) / np.sqrt(n_fish)

            j = 3
            ax = axes[j, i]

            ax.plot(
                mean,
                color=agent.color, label=agent.name,
                linestyle='solid', linewidth=1, zorder=+100)
            ax.fill_between(
                np.arange(len(mean)), mean - sem, mean + sem,
                color=agent.color, alpha=0.3, zorder=+100)
            # Plot shuffled data
            ax.plot(
                shuffled_mean,
                color='grey', label='Random',
                linestyle='dashed', linewidth=1, zorder=-100,
            )
            ax.fill_between(
                np.arange(len(shuffled_mean)), shuffled_mean - shuffled_sem, shuffled_mean + shuffled_sem,
                color='grey', alpha=0.3, zorder=+100)

            ax.set_ylim([-0.1, 1])
            set_ticks(ax, x_ticks=[0, 5, 10, 15, 20], y_ticks=np.linspace(0, 1, 5))
            set_bounds(ax, x=[0, 20], y=[0, 1])
            hide_spines(ax)
            set_spine_position(ax)
            set_labels(ax, x='# Turns', y='Cumulative\nturn direction')
            if i != 0:
                # Hide y axis
                ax.set_ylabel('')
                hide_spines(ax, ['left'])
                ax.tick_params(axis='y', size=0)

        fig.suptitle(name)
        fig.tight_layout()
        savefig(fig, path_to_fig_folder.joinpath(f'angle-threshold{angle_threshold:02d}_{name}.pdf'), close_fig=True)

print(stat_str)
with open(path_to_stat_file, "w") as text_file:
    text_file.write(stat_str)
