# Import packages and default settings
import numpy as np
import pandas as pd

from fig1_helpers import *

from settings.agent_settings import *

# #############################################################################
# User settings
# #############################################################################
# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath('figS7')
path_to_fig_folder.mkdir(exist_ok=True)

# Data generated in fig1_and_4_all_agents.py
path_to_stats1_file = path_to_main_data_folder.joinpath('fig1', 'stats.hdf5')

# Agents
ref_agents = [Larva(), Juvie()]
test_agents = [LarvaAgent(), JuvieAgent()]

# Plot settings
ax1_l, ax1_b, ax1_w, ax1_h = (
    pad_x_cm, pad_y_cm,
    2 * ax_x_cm, ax_y_cm,
)
ax2_l, ax2_b, ax2_w, ax2_h = (
    ax1_l + ax1_w + 2*pad_x_cm, ax1_b,
    ax1_w, ax1_h,
)

# #############################################################################
# Load Performance data
# #############################################################################
try:
    df_performance = pd.read_hdf(path_to_stats1_file, key='performance')
except FileNotFoundError:
    raise FileNotFoundError('First run fig1_and_4_all_agents.py')

# #############################################################################
# Compute BIC
# #############################################################################
ref_agents_str = '_'.join([agent.name for agent in ref_agents])
ref_p_array = read_prob_array(path_to_stats1_file, ref_agents_str)
ref_p_arrays = np.asarray([ref_p_array])  # Wrap in list to keep dimensions consistent with test_p_arrays

test_p_array_list = []
test_agent_names = []
test_agent_labels = []
test_n_pars = []
for agent_name, agent_dict in agent_mapping.items():
    if agent_name == 'A_DC_A_A_A_wCx5':
        continue

    test_agents_str = f'{agent_name}_05dpf_{agent_name}_27dpf'
    test_agents_str = test_agents_str[:30]  # Limit length of test_agents_str
    test_p_array = read_prob_array(path_to_stats1_file, test_agents_str)

    if isinstance(test_p_array, np.ndarray):
        test_p_array_list.append(test_p_array)
        test_agent_names.append(agent_name)
        test_agent_labels.append(agent_dict['label'])
        test_n_pars.append(agent_dict['n_pars'])

test_p_arrays = np.array(test_p_array_list)
test_n_pars_array = np.array(test_n_pars)

# Compute mean over individuals
ref_means = np.nanmean(ref_p_arrays, axis=4)
test_means = np.nanmean(test_p_arrays, axis=4)

# Compute BIC
y_true = ref_means
y_pred = test_means
# Compute log likelihood function
logL = compute_logl_prob_data(y_true, y_pred)
# Get number of parameters and number of data points
k = test_n_pars_array[:, np.newaxis, np.newaxis] - 0 * logL # Trick to broadcast n_pars across all stimuli and ages
n = np.log(y_true.shape[3])
# Compute BIC
BIC_array = n * k + 2 * logL

# Convert to pandas dataframe
# # Build MultiIndex
ages = np.array([5, 27])
index = pd.MultiIndex.from_product(
    [test_agent_labels, ages, stim_names],
    names=['fish_genotype', 'fish_age', 'stimulus']  # Keep identical to fish datasets
)
# # Flatten array
BIC_df = pd.DataFrame({
    'BIC': BIC_array.ravel(),
}, index=index)
# # Remove rows for stimulus 'control' and 'azimuth_left_dark_right_bright_virtual_yes'
BIC_df = BIC_df.loc[~BIC_df.index.get_level_values('stimulus').isin(['control', 'azimuth_left_dark_right_bright_virtual_yes'])].copy()
BIC_df = BIC_df.groupby(level=['fish_genotype', 'fish_age']).mean()

# Sort values by their mean
BIC_mean_df = BIC_df.groupby(level='fish_genotype')['BIC'].mean()
BIC_mean_df.sort_values(ascending=True, inplace=True)
BIC_sorted_df = BIC_df.sort_index(
    level='fish_genotype',
    key=lambda x: x.map(BIC_mean_df),
    ascending=True
)

# #############################################################################
# Create figure
# #############################################################################
fig = create_figure(fig_width=fig_width_cm, fig_height=fig_height_cm / 2)

# Fig S7A: Performance ########################################################
# Create ax
ax = add_axes(fig, ax1_l, ax1_b, ax1_w, ax1_h)

for label, df in df_performance.groupby('fish_genotype'):
    n_par = df['n_par'].iloc[0]
    perf_mean = df['Performance'].mean()
    if label == 'Proposed*':
        # Fine-tuned model is only for larval data
        perf_mean = df.xs('larva', level='ref_agent')['Performance'].mean()

    ax.scatter(
        n_par, perf_mean,
        marker='o', s=MARKER_SIZE_LARGE, color='k', zorder=10
    )

    # Label and label positioning
    s = label
    text_x = n_par + 1
    text_y = perf_mean
    ha = 'left'
    if s == 'A':
        # Special case for 'A' to avoid overlap with other labels
        text_x = n_par - 1
        ha = 'right'
    ax.text(
        text_x, text_y, s,
        fontsize=6, color='black',
        ha=ha, va='center'
    )

# Format
y_ticks = [-25, 0, 25, 50, 75, 100]
set_axlines(ax, axhlines=0, hlim=(0, 70), color='grey')
set_ticks(ax, x_ticks=np.linspace(0, 70, 8), y_ticks=y_ticks)
set_bounds(ax, x=(0, 70), y=(min(y_ticks), max(y_ticks)))
set_lims(ax, x=(0, 70))  # , y=bottom=-1, max(y_ticks)))  # Ensure lowest point is visible
set_labels(ax, x='Nr. parameters', y='Mean performance\nscore (a.u.)')
hide_spines(ax, ['top', 'right'])
set_spine_position(ax)

# Fig S7B: BIC ################################################################
y_ticks = np.linspace(0, 400, 5)

# Create ax
ax = add_axes(fig, ax2_l, ax2_b, ax2_w, ax2_h )

sns.barplot(
    data=BIC_sorted_df.reset_index(),
    x='fish_genotype', y='BIC',
    hue='fish_age', palette={5: LarvaAgent().color, 27: JuvieAgent().color},
    legend=False, ax=ax,
)

# Format
set_labels(ax, x='', y='BIC')
plt.xticks(rotation=60, ha='center')
set_ticks(ax, x_ticksize=0, y_ticks=y_ticks)
set_bounds(ax, y=(min(y_ticks), max(y_ticks)))
hide_spines(ax, ['top', 'right', 'bottom'])

# Highlight lowest value for each age group
for (fish_age, age_df), color in zip(BIC_df.groupby('fish_age'), [LarvaAgent().color, JuvieAgent().color]):
    min_value = age_df['BIC'].min()
    ax.axhline(min_value, color='k', linestyle='--', zorder=-100)

savefig(fig, path_to_fig_folder.joinpath('figS7.pdf'), close_fig=False)
