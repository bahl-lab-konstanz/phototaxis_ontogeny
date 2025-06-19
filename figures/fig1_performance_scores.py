# Import packages, default settings and functions for figure 1
from fig1_helpers import *

from settings.agent_settings import *
from settings.general_settings import *
from settings.plot_settings import *

# #############################################################################
# User settings
# #############################################################################
# Paths
experiment_name = 'arena_locked'
path_name = experiment_name

path_to_fig_folder = path_to_main_fig_folder.joinpath(f'fig1_{experiment_name}')
path_to_fig_folder.mkdir(exist_ok=True)
path_to_stats_file = path_to_fig_folder.joinpath(f'stats_250612.hdf5')

# Agents
ref_agents = [Larva(), Juvie()]
test_agents = [LarvaAgent(), JuvieAgent()]

# Plot settings
ax_x_cm, ax_y_cm = 10, 3  # cm
metrics = ['SSD', 'Z-score', 'MSE', 'KL', ]
metric_suffices = ['', '_BIC', '_AIC']

# Agent mappings ##############################################################
agent_base_norm = 'Blind'
agent_mapping = {
    'model_ptB_plB_aB_tB_sB': (agent_base_norm, 5),     # Blind
    'A_B_A_A_A':    ('A', 4*2+1),   # Averaging pathway (Fig. 2)
    'B_C_B_B_B': ('C', 6),          # Contrast pathway
    'B_D_B_B_B':    ('D', 7),       # Derivative pathway
    'B_DC_B_B_B': ('D+C', 8),       # Contrast + Derivative (Fig. 3)
    'A_DC_A_A_A_wCx5': ('Proposed*', 4*2 + 4),
    'superfit': ('Five-pathways', 5 * 11),
    'model_ptT_plT_aT_tT_sT': ('AD', 5 * 4),    # Averaging-derivative pathway (Fig. S4)
    'DA_DA_DA_DA_DA': ('DA', 5 * 4),
    'model_ptAV_plST_aAV_tAV_sAV': ('Proposed', 4*2 + 4),
}

# Ensure order in plot
main_order = ['A', 'C', 'D', 'D+C', 'Proposed', ]
supp_order = [
    agent_base_norm, 'A', 'C',
    'D', 'D+C', 'Proposed', 'Proposed*',
    'Five-pathways',
    'AD', 'DA',
]

# #############################################################################
# Load and prepare data
# #############################################################################
score_df_list = []
for agent_name in agent_mapping.keys():
    for age_str in ['_05dpf', '_27dpf']:
        try:
            score_df_list.extend([
                pd.read_hdf(path_to_stats_file, key=agent_name + age_str),
            ])
        except Exception as e:
            if agent_name == agent_base_norm:
                raise ValueError(
                    f"Error loading data for {agent_name}{age_str}: {e}. "
                    "Ensure the agent_base_norm is correctly specified."
                )
            # If the agent is not found, skip it
            print(f"Error loading data for {agent_name}{age_str}: {e}")
            continue

full_score_df = pd.concat(score_df_list)


# Add agent_category ##########################################################
# Separate age and base
def extract_age(x):
    age_str = x.split('_')[-1]
    if age_str == 'juvie':
        return 27
    return int(age_str.replace('dpf', ''))  # Convert '05dpf' -> 5, '27dpf' -> 27

full_score_df['test_agent_age'] = full_score_df['test_agent'].apply(extract_age).astype(int)
full_score_df['test_agent_base'] = full_score_df['test_agent'].str.replace('_27dpf', '')
full_score_df['test_agent_base'] = full_score_df['test_agent_base'].str.replace('_05dpf', '')

# Agent name mapping: add agent_category and n_par
# Map the values to a tuple
mapped = full_score_df['test_agent_base'].map(agent_mapping)
# Split the tuple into two new columns
full_score_df['agent_category'] = mapped.apply(lambda x: x[0] if x else None)
full_score_df['n_par'] = mapped.apply(lambda x: x[1] if x else None)

# Clean up dataframe
full_score_df.drop(columns=[
    'bin',
    'test_agent', 'test_agent_age', 'test_agent_base',
], inplace=True)


# Set index ###################################################################
_score_df = (
    full_score_df
    .set_index([  # Set index to keep overview
        'ref_agent', 'agent_category',
        'stimulus',
        'do_subtract_control',
        'do_bootstrap', 'i_bootstrap',
    ])
    .xs(False, level='do_subtract_control')  # Keep only original data
    .xs(False, level='do_bootstrap')     # Keep only bootstrapped data
)

# Add ref_agent as mean of larva and juvie #####################################
df = (_score_df.xs('larva', level='ref_agent')[metrics + ['n_par']] + _score_df.xs('juvie', level='ref_agent')[metrics + ['n_par']])/2
df['ref_agent'] = 'mean'
score_df = (
    pd.concat([_score_df.reset_index(), df.reset_index()], axis=0)
    .set_index([  # Set index to keep overview
        'ref_agent', 'agent_category',
        'stimulus',
        'i_bootstrap',
    ])
)

# Compute mean over stimuli ###################################################
mean_score_df = (
    score_df
    .loc[
        pd.IndexSlice[
        :, :,
        ['splitview_left_dark_right_bright', 'azimuth_left_dark_right_bright',
         'center_bright_outside_dark', 'center_dark_outside_bright', ],
        :, :
        ]
    ]
    .groupby([
        'ref_agent', 'agent_category', 'i_bootstrap',
        # 'do_bootstrap',
        # 'test_agent', 'bin', 'test_agent_age', 'test_agent_base',
    ])
    .mean()
)

# Compute AIC and BIC #########################################################
n = 4 * nbins  # Number of observations (4 stimuli times nbins bins)
mean_score_df['SSD_BIC'] = mean_score_df['n_par'] * np.log(n) - 2 * np.log(mean_score_df['SSD'])
mean_score_df['SSD_AIC'] = 2 * mean_score_df['n_par'] - 2 * np.log(mean_score_df['SSD'])
mean_score_df['Z-score_BIC'] = mean_score_df['n_par'] * np.log(n) - 2 * np.log(mean_score_df['Z-score'])
mean_score_df['Z-score_AIC'] = 2 * mean_score_df['n_par'] - 2 * np.log(mean_score_df['Z-score'])
mean_score_df['MSE_BIC'] = mean_score_df['n_par'] * np.log(n) - 2 * np.log(mean_score_df['MSE'])
mean_score_df['MSE_AIC'] = 2 * mean_score_df['n_par'] - 2 * np.log(mean_score_df['MSE'])
mean_score_df['KL_BIC'] = mean_score_df['n_par'] * np.log(n) - 2 * np.log(mean_score_df['KL'])
mean_score_df['KL_AIC'] = 2 * mean_score_df['n_par'] - 2 * np.log(mean_score_df['KL'])

# Compute performance: normalise Z-scores relative to blind agent #############
control_df = (
    mean_score_df
    .query("agent_category == @agent_base_norm")
)
# Merge control values onto the main dataframe
norm_score_df = (
    mean_score_df
    .reset_index('agent_category')
    .merge(
        control_df,
        on=['ref_agent', 'i_bootstrap'],
        suffixes=('', '_control')
    )
    .set_index('agent_category', append=True)
)

# Normalise values compared to control for MSE
norm_score_df['Performance'] = (
    norm_score_df['MSE_control'] - norm_score_df['MSE']
).div(norm_score_df['MSE_control'].abs()) * 100


# #############################################################################
# Figures
# #############################################################################
# Ensure larval agents are plotted before juvenile agents
mean_score_df.sort_values('ref_agent', ascending=False, inplace=True)
norm_score_df.sort_values('ref_agent', ascending=False, inplace=True)

# Normalised Z-scores (main) ##################################################
pad_y_cm = 3  # cm, extra padding for model labels
ax_y_cm = 1.8   # cm, full axis height (incl. ticks and labels)
ax_x_cm = 3   # cm
bar_width = 0.7  # Width of the bars in the barplot
y_ticks = [0, 15, 30]

fig = create_figure(fig_height=2 * pad_y_cm + ax_y_cm, fig_width=2*pad_x_cm + ax_x_cm)
# Create ax
i, j = 0, 0
l, b, w, h = (
    pad_x_cm + i * ax_x_cm,
    pad_y_cm + j * (ax_y_cm + pad_y_cm),
    ax_x_cm,
    ax_y_cm
)
ax = add_axes(fig, l, b, w, h)

# Barplot for normalised values
barplot = sns.barplot(
    data=norm_score_df.drop('mean', level='ref_agent'),  # Remove mean level
    x='agent_category', y='Performance',
    hue='ref_agent', palette={'larva': LarvaAgent().color, 'juvie': JuvieAgent().color}, width=bar_width,
    order=main_order,
    errorbar=None,
    legend=False, ax=ax
)
set_axlines(ax, axhlines=0, hlim=(-bar_width/2, len(main_order)-1 +bar_width/2), zorder=-100)
plt.xticks(rotation=30, ha='center')
set_ticks(ax, x_ticksize=0, y_ticks=y_ticks)
set_bounds(ax, y=(min(y_ticks), max(y_ticks)))
set_labels(ax, x='', y='Performance score\n(a.u.)')
hide_spines(ax, ['top', 'right', 'bottom'])
savefig(fig, path_to_fig_folder.joinpath('performance_scores', f'performance.pdf'), close_fig=True)

# Normalised Z-scores (supplementary) #########################################
grid_y_cm = 2.4   # cm, full axis height (incl. ticks and labels)
grid_x_cm = 3   # cm
ax_y_cm = 1.5
ax_x_cm = 1   # cm
# pad_y_cm = 1.5  # cm
# pad_x_cm = 2  # cm
bar_width = 0.2  # Width of the bars in the barplot

y_ticks = [0, 50, 100]  # Y-ticks for the supplementary plots
y_ticks = np.linspace(0, 90, 4)

fig = create_figure(fig_height=25, fig_width=5)
for k, agent_category in enumerate(supp_order[::-1]):  # Reverse order for plotting
    if not len(agent_category):
        # Skip empty agent_category for spacing
        continue

    i = 0  # Column index
    j = k - 1  # Row index
    l, b, w, h = (
        pad_x_cm + i * grid_x_cm,
        pad_y_cm + j * grid_y_cm,
        ax_x_cm,
        ax_y_cm
    )
    ax = add_axes(fig, l, b, w, h)

    # Barplot for normalised values
    barplot = sns.barplot(
        data=norm_score_df.drop('mean', level='ref_agent'),  # Remove mean level
        x='agent_category', y='Performance',
        hue='ref_agent', palette={'larva': LarvaAgent().color, 'juvie': JuvieAgent().color}, width=bar_width,
        order=[agent_category],
        errorbar=None,
        legend=False, ax=ax
    )
    set_axlines(ax, axhlines=0, hlim=(-bar_width, + bar_width), zorder=-100)
    # plt.xticks(rotation=90, ha='center')

    set_ticks(ax, x_ticksize=0, y_ticks=y_ticks)
    set_bounds(ax, y=(min(y_ticks), max(y_ticks)))
    set_lims(ax, y=(-5, max(y_ticks)))  # Ensure axhline is visible

    set_labels(ax, x='', y='Performance\nscore (a.u.)')
    hide_spines(ax, ['top', 'right', 'bottom'])
    set_spine_position(ax)
savefig(fig, path_to_fig_folder.joinpath('performance_scores', f'performance_suppl.pdf'), close_fig=True)

# Performance as function of number of parameters #############################
# Add larval value for 'Proposed*' agent to mean
norm_score_df.iloc[0] = norm_score_df.xs('larva', level='ref_agent').xs('Proposed*', level='agent_category').copy()

metric = 'Performance'
fig = create_figure(fig_width=12.1, fig_height=6.5)
l, b, w, h = (
    pad_x_cm,
    1,
    8,
    3
)
ax = add_axes(fig, l, b, w, h)
ax.scatter(
    norm_score_df.xs('mean', level='ref_agent').n_par,
    norm_score_df.xs('mean', level='ref_agent')[metric],
    marker='o', s=MARKER_SIZE_LARGE, color='k', zorder=10
)

# Collect all text objects in a list
plot_df = norm_score_df.xs('mean', level='ref_agent').reset_index()
texts = []
for _, row in plot_df.iterrows():
    x = row['n_par'] + 0.5
    y = row[metric]
    s = str(row['agent_category'])
    ha = 'left'
    if s == 'C':
        # Special case for 'C' to avoid overlap with other labels
        x = row['n_par'] - 1
        ha = 'right'

    texts.append(
        ax.text(
            x, y, s,
            fontsize=6, color='black',
            ha=ha, va='top',
        )
    )

# Format
y_ticks = [-25, 0, 25, 50, 75, 100]
set_axlines(ax, axhlines=0, hlim=(0, 55), color='grey')
set_ticks(ax, x_ticks=np.linspace(0, 55, 12), y_ticks=y_ticks)
set_bounds(ax, x=(0, 55), y=(min(y_ticks), max(y_ticks)))
set_lims(ax, x=(0, 56))  # , y=bottom=-1, max(y_ticks)))  # Ensure lowest point is visible
set_labels(ax, x='Nr. parameters', y='Mean performance\nscore (a.u.)')
hide_spines(ax, ['top', 'right'])
set_spine_position(ax)
savefig(fig, path_to_fig_folder.joinpath('performance_scores', f'mean_performance.pdf'), close_fig=True)
