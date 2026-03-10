"""
Create figure 4B
Compute performance scores, also used for fig S5, fig S7
"""

# Import packages and default settings
import numpy as np
import pandas as pd

from settings.general_settings import *
from fig1_helpers import *

from settings.agent_settings import *
from settings.general_settings import *
from settings.plot_settings import *

# #############################################################################
# User settings
# #############################################################################
# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath('fig1_and_4')

# Data generated in fig1_and_4_all_agents.py
path_to_stats1_file = path_to_main_data_folder.joinpath('fig1_and_4', 'stats.hdf5')
if not path_to_stats1_file.exists():
    raise FileNotFoundError('First run fig1_and_4_all_agents.py')

# #############################################################################
# Compute performance scores
# #############################################################################
# Load distribution data
score_df_list = []
for agent_name, agent_dict in agent_mapping.items():
    for age_str in ['_05dpf', '_27dpf']:
        try:
            _df = pd.read_hdf(path_to_stats1_file, key=agent_name + age_str)
            _df['fish_genotype'] = agent_dict['label']
            _df['n_par'] = agent_dict['n_pars']

            score_df_list.extend([_df])
        except Exception as e:
            if agent_name == agent_base_norm:
                raise ValueError(
                    f"Error loading data for {agent_name}{age_str}: {e}. "
                    "Ensure the agent_base_norm is correctly specified."
                )
            # # If the agent is not found, skip it
            # print(f"\033[93mError loading data for {agent_name}{age_str}: {e}\033[0m")
            continue

full_score_df = pd.concat(score_df_list)

# Drop rows where index 'stimulus' is 'control' or 'azimuth_left_dark_right_bright_virtual_yes'
full_score_df = full_score_df[~full_score_df['stimulus'].isin(['control', 'azimuth_left_dark_right_bright_virtual_yes'])]

# Clean up dataframe
full_score_df.drop(columns=[
    'bin',
    'test_agent', 'stimulus'
], inplace=True)
# Set index
full_score_df.set_index([ # Set index to keep overview
    'ref_agent', 'fish_genotype',
    'do_subtract_control',
    'do_bootstrap', 'i_bootstrap',
], inplace=True)

# Compute mean over bootstrapped values
mean_score_df = (
    full_score_df
    .xs(False, level='do_subtract_control')  # Keep only original data
    .xs(False, level='do_bootstrap')  # Keep only bootstrapped data
    .groupby(['ref_agent', 'fish_genotype'])  # Group by relevant levels
    .mean()  # Compute mean over bootstrapped values
)

# Compute performance: normalise scores relative to blind agent ###############
control_df = mean_score_df.query("fish_genotype == @agent_base_norm")
# Merge control values onto the main dataframe
df_performance = (
    mean_score_df
    .reset_index('fish_genotype')
    .merge(
        control_df,
        on=['ref_agent'],
        suffixes=('', '_control')
    )
    .set_index('fish_genotype', append=True)
)
# Normalise values compared to control for MSE
df_performance['Performance'] = (
    df_performance['MSE_control'] - df_performance['MSE']
).div(df_performance['MSE_control'].abs()) * 100

# Ensure larval agents are plotted before juvenile agents
df_performance.sort_values('ref_agent', ascending=False, inplace=True)

# Store to disk
df_performance.to_hdf(path_to_stats1_file, key='performance')

# #############################################################################
# Figure 4B
# #############################################################################
# Figure settings
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

# Barplot
barplot = sns.barplot(
    data=df_performance,
    x='fish_genotype', y='Performance',
    hue='ref_agent', palette={'larva': LarvaAgent().color, 'juvie': JuvieAgent().color}, width=bar_width,
    order=fig4_order,
    errorbar=None,
    legend=False, ax=ax
)
set_axlines(ax, axhlines=0, hlim=(-bar_width/2, len(fig4_order)-1 +bar_width/2), zorder=-100)
plt.xticks(rotation=30, ha='center')
set_ticks(ax, x_ticksize=0, y_ticks=y_ticks)
set_bounds(ax, y=(min(y_ticks), max(y_ticks)))
set_labels(ax, x='', y='Performance score\n(a.u.)')
hide_spines(ax, ['top', 'right', 'bottom'])
savefig(fig, path_to_main_fig_folder.joinpath('fig1_and_4', f'fig4B_performance.pdf'), close_fig=False)
