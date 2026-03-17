
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
path_to_fig_folder = path_to_main_fig_folder.joinpath('figS7')
path_to_fig_folder.mkdir(exist_ok=True)


# #############################################################################
# Performance
# #############################################################################
# Load data
path_to_stats1_file = path_to_main_data_folder.joinpath('fig1').joinpath('stats.hdf5')
try:
    df_performance = pd.read_hdf(path_to_stats1_file, key='performance')
except FileNotFoundError:
    raise FileNotFoundError('First run fig1_and_4_all_agents.py')
except KeyError:
    raise FileNotFoundError('First run fig4B.py')

# Ensure data is organised by reference agent
df_performance.sort_values('ref_agent', ascending=False, inplace=True)

# Figure S5 ###################################################################
grid_y_cm = 2.4   # cm, full axis height (incl. ticks and labels)
grid_x_cm = 3   # cm
ax_y_cm = 1.5
ax_x_cm = 1   # cm
bar_width = 0.2  # Width of the bars in the barplot
y_ticks = [0, 50, 100]  # Y-ticks for the supplementary plots
y_ticks = np.linspace(0, 90, 4)


fig = create_figure(fig_height=30, fig_width=5)
for k, agent_category in enumerate(figS7_order[::-1]):  # Reverse order for plotting
    i = 0  # Column index
    j = k  # Row index
    l, b, w, h = (
        pad_x_cm + i * grid_x_cm,
        pad_y_cm + j * grid_y_cm,
        ax_x_cm,
        ax_y_cm
    )
    ax = add_axes(fig, l, b, w, h)

    # Barplot for normalised values
    barplot = sns.barplot(
        data=df_performance,
        x='fish_genotype', y='Performance',
        hue='ref_agent', palette={'larva': LarvaAgent().color, 'juvie': JuvieAgent().color}, width=bar_width,
        order=[agent_category],
        errorbar=None,
        legend=False, ax=ax
    )
    set_axlines(ax, axhlines=0, hlim=(-bar_width, + bar_width), zorder=-100)

    set_ticks(ax, x_ticksize=0, y_ticks=y_ticks)
    set_bounds(ax, y=(min(y_ticks), max(y_ticks)))
    set_lims(ax, y=(-5, max(y_ticks)))  # Ensure axhline is visible

    set_labels(ax, x='', y='Performance\nscore (a.u.)')
    hide_spines(ax, ['top', 'right', 'bottom'])
    set_spine_position(ax)
savefig(fig, path_to_main_fig_folder.joinpath('figS5', f'figS5.pdf'), close_fig=False)
