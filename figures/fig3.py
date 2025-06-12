# Standard library imports
import datetime

# Local library imports
from fig3_helpers import get_peaks
from utils.plot_utils import *
from utils.general_utils import load_event_df, load_median_df, get_median_df_time, get_n_fish, get_b_values, get_stats_two_groups
from utils.models import ModelD_C
from settings.general_settings import path_to_main_fig_folder, path_to_main_data_folder
from settings.agent_settings import *
from settings.prop_settings import *
from settings.plot_settings import *


# #############################################################################
# User settings
# #############################################################################
# Stimulus settings
from settings.stim_brightness_choice_simple import *

# Paths
path_to_fig_folder = path_to_main_fig_folder.joinpath(f'fig3_{experiment_name}')
path_to_fig_folder.mkdir(exist_ok=True)

# Agents
agents = [Larva(), Juvie()]
agents_str = '_and_'.join([agent.name for agent in agents])

# Properties
prop_class = PercentageLeft()

# Models
model = ModelD_C()
hdf5_file = path_to_fig_folder.joinpath('models', 'fit_dfs', f'fit_df_{model.name}.hdf5')
key_base = model.name

# Create stat str and file for this experiment
path_to_stat_file = path_to_fig_folder.joinpath(f'stats_{experiment_name}.txt')
stat_str = (f'Statistics for figure 3 {agents[0].name} and {agents[1].name}\n'
            f'\t{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}')

# Plot settings
fig_width_cm = 12.1         # cm
ax_x_cm = fig_width_cm / 5  # cm
ax_y_cm = fig_width_cm / 5  # cm
pad_x_cm = 1
pad_y_cm = 1


# #############################################################################
# Load and prepare data
# #############################################################################
# Load data
event_df = load_event_df(path_to_main_data_folder, experiment_name, agents)
median_df = get_median_df_time(event_df, resampling_window)
n_fish_dict = get_n_fish(event_df, agents)

if not hdf5_file.exists():
    raise FileNotFoundError(
        f'Run "fig3_fit_models.py" first to create the fit data file: {hdf5_file}'
    )

ind_fit_df = pd.read_hdf(hdf5_file, key=f'{key_base}_meta')
fit_df = pd.read_hdf(hdf5_file, key=f'{key_base}_meta_mean')

fig = create_figure(fig_width_cm + pad_x_cm, fig_width_cm)

# #############################################################################
# Fig 3B: time series
# #############################################################################
# Specify time and brightness for plotting
dt_hat = 1
ts_hat = np.arange(dt_hat / 2, t_ns[-1] + dt_hat / 2, dt_hat)
b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)

# Prepare axis
i = 2
time_pad_y_cm = 0.5
time_x_cm = 3 * ax_x_cm
time_y_cm = 2  # cm
offset_x_cm = i * ax_x_cm
offset_y_cm = 2 * ax_y_cm + pad_y_cm
for j, agent in enumerate(agents[::-1]):
    l, b, w, h = (
        offset_x_cm,
        offset_y_cm + j * (time_y_cm),
        time_x_cm,
        time_y_cm - time_pad_y_cm,
    )
    ax = add_axes(fig, l, b, w, h)

    # Select data for this agent
    agent_df = median_df.query(agent.query)

    # Plot mean and sem over individuals
    group = agent_df.groupby('time')[prop_class.prop_name]
    mean = group.mean()
    sem = group.sem()
    std = group.std()
    ax.plot(mean.index, mean, color=agent.color, linestyle='-', label=agent.label)
    ax.fill_between(mean.index, mean - sem, mean + sem, color=agent.color, alpha=ALPHA)

    # Plot fit
    mean_ind_meta_popt = (
        fit_df
        .query(agent.query)
        .xs(prop_class.prop_name, level='prop_name')
        [model.par_names].values[0]
    )
    mean_ind_hat = model.eval_cont(b_left, b_right, dt_hat, *mean_ind_meta_popt)
    ax.plot(ts_hat, mean_ind_hat, color=COLOR_MODEL, linestyle='--', label='Fit (ind)')

    # Format
    hide_spines(ax, ['top', 'right'])
    set_ticks(ax,
              x_ticks=time_ticks,
              y_ticks=prop_class.par_ticks[0])
    set_bounds(ax, y=[prop_class.par_ticks[0][0], prop_class.par_ticks[0][-1]])
    set_lims(ax, time_lim, prop_class.par_lims[0])
    set_spine_position(ax, ['left', 'bottom'])
    set_labels(ax, 'Time (s)')  # , f'{prop_class.label}\n({prop_class.unit})')
    set_axlines(ax, axhlines=prop_class.prop_axlines)

    # Add vertical lines for time reference
    set_axlines(ax, axvlines=t_ns[1:-1], vlim=[prop_class.par_ticks[0][0], prop_class.par_ticks[0][-1]], alpha=0.2)

# Hide shared-x axis for larvae
hide_spines(ax, ['bottom'], hide_ticks=True)

# Add stimulus bar in separate axes
offset_stim_y_cm = 9.5  # cm
stim_x_cm = time_x_cm
stim_y_cm = 0.25 # cm


l, b, w, h = (
    offset_x_cm,
    offset_stim_y_cm,
    stim_x_cm,
    stim_y_cm,
)
ax = add_axes(fig, l, b, w, h)
ax.set_xlim(time_lim[0] + 0.1, time_lim[-1])
add_stimulus_bar(ax, t_ns, b_left_ns, b_right_ns, axvline=False, y_pos=0.5, height=1)
set_ticks(ax, x_ticks=[], y_ticks=[])

# #############################################################################
# Fig 3D: fitted parameters
# #############################################################################
offset_x_cm = 0.2
_pad_x_cm = 1.5
jitter = 0.5

# Format
plot_dict = {
    'tau_lpf': {'ticks': np.arange(5), 'ticklabels': np.arange(5),
                'lims': [0, 4], 'bounds': [0, 4], },
    'wD_pos': {'ticks': np.linspace(0, 200, 5),
               'ticklabels': np.linspace(0, 200, 5),
               'lims': [-22, 220], 'bounds': [0, 200], },
    'wD_neg': {'ticks': -1 * np.linspace(0, 200, 5),
               'ticklabels': np.linspace(0, 200, 5),
               'lims': [-220, 22], 'bounds': [-200, 0], },
    'wC': {'ticks': np.linspace(0, 30, 4),
           'ticklabels': np.linspace(0, 30, 4),
           'lims': [-2, 30], 'bounds': [0, 30], },
}

axs = []
for j in range(2):
    for i in range(2):
        l, b, w, h = (
            offset_x_cm + _pad_x_cm + i * ax_x_cm,
            pad_y_cm + j * ax_y_cm,
            ax_x_cm - _pad_x_cm,
            ax_y_cm - pad_y_cm,
        )
        ax = add_axes(fig, l, b, w, h)
        axs.append(ax)

# We need to reorder the parameters to match the order of the axes
par_names = ['wD_pos', 'wD_neg', 'tau_lpf', 'wC', ]
for num, (ax, meta_par_name) in enumerate(zip(axs, par_names)):
    ind_meta_popt = ind_fit_df.xs(prop_class.prop_name, level='prop_name')
    mean_ind_meta_popt = fit_df.xs(prop_class.prop_name, level='prop_name')

    sns.stripplot(  # Individual fits
        data=ind_meta_popt.reset_index(), x='fish_age', y=meta_par_name,
        hue='fish_age', palette=AGE_PALETTE_DICT, alpha=ALPHA, size=MARKER_SIZE,
        marker='o',
        dodge=True, jitter=jitter,
        label='Bootstrapped', legend=False,
        ax=ax
    )
    sns.stripplot(  # Mean over fits to individual data
        data=mean_ind_meta_popt.reset_index(),
        x='fish_age', y=meta_par_name,
        hue='fish_age', palette=ListedColormap([COLOR_ANNOT]), size=MARKER_SIZE_LARGE, marker='X',
        dodge=True,
        label='Mean', legend=False,
        ax=ax,
    )

    # Add statistics
    _stat_str, stat_dict = get_stats_two_groups(
        agents[0].name, agents[1].name,
        ind_meta_popt.query(agents[0].query)[meta_par_name],
        ind_meta_popt.query(agents[1].query)[meta_par_name],
    )
    add_stats(ax, 0, 1, ANNOT_Y, p_value_to_stars(stat_dict['Mann-Whitney U test']['p']))

    # # Get ticks
    ticks = plot_dict[meta_par_name]['ticks']
    ticklabels = plot_dict[meta_par_name]['ticklabels']
    lims = plot_dict[meta_par_name]['lims']
    bounds = plot_dict[meta_par_name]['bounds']

    set_ticks(ax, x_ticks=[], y_ticks=ticks)
    set_labels(ax, x='', y=model.par_labels_dict[meta_par_name])
    set_bounds(ax, y=bounds)
    set_lims(ax, y=lims)
    set_spine_position(ax, )
    hide_spines(ax, ['top', 'right', 'bottom'])

    # # Inverse y-ticks
    if meta_par_name == 'wD_neg':
        ax.invert_yaxis()

# #############################################################################
# Fig 3E: Integrator
# #############################################################################
# Stimulus settings
from settings.stim_brightness_integrator import *


# Load data
event_df = load_event_df(path_to_main_data_folder, experiment_name, agents)
median_df = load_median_df(path_to_main_data_folder, experiment_name, agents)
n_fish_dict = get_n_fish(median_df, agents)

# Detect peak positions #######################################################
peak_df = get_peaks(median_df, fit_df, agents, model, t_intervals)

# fig = create_figure(fig_width_cm + pad_x_cm, fig_width_cm)
_pad_x_cm, _pad_y_cm = 0, 1  # cm
offset_x_cm = 2.6 * ax_x_cm
offset_y_cm = 1.5
_ax_x_cm = 1.25 * ax_x_cm

# Plot peaks
axs = []
for k, agent in enumerate(agents[::-1]):
    for j, stim_query in enumerate(['_LbrightRbright_LdarkRbright', '_LdarkRdark_LdarkRbright']):
        i = 3 + k

        l, b, w, h = (
            offset_x_cm + j * (_ax_x_cm + _pad_x_cm),
            offset_y_cm + k * (ax_y_cm - _pad_y_cm),
            _ax_x_cm,
            ax_y_cm - _pad_y_cm,
        )
        ax = add_axes(fig, l, b, w, h)
        axs.append(ax)

        plot_df = (
            peak_df
            .query(agent.query)
            .xs('data', level='type')
            .xs(stim_query, level='stim_query')
        )
        ax.errorbar(plot_df['t_interval'], plot_df['peak_mean'], plot_df['peak_std'], color=agent.color, fmt='o')

        plot_df = (
            peak_df
            .query(agent.query)
            .xs('fit', level='type')
            .xs(stim_query, level='stim_query')
        )
        ax.plot(plot_df['t_interval'], plot_df['peak_mean'], color=COLOR_MODEL, linestyle='--', zorder=-100)

axes = np.reshape(axs, (2, -1))

# Format all axes
hide_spines(axes)
set_axlines(axes, axhlines=50, hlim=[0, 10])
# set_labels(axes[:, 0], y=f'{prop_class.label}\n({prop_class.unit})')
set_labels(axes[0, :], x='Interval (s)')
set_lims(axes, x=[-1, 11], y=[15, 90])
set_bounds(axes, x=[0, 10], y=[20, 80])
set_ticks(axes, x_ticks=[0, 5, 10], y_ticks=[20, 50, 80])

hide_spines(axes[:, 1], spines=['left'], hide_ticks=True)
hide_spines(axes[1, :], spines=['bottom'], hide_ticks=True)

savefig(fig, path_to_fig_folder.joinpath('fig3.pdf'))

