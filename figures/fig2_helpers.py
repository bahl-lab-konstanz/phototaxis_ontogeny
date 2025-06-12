
# Third party library imports
from scipy.optimize import curve_fit
from scipy.io import loadmat

# Local library imports
from utils.general_utils import log_model, my_double_linear, get_stats_two_groups
from utils.plot_utils import *
from settings.general_settings import rng
from settings.plot_settings import *
from settings.prop_settings import *
from utils.models import FullModel


# #############################################################################
# Settings
# #############################################################################
# Stimulus settings
label = 'Brightness (lux)'
col_name = 'brightness'
bin_name = 'brightness_bin'
model_names = ['log', 'linear']
brightness_bins = np.arange(0, 301, 50)
brightness_bin_centers = (brightness_bins[:-1] + brightness_bins[1:]) / 2
brightness_bin_ticks = [10, 150, 300]
brightness_bin_tick_labels = brightness_bin_ticks

# Properties
prop_classes = [
    PercentageTurns(),
    TurnAngle(),
    TotalDuration(),
    Distance(),
    PercentageLeft(),
]

# #############################################################################
# Functions
# #############################################################################
# Misc ########################################################################
def map_azimuth_brightness(df, c_min=10, c_max=300):
    df['azimuth_rad'] = np.arctan2(df['y_position'], df['x_position'])  # -pi to pi rad
    return (c_max - c_min) * (np.pi - np.abs(df['azimuth_rad'])) / np.pi + c_min  # c_min to c_max lux


# Fit models ##################################################################
def fit_model(
        median_df, agents, prop_classes,
        col_name, bin_name, model_names,
):
    def _fit_models(x, y, model_names):
        res_list, column_list = [], []

        for model_name in ['log', 'linear', 'double_linear']:
            if model_name in model_names and model_name == 'log':
                # Fit logarithmic model: a log(x) + b
                res = curve_fit(log_model, x, y, p0=[1, 1])
                a_log, b_log = res[0]
                error_log = np.mean((y - log_model(x, a_log, b_log)) ** 2)  # MSE
                res_list.extend([a_log, b_log, error_log])
                column_list.extend(['a_log', 'b_log', 'error_log'])
            if model_name in model_names and model_name == 'linear':
                # Fit linear model: p0 + p1 * x
                p = np.polynomial.Polynomial.fit(x, y, deg=1)
                p0, p1 = p.convert().coef
                error_lin = np.mean((y - p(x)) ** 2)  # MSE
                res_list.extend([p0, p1, error_lin])
                column_list.extend(['p0', 'p1', 'error_lin'])
            if model_name in model_names and model_name == 'double_linear':
                # Fit double-linear model: a_pos * x_+ + a_neg * x_- + b
                res = curve_fit(my_double_linear, x, y, p0=[1, 1, 1])
                a_pos, a_neg, b = res[0]
                error_double_lin = np.mean((y - my_double_linear(x, *res[0])) ** 2)  # MSE
                res_list.extend([a_pos, a_neg, b, error_double_lin])
                column_list.extend(['a_pos', 'a_neg', 'b', 'error_double_linear'])
        return res_list, column_list

    index = [
        'prop_name', 'dist_name', 'bin_name',
        'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
        'fish_age', 'fish_genotype',  # required for agent query
        'par_name'
    ]

    mean_meta_fit_results = []
    ind_meta_fit_results = []
    for prop_class in prop_classes:
        for agent in agents:
            agent_df = median_df.query(agent.query)
            fish_age = agent_df.index.unique('fish_age')[0]
            fish_genotype = agent_df.index.unique('fish_genotype')[0]

            # Mean over individuals ###########################################
            # Retrieve values
            group = agent_df.groupby(bin_name, observed=True)
            mean = group[prop_class.prop_name].mean()
            sem = group[prop_class.prop_name].sem()
            std = group[prop_class.prop_name].std()
            x = mean.index.to_numpy()
            experiment_ID, fish_or_agent_name, experiment_repeat = 0, 0, 0

            # Fit models
            res, columns = _fit_models(x, mean.values, model_names)

            # Append results
            mean_meta_fit_results.append([
                prop_class.prop_name, prop_class.dist_name, bin_name,
                experiment_ID, fish_or_agent_name, experiment_repeat,
                fish_age, fish_genotype,
                'median',
                *res
            ])

            # Individual fits #################################################
            for idx, exp_df in agent_df.groupby(['experiment_ID', 'fish_or_agent_name', 'experiment_repeat']):
                # Retrieve values, drop NaN values
                exp_df = exp_df.dropna(subset=[prop_class.prop_name])
                x = exp_df[col_name]
                y = exp_df[prop_class.prop_name].values
                experiment_ID, fish_or_agent_name, experiment_repeat = idx

                # Check if we have enough input values to fit
                if len(x) < 3:
                    continue

                # Fit models
                res, columns = _fit_models(x, y, model_names)

                # Append results
                ind_meta_fit_results.append([
                    prop_class.prop_name, prop_class.dist_name, bin_name,
                    experiment_ID, fish_or_agent_name, experiment_repeat,
                    fish_age, fish_genotype,
                    'median',
                    *res
                ])

    # Create dataframe of fit results #########################################
    # Fit to mean over fish
    mean_meta_fit_df = pd.DataFrame(mean_meta_fit_results, columns=index + columns)
    mean_meta_fit_df.set_index(index, inplace=True)
    mean_meta_fit_df.sort_index(inplace=True)

    # Fit to individual fish
    ind_meta_fit_df = pd.DataFrame(ind_meta_fit_results, columns=index + columns)
    ind_meta_fit_df.set_index(index, inplace=True)
    ind_meta_fit_df.sort_index(inplace=True)

    # Mean over fits to individual fish
    mean_ind_meta_fit_df = (
        ind_meta_fit_df
        .groupby([
            'prop_name', 'dist_name', 'bin_name',
            'fish_age', 'fish_genotype',
            'par_name',
        ])
        .mean()
    )
    # # Update index to match mean_meta_fit_df
    mean_ind_meta_fit_df['experiment_ID'] = 0
    mean_ind_meta_fit_df.set_index('experiment_ID', append=True, inplace=True)

    return ind_meta_fit_df, mean_ind_meta_fit_df, mean_meta_fit_df


# Plot functions ##############################################################
def plot_median(
        median_df, mean_ind_meta_fit_df,
        agents, prop_classes,
        bin_name, model_name,
        label, ticks, tick_labels,
        ax_x_cm=3, ax_y_cm=4,     # cm
        alpha=0.1,  # For individual lines
        direction='horizontal',  # 'horizontal' or 'vertical'
):
    if direction == 'horizontal':
        fig_width, fig_height = 16.2, 6     # cm
    else:
        fig_width, fig_height = 7, 21     # cm

    # Create figure
    fig = create_figure(fig_width=fig_width, fig_height=fig_height)  # cm
    for prop_num, prop_class in enumerate(prop_classes):
        if direction == 'horizontal':
            i = prop_num
            j = 0
        else:
            i = 0
            j = len(prop_classes) - prop_num - 1

        # Add axes
        l, b, w, h = (
            pad_x_cm + i * ax_x_cm,
            pad_y_cm + j * ax_y_cm,
            ax_x_cm - pad_x_cm,
            ax_y_cm - pad_y_cm
        )
        ax = add_axes(fig, l, b, w, h)

        for agent in agents:
            agent_df = median_df.query(agent.query)

            # Individual fish #################################################
            sns.lineplot(
                data=agent_df.reset_index(), x=bin_name, y=prop_class.prop_name,
                hue='experiment_ID', palette=ListedColormap([agent.color]), alpha=alpha,
                errorbar=None,
                zorder=-100,
                ax=ax, legend=False,
            )
            # For legend
            ax.plot([], [], color=agent.color, label=agent.label_single, linestyle='-', alpha=alpha)

            # Compute mean and sem over individuals ###########################
            group = agent_df.groupby(bin_name, observed=True)
            mean = group[prop_class.prop_name].mean()
            sem = group[prop_class.prop_name].sem()
            std = group[prop_class.prop_name].std()
            ax.errorbar(
                mean.index, mean,
                yerr=sem,
                color=agent.color, markerfacecolor=agent.markerfacecolor, markeredgecolor=agent.color,
                fmt='o', linestyle='none',
                label=f'{agent.label} (mean + SEM)', zorder=+100,
            )

            # Plot fits (to mean data) ########################################
            # Retrieve x values
            if bin_name == 'brightness_bin':
                x_fit = np.linspace(10, 300)    # lux
            elif bin_name == 'contrast_bin':
                x_fit = np.linspace(0, 1)       # Michelson-contrast
            elif bin_name == 'temporal_bin':
                x_fit = np.linspace(-300, 300)       # lux/s

            # Retrieve model parameters and plot
            if isinstance(mean_ind_meta_fit_df, type(None)) or isinstance(model_name, type(None)):
                pass
            elif model_name == 'log':
                # Plot logarithmic model
                a_log, b_log = (
                    mean_ind_meta_fit_df
                    .xs(prop_class.prop_name, level='prop_name')
                    .xs(bin_name, level='bin_name')
                    .xs('median', level='par_name')
                    .query(agent.query)
                    [['a_log', 'b_log']].values[0]
                )
                y_fit = log_model(x_fit, a_log, b_log)
                ax.plot(x_fit, y_fit,
                        color=COLOR_MODEL, linestyle='--', label=r'$a + b\ \ln\left(x\right)$',
                        zorder=0,
                        )
            elif model_name == 'linear':
                # Plot linear model
                p0, p1 = (
                    mean_ind_meta_fit_df
                    .xs(prop_class.prop_name, level='prop_name')
                    .xs(bin_name, level='bin_name')
                    .xs('median', level='par_name')
                    .query(agent.query)
                    [['p0', 'p1']].values[0]
                )
                y_fit = p0 + p1 * x_fit
                ax.plot(x_fit, y_fit, color=COLOR_MODEL, linestyle='--', label=r'a + b\ x$', zorder=0)
            elif model_name == 'double_linear':
                # Plot double linear model
                a_pos, a_neg, b = (
                    mean_ind_meta_fit_df
                    .xs(prop_class.prop_name, level='prop_name')
                    .xs(bin_name, level='bin_name')
                    .xs('median', level='par_name')
                    .query(agent.query)
                    [['a_pos', 'a_neg', 'b']].values[0]
                )
                y_fit = my_double_linear(x_fit, a_pos, a_neg, b)
                ax.plot(x_fit, y_fit, color=COLOR_MODEL, linestyle='--',
                        label=r'$a_\mathrm{pos}\ x_{+} + a_\mathrm{neg}\ x_{-} + b$', zorder=0)
            elif model_name == 'full_fig2and3':
                # Evaluate model
                model = FullModel()

                # Params
                params = (
                    mean_ind_meta_fit_df
                    .xs(prop_class.prop_name, level='prop_name')
                    .xs('median', level='par_name')
                    .query(agent.query)
                    [model.par_names].values[0]
                )

                # This dataset contains only average brightness values, so we ignore the
                # other pathways.
                params1 = []
                for param, par_name in zip(params, model.par_names):
                    if par_name == 'a' or par_name == 'bA':
                        params1.append(param)
                    else:
                        params1.append(0)

                y_fit = model.eval_cont(
                    x_fit, x_fit, 1,  # x1, x2, dt
                    *params1
                )
                ax.plot(x_fit, y_fit,color=COLOR_MODEL, linestyle='--',
                        label=model.label, zorder=0)

        # Format
        hide_spines(ax, ['top', 'right'])
        set_spine_position(ax, spines=['left', 'bottom'])
        # Format x-axis
        ax.set_xlabel(label)
        set_ticks(ax, x_ticks=ticks, x_ticklabels=tick_labels)
        set_bounds(ax, x=(ticks[0], ticks[-1]))
        # Format y-axis
        ax.set_ylabel(f'{prop_class.label} ({prop_class.unit})')
        set_lims_and_bounds(ax, y=prop_class.par_lims[0])
        set_ticks(ax, y_ticks=prop_class.par_ticks[0], y_ticklabels=prop_class.par_ticklabels[0])
        set_axlines(ax, y=prop_class.par_axlines[0])

    if direction == 'horizontal':
        # Add legend above last plot
        set_legend(ax, loc='lower right', bbox_to_anchor=(1.1, 1.1))
    else:
        # Add legend next to last plot
        set_legend(ax, loc='upper left', bbox_to_anchor=(1.1, 1.0))

    return fig


def plot_fitted_pars(
        ind_meta_fit_df, mean_meta_fit_df,
        agents, prop_classes,
        bin_name, model_name,
        label,
        ax_x_cm=3, ax_y_cm=4,     # cm
        jitter=0.3,
):
    stat_str = ''
    fig_list = []

    # Retrieve model settings
    if model_name == 'log':
        meta_par_names = ['a_log', 'b_log']
        label_adds = ['Offset (a)\n', 'Logarithmic slope (b)\n']
        unit_add = ''
    elif model_name == 'linear':
        meta_par_names = ['p1', 'p0']
        label_adds = ['Slope (a)\n', 'Offset (b)\n']
        unit_add = f'/{label}'
    elif model_name == 'double_linear':
        meta_par_names = ['a_pos', 'a_neg', 'b']
        label_adds = [r'Slope (positive $W_{AD}^+$)' + '\n', r'Slope (negative $W_{AD}^-$)' + '\n', 'Offset (a)\n']
        unit_add = f'/{label}'
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Create figure for each fitted parameter
    for k, (meta_par_name, label_add) in enumerate(zip(meta_par_names, label_adds)):
        fig = create_figure(5*(pad_x_cm + pad_y_cm), 3*(pad_x_cm + pad_y_cm))
        for i, prop_class in enumerate(prop_classes):
            ind_prop_fit_df = (
                ind_meta_fit_df
                .xs(prop_class.prop_name, level='prop_name')
                .xs(bin_name, level='bin_name')
                .xs('median', level='par_name')
            )
            mean_prop_fit_df = (
                mean_meta_fit_df
                .xs(prop_class.prop_name, level='prop_name')
                .xs(bin_name, level='bin_name')
                .xs('median', level='par_name')
            )

            # Get data lim and ticks
            data_max = ind_prop_fit_df[meta_par_name].abs().max()
            data_lim = prop_class.par_lim_dict.get(meta_par_name, [-0, 0])
            if -data_max < data_lim[0] or data_max > data_lim[1]:
                print(f'\tplot_fitted_pars(): {prop_class.prop_name} | {meta_par_name} | data_max={data_max:.4f} but data_lim={data_lim}')
                data_lim = [-data_max, data_max]
            data_bins = np.linspace(data_lim[0], data_lim[1], 10)  # Even number to include 0 bin

            # data_ticks = prop_class.par_tick_dict.get(meta_par_name, None)  # TODO: Implement
            data_ticks = np.round(np.linspace(data_lim[0], data_lim[1], 5), 3)
            if 'lux/s' in label:
                data_ticklabels = data_ticks
                if i != 0:
                    label_add = ''  # Only add label to first column
            elif 'mm' in prop_class.unit:
                # Convert cm to mm
                data_ticklabels = np.round(np.linspace(10 * data_lim[0], 10 * data_lim[1], 5), 1)
                data_ticklabels = [f'{tick: .0f}' for tick in data_ticklabels]
            elif '%' in prop_class.unit or 'deg' in prop_class.unit:
                # Set tick labels as integer
                data_ticklabels = [f'{tick: .0f}' for tick in data_ticks]
            else:
                data_ticklabels = [f'{tick: .1f}' for tick in data_ticks]
            fish_ticks = np.arange(0, 30 + 1, 10).astype(int)

            # Statistics ##########################################################
            stat_str += f"\n{prop_class.prop_name} | {bin_name} | {model_name} | {meta_par_name}\n"
            _stat_str, stat_dict = get_stats_two_groups(
                agents[0].name, agents[1].name,
                ind_prop_fit_df.query(agents[0].query)[meta_par_name],
                ind_prop_fit_df.query(agents[1].query)[meta_par_name],
            )
            stat_str += _stat_str

            # Stripplot ###########################################################
            j = 1
            l, b, w, h = (
                pad_x_cm + i * ax_x_cm,
                pad_y_cm + j * ax_y_cm,
                ax_x_cm - pad_x_cm,
                ax_y_cm - pad_y_cm
            )
            ax = add_axes(fig, l, b, w, h)
            sns.stripplot(  # Individual fits
                data=ind_prop_fit_df.reset_index(), x='fish_age', y=meta_par_name,
                hue='fish_age', palette=AGE_PALETTE_DICT, alpha=ALPHA, size=MARKER_SIZE,
                marker=MARKER_HOLLOW, dodge=False, jitter=jitter,
                ax=ax, legend=False,
            )
            sns.stripplot(  # Add mean fit
                data=mean_prop_fit_df.reset_index(), x='fish_age', y=meta_par_name,
                hue='fish_age', palette=ListedColormap([COLOR_MODEL]), size=MARKER_SIZE_LARGE, marker='X',
                ax=ax, dodge=False, legend=False,
            )

            # Add statistics
            # ax.text(0.5, 1.05, p_value_to_stars(stat_dict['Bootstrapping']['p']), ha='center', va='bottom', transform=ax.transAxes)
            add_stats(ax, 0, 0, ANNOT_Y, p_value_to_stars(stat_dict['Bootstrapping']['p0']))
            add_stats(ax, 1, 1, ANNOT_Y, p_value_to_stars(stat_dict['Bootstrapping']['p1']))

            # Format
            set_bounds(ax, y=data_lim)
            set_ticks(ax, x_ticks=[], y_ticks=data_ticks, y_ticklabels=data_ticklabels)
            set_labels(ax, x='', y=f'{label_add}{prop_class.label} ({prop_class.unit}{unit_add})')
            set_axlines(ax, axhlines=0)
            set_spine_position(ax, )
            hide_spines(ax, ['top', 'right', 'bottom'])

            # Histplot ############################################################
            j = 0
            l, b, w, h = (
                pad_x_cm + i * ax_x_cm,
                pad_y_cm + j * ax_y_cm,
                ax_x_cm - pad_x_cm,
                ax_y_cm - pad_y_cm
            )
            ax = add_axes(fig, l, b, w, h)
            sns.histplot(  # Individual fits
                data=ind_prop_fit_df.reset_index(), y=meta_par_name, bins=data_bins,
                hue='fish_age', palette=AGE_PALETTE_DICT, linestyle='none',
                ax=ax, legend=False,
            )

            # Format
            set_ticks(ax, x_ticks=fish_ticks, y_ticks=data_ticks, y_ticklabels=data_ticklabels)
            set_labels(ax, y=f'{label_add}{prop_class.label} ({prop_class.unit}{unit_add})', x='Nr. fish')
            set_bounds(ax, y=data_lim)
            set_axlines(ax, axhlines=0)
            # set_spine_position(ax, ['left'])
            hide_spines(ax, ['top', 'right'])
        fig_list.append(fig)
    return fig_list, meta_par_names, stat_str


def plot_nr_events(
        event_df,
        col_name, bins,
        label, ticks, tick_labels,
):
    ax_x_cm, ax_y_cm = 4, 4  # cm

    fig = create_figure(ax_x_cm + pad_x_cm, ax_y_cm + pad_y_cm)
    # Add axes
    i, j = 0, 0
    l, b, w, h = (
        pad_x_cm + i * ax_x_cm,
        pad_y_cm + j * ax_y_cm,
        ax_x_cm - pad_x_cm,
        ax_y_cm - pad_y_cm
    )
    ax = add_axes(fig, l, b, w, h)

    sns.histplot(
        data=event_df.reset_index(), x=col_name, bins=bins,
        hue='fish_age', palette=AGE_PALETTE_DICT, linestyle='none',
        ax=ax, legend=False,
    )

    # Format
    hide_spines(ax)
    set_spine_position(ax, ['left'])
    set_labels(ax, x=label, y='Nr. swims')
    set_ticks(ax, x_ticks=ticks, x_ticklabels=tick_labels)
    set_bounds(ax, x=(ticks[0], ticks[-1]))
    return fig


# #############################################################################
# Turn sequences (Dunn et al. 2016)
# #############################################################################
def load_dunn_df(path_to_mat_file):
    mat = loadmat(path_to_mat_file)
    concat_turns = mat['concat_turns']
    # Loop over all 19 fish in the dataset
    dunn_df_list = []
    for fish_ind in range(19):
        # Loop over all valid turn sequences for this fish
        for sequence_ind in range(len(concat_turns[0][fish_ind][0])):
            seq = concat_turns[0][fish_ind][0][sequence_ind]

            # Store sequence data in a pandas dataframe
            _df = pd.DataFrame(
                data={
                    'estimated_orientation_change': seq[0],
                    'direction': np.sign(seq[0]),
                },
            )
            _df['experiment_ID'] = fish_ind
            _df['trial'] = 0
            _df['sequence'] = sequence_ind
            # Place holder to allow our agent.query to work
            _df['fish_or_agent_name'] = 'dunn2016'
            _df['fish_genotype'] = 'wt-kn'
            _df['fish_age'] = 5
            _df['stimulus_name'] = 'dunn2016'
            _df['experiment_repeat'] = 0
            _df['arena_index'] = 0
            _df['setup_index'] = 0
            _df['folder_name'] = 'dunn2016'
            dunn_df_list.append(_df)
    # Concatenate all sequences
    dunn_df = pd.concat(dunn_df_list)

    dunn_df.set_index([
        'stimulus_name', 'trial', 'fish_genotype', 'fish_age', 'experiment_ID', 'fish_or_agent_name',
        'experiment_repeat', 'arena_index', 'setup_index', 'folder_name',
    ], inplace=True)

    return dunn_df


def get_turn_direction(turn_angle, angle_threshold):
    if turn_angle > angle_threshold:
        return +1  # left
    elif turn_angle < -1 * angle_threshold:
        return -1  # right
    else:
        return 0  # straight


def get_turn_label(turn_angle, angle_threshold):
    if turn_angle > angle_threshold:
        return 'L'
    elif turn_angle < -1 * angle_threshold:
        return 'R'
    else:
        return 'S'


def count_turn_pairs(
        df, angle_threshold,
):
    order = ['LR', 'RL', 'LL', 'RR', 'SL', 'SR', 'LS', 'RS', 'SS']

    # Binarise swims in left/right/straight: L, R, S
    df = df.copy()
    df['turn_label'] = df['estimated_orientation_change'].apply(
        lambda x: get_turn_label(x, angle_threshold)
    )
    
    # Shift data to get previous swim
    df['prev_turn_label'] = (
        df
        .groupby([
            'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
            'arena_index', 'setup_index', 'folder_name',
            'fish_genotype', 'fish_age', 'stimulus_name', 'trial',
        ])
        ['turn_label']
        .shift(1)  # Shift to get previous swim
    )

    # Add column with shuffled previous swims
    df['shuffled_turn_label'] = (
        df
        .groupby([
            'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
            'arena_index', 'setup_index', 'folder_name',
            'fish_genotype', 'fish_age', 'stimulus_name', 'trial',
        ])
        ['turn_label']
        # Use transform to avoid index issues
        .transform(lambda x: x.sample(frac=1, replace=False).values)
    )

    # Add strings
    df['turn_pair'] = df['prev_turn_label'] + df['turn_label']
    df['shuffled_turn_pair'] = df['shuffled_turn_label'] + df['turn_label']

    # Count turn pairs
    def count_turn_pairs(df, colname, kind):
        return (
            df
            .groupby([
                'experiment_ID', 'fish_or_agent_name', 'experiment_repeat',
                'arena_index', 'setup_index', 'folder_name',
                'fish_genotype', 'fish_age',
            ])[colname]
            .value_counts()
            .rename('count')
            .reset_index()
            .rename(columns={colname: 'turn_pair'})
            .assign(kind=kind)
        )

    counted_pair_df = pd.concat([
        count_turn_pairs(df, 'turn_pair', 'real'),
        count_turn_pairs(df, 'shuffled_turn_pair', 'shuffled')
    ])

    # Determine axis width on total number of turn pairs
    turn_pairs = counted_pair_df['turn_pair'].unique()
    if turn_pairs.size > 4:
        _ax_x_cm = 2 * ax_x_cm
        _order = order
    else:
        _ax_x_cm = ax_x_cm
        _order = [value for value in order if value in turn_pairs]

    return counted_pair_df, _order


def cumulative_turn_direction(df, totsize, angle_threshold, shuffle=False):
    groupby_labels = [
        'experiment_ID', 'fish_or_agent_name',
        'fish_genotype', 'fish_age',
        'stimulus_name', 'trial', 'experiment_repeat',
    ]

    # Convert estimated_turn_direction to +1 and -1
    df = df.copy()
    df['direction'] = df['estimated_orientation_change'].apply(
        lambda x: get_turn_direction(x, angle_threshold)
    )
    # Group individual fish
    grouped = df.groupby(groupby_labels)
    n_fish = grouped.ngroups
    fish_mean = np.zeros(shape=(n_fish, totsize)) * np.nan

    # Loop over fish
    for fish_num, (idx, fish_group) in enumerate(grouped):
        _cumulative_list = []

        if shuffle:
            # Shuffle data within fish
            shuffled_group = fish_group.copy()
            # shuffled_group['estimated_orientation_change'] = fish_group['estimated_orientation_change'].sample(frac=1).values
            # # Compute direction again
            # shuffled_group['direction'] = shuffled_group['estimated_orientation_change'].apply(get_turn_direction)
            shuffled_group['direction'] = rng.integers(-1, 2, size=len(shuffled_group))
            fish_group = shuffled_group

        # Loop over trials and turn sequences
        #   sequences are be separated because fish leave and enter the valid
        #   consideration radius (1 cm from dish border) during an experiment
        for (trial_ID, sequence_ID), group in fish_group.groupby(['trial', 'sequence']):
            # Compute the cumulative direction of the sequence
            # First we detect switches, and only include switches above a certain threshold
            switch_indices = np.where(
                (group['direction'].diff() != 0) &  # detect switches
                (group['estimated_orientation_change'].abs() > angle_threshold)  # only include switches above a certain threshold
            )[0]

            totcumulative = np.zeros(shape=(len(switch_indices), totsize)) * np.nan
            turn_amp = group['direction'].values

            # Then we loop over switches
            for switch_num, switch_ind in enumerate(switch_indices[1:]):  # Start with full first set (following Dunn 2016)
                if group['direction'].iloc[switch_ind] > 0:
                    _values = np.cumsum(
                        turn_amp[switch_ind:switch_ind + totsize]
                    ) - turn_amp[switch_ind]
                else:
                    _values = -1 * np.cumsum(
                        turn_amp[switch_ind:switch_ind + totsize]
                    ) + turn_amp[switch_ind]
                totcumulative[switch_num, :len(_values)] = _values

            # If there is no switch event in this sequence, the fish is in the
            # middle of a sequence. Incorporate these sequences.
            if len(switch_indices) == 1 and len(turn_amp) > 0:
                n_values = np.min([len(turn_amp), totsize])
                totcumulative[0, :n_values] = np.arange(n_values)
                _values = np.arange(totsize)

            # Incorporate the chain that exists already *before* the first detected switch
            if len(switch_indices) > 1:
                n_swims = np.min([switch_indices[1], totsize])  # Limit to totsize
                _values = np.arange(n_swims)
                totcumulative[-1, :len(_values)] = _values

            _cumulative_list.extend(totcumulative)

        # Compute mean within fish
        if len(_cumulative_list):
            # Sometimes a fish has an empty row
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                fish_mean[fish_num, :] = np.nanmean(np.asarray(_cumulative_list), axis=0)

    return fish_mean, n_fish


def get_streak_length_distributions(df, angle_threshold, do_shuffle=False, streak_bins=np.arange(0, 15, 1)):
    groupby_labels = [
        'experiment_ID', 'fish_or_agent_name',
        'fish_genotype', 'fish_age',
        'stimulus_name', 'trial', 'experiment_repeat',
    ]

    streak_bincenters = (streak_bins[1:] + streak_bins[:-1]) / 2

    # Convert estimated_turn_direction to +1 and -1
    df = df.copy()
    df['direction'] = df['estimated_orientation_change'].apply(
        lambda x: get_turn_direction(x, angle_threshold)
    )
    # Group individual fish
    grouped = df.groupby(groupby_labels)
    n_fish = grouped.ngroups

    streak_hist_df_list = []
    # Loop over fish and stimuli
    for fish_num, (idx, fish_group) in enumerate(grouped):
        streaklengths = []

        # Shuffle data within fish
        if do_shuffle:
            # Shuffle data within fish
            shuffled_group = fish_group.copy()
            shuffled_group['direction'] = rng.integers(-1, 2, size=len(shuffled_group))
            fish_group = shuffled_group

        # Loop over trials and turn sequences
        #   sequences are separated because fish leave and enter the valid
        #   consideration radius (1 cm from dish border) during an experiment
        for (trial_ID, sequence_ID), group in fish_group.groupby(['trial', 'sequence']):
            # Loop over swims in this sequence and count streaklengths
            prev_direction = None
            streaklength = 0
            for swim_num, swim in group.iterrows():
                if prev_direction is None:
                    streaklength = 1
                elif swim['direction'] == prev_direction:
                    streaklength += 1
                else:
                    streaklengths.append(streaklength)
                    streaklength = 0
                prev_direction = swim['direction']

        # Compute streak length distribution
        hist, _ = np.histogram(streaklengths, bins=streak_bins, density=True)
        # Store histogram in dataframe
        _df = pd.DataFrame({
            'streak_hist': hist,
            'streak_hist_cumsum': np.cumsum(hist),
            'bins': streak_bincenters
        }, )
        _df['stimulus_name'] = idx[4]
        _df['experiment_ID'] = idx[0]
        _df['fish_genotype'] = idx[2]
        _df['fish_age'] = idx[4]
        # Store dataframe in list
        streak_hist_df_list.append(_df)
    return pd.concat(streak_hist_df_list), streak_bincenters


