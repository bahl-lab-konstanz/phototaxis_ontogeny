
# Third party library imports
from scipy.optimize import curve_fit

# Local library imports
from utils.general_utils import get_b_values, get_stats_two_groups, compute_AIC, compute_BIC
from utils.plot_utils import *
from settings.plot_settings import *
from settings.prop_settings import *


# #############################################################################
# Functions
# #############################################################################
# Fit models ##################################################################
def fit_spatial_temporal_model(
    median_df, agents, prop_classes,
    t_ns, b_left_ns, b_right_ns,
    path_to_fig_folder=None,
    dt_hat=1  # s, bin size of data
):
    # Ensure agents and prop_classes are iterable
    if not isinstance(agents, list):
        agents = [agents]
    if not isinstance(prop_classes, list):
        prop_classes = [prop_classes]

    # Ensure data is same length as t_ns
    # min_t, max_t = min(t_ns), max(t_ns)
    # median_df = median_df.query(f'{min_t} <= time < {max_t}')
    max_t = max(t_ns)
    median_df = median_df.query(f'time < {max_t}')

    ind_meta_fit_results = []
    mean_meta_fit_results = []
    meta_fit_index = [
        'prop_name', 'dist_name', 'bin_name',
        'fish_age', 'fish_genotype',
        'experiment_ID', 'par_name',
    ]

    for j, agent in enumerate(agents):
        # Select data for this agent
        agent_df = median_df.query(agent.query)

        for prop_class in prop_classes:
            print(f"\tFitting {prop_class.prop_name} | {agent.name} ", end='')

            # Fit mean over individuals ###########################################
            group = agent_df.groupby('time')[prop_class.prop_name]
            mean = group.mean()
            std = group.std()
            ts = mean.index
            ts_hat = np.arange(t_ns[0] + dt_hat / 2, t_ns[-1] + dt_hat / 2, dt_hat)
            fish_age = agent_df.index.unique('fish_age')[0]
            fish_genotype = agent_df.index.unique('fish_genotype')[0]
            exp_ID = 0

            # Exclude all time stamps where mean is nan
            ts_hat = ts_hat[~np.isnan(mean)]
            mean = mean[~np.isnan(mean)]

            # Set Spatial-temporal model
            model = prop_class.model
            model.set_stimulus(ts_hat, t_ns, b_left_ns, b_right_ns)

            # First optimization
            try:
                res = curve_fit(
                    f=model.fitfunc,
                    xdata=ts_hat, ydata=mean,
                    p0=model.x0,
                    bounds=model.bounds_curve_fit,
                    nan_policy='omit',
                )
                mean_meta_popt = res[0]
            except Exception as e:
                # Print in red
                print(f"\t\033[91mfit_spatial_temporal_model(): curve_fit error {e}\033[0m")
                continue

            # Compute MSE and AIC
            n_par = len(model.par_names)
            model.set_stimulus(ts_hat, t_ns, b_left_ns, b_right_ns)
            b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)  # shift by dt to correct for binning
            y_hat = model.eval_cont(b_left, b_right, dt_hat, *mean_meta_popt)
            _error_dict = {
                'MSE': np.mean((mean.values - y_hat) ** 2),
                'AIC': compute_AIC(mean.values, y_hat, n_par),
                'BIC': compute_BIC(mean.values, y_hat, n_par),
                'n_par': n_par,
            }

            # Store fit results as dictionary
            _index_dict = dict(zip(meta_fit_index, [
                prop_class.prop_name, prop_class.dist_name, prop_class.prop_name,
                fish_age, fish_genotype, exp_ID, 'median',
            ]))
            _popt_dict = dict(zip(model.par_names, mean_meta_popt))
            mean_meta_fit_results.append({**_index_dict, **_popt_dict, **_error_dict})

            # Inspect result
            if isinstance(path_to_fig_folder, Path):
                # Get brightness values
                b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)    # shift by dt to correct for binning
                y_hat = model.eval_cont(b_left, b_right, dt_hat, *mean_meta_popt)
                fig, ax = plt.subplots(1, 1)
                ax.plot(ts, mean, '.-', color=agent.color, label='data')
                ax.plot(ts_hat, y_hat, '--', color=COLOR_MODEL, label='fit')
                hide_spines(ax)
                set_lims(ax, [0, 270], prop_class.par_lims[0])
                set_labels(ax, 'Time (s)', f'{prop_class.label}\n({prop_class.unit})')
                set_axlines(ax, axhlines=prop_class.prop_axlines)
                set_ticks(ax,
                          x_ticks=np.arange(0, 270 + 1, 30), x_ticksize=0,
                          y_ticks=prop_class.par_ticks[0], y_ticksize=5, )
                add_stimulus_bar(ax, t_ns, b_left_ns, b_right_ns)
                savefig(fig, path_to_fig_folder.joinpath(f'fit_spatial_temporal_model', f'{prop_class.prop_name}_mean_{agent.name}.pdf'), close_fig=True)

            # Fit individuals #####################################################
            grouped = agent_df.groupby(['fish_age', 'fish_genotype', 'experiment_ID'])
            for idx, exp_df in grouped:
                fish_age, fish_genotype, exp_ID = idx

                # Drop NaN values
                exp_df.dropna(subset=[prop_class.prop_name], inplace=True)
                exp_df.sort_values('time', inplace=True)
                y = exp_df[prop_class.prop_name].values
                ts = exp_df['time'].values
                ts_hat = ts + dt_hat/2

                # # Get number of non-nan datapoints from y
                # if np.count_nonzero(~np.isnan(y)) <= 100:
                #     print(f"Skipping {agent.name} {exp_ID:03d}: too few data points")
                #     continue

                # Exclude all time stamps where mean is nan
                ts_hat = ts_hat[~np.isnan(y)]
                y = y[~np.isnan(y)]

                # Set Spatial-temporal model
                model = prop_class.model
                model.set_stimulus(ts_hat, t_ns, b_left_ns, b_right_ns)

                # First optimization
                try:
                    res = curve_fit(
                        f=model.fitfunc,
                        xdata=ts_hat, ydata=y,
                        p0=model.x0,
                        bounds=model.bounds_curve_fit,
                    )
                    ind_meta_popt = res[0]
                except Exception as e:
                    print(f"\tfit_spatial_temporal_model(): curve_fit error {agent.name} {exp_ID:03d}: {e}")
                    continue

                # Compute MSE and AIC
                n_par = len(model.par_names)
                b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)  # shift by dt to correct for binning
                y_hat = model.eval_cont(b_left, b_right, dt_hat, *ind_meta_popt)
                _error_dict = {
                    'MSE': np.mean((y - y_hat) ** 2),
                    'AIC': compute_AIC(y, y_hat, n_par),
                    'BIC': compute_BIC(y, y_hat, n_par),
                    'n_par': n_par,
                }

                # Store fit results as dictionary
                _index_dict = dict(zip(meta_fit_index, [
                    prop_class.prop_name, prop_class.dist_name, prop_class.prop_name,
                    fish_age, fish_genotype, exp_ID, 'median',
                ]))
                _popt_dict = dict(zip(model.par_names, ind_meta_popt))
                ind_meta_fit_results.append({**_index_dict, **_popt_dict, **_error_dict})

                # # Plot individual results
                # if isinstance(path_to_fig_folder, Path):
                #     b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)
                #     y_hat = model.eval(b_left, b_right, dt_hat, *ind_meta_popt)
                #     fig, ax = plt.subplots(1, 1)
                #     ax.plot(ts, y, '.', color=agent.color, label='data')
                #     ax.plot(ts_hat, y_hat, '--', color=COLOR_MODEL, label='fit')
                #     hide_spines(ax)
                #     set_lims(ax, [0, 270], prop_class.par_lims[0])
                #     set_labels(ax, 'Time (s)', f'{prop_class.label}\n({prop_class.unit})')
                #     set_axlines(ax, axhlines=prop_class.prop_axlines)
                #     set_ticks(ax,
                #               x_ticks=np.arange(0, 270 + 1, 30), x_ticksize=0,
                #               y_ticks=prop_class.par_ticks[0], y_ticksize=5, )
                #     savefig(
                #         fig,
                #         path_to_fig_folder.joinpath(
                #             f'fit_spatial_temporal_model',
                #             f'{prop_class.prop_name}_{agent.name}',
                #             f'{prop_class.prop_name}_{exp_ID:03d}.pdf'),
                #         close_fig=True,
                #     )

            print(f"\033[92mdone\033[0m")

    # Create dataframe of meta fit results ####################################
    # Fit to individual fish
    ind_meta_fit_df = pd.DataFrame(ind_meta_fit_results).set_index(meta_fit_index).sort_index()
    # Fit to mean over fish
    mean_meta_fit_df = pd.DataFrame(mean_meta_fit_results).set_index(meta_fit_index).sort_index()
    # Mean over fits to individual fish
    groupby_labels = ['prop_name', 'dist_name', 'bin_name', 'fish_age', 'fish_genotype', 'par_name', ]
    mean_ind_meta_fit_df = ind_meta_fit_df.groupby(groupby_labels).mean()
    # # Update index to match mean_meta_fit_df
    mean_ind_meta_fit_df['experiment_ID'] = 0
    mean_ind_meta_fit_df.set_index('experiment_ID', append=True, inplace=True)

    return ind_meta_fit_df, mean_ind_meta_fit_df, mean_meta_fit_df,


def fit_spatial_temporal_model_v2(
        median_df, agents,
        prop_classes, models,
        t_ns, b_left_ns, b_right_ns,
        path_to_fig_folder=None,
        dt_hat=1  # s, bin size of data
):
    # Ensure agents, prop_classes and models are iterable
    if not isinstance(agents, list):
        agents = [agents]
    if not isinstance(prop_classes, list):
        prop_classes = [prop_classes]
    if not isinstance(models, list):
        models = [models]

    # Ensure data is same length as t_ns
    max_t = max(t_ns)
    median_df = median_df.query(f'time < {max_t}')

    ind_meta_fit_results = []
    mean_meta_fit_results = []
    meta_fit_index = [
        'prop_name', 'dist_name', 'bin_name',
        'fish_age', 'fish_genotype',
        'experiment_ID',
        'model_name', 'par_name',
    ]

    for j, agent in enumerate(agents):
        # Select data for this agent
        agent_df = median_df.query(agent.query)

        for prop_class in prop_classes:
            print(f"\tFitting {agent.name} {prop_class.prop_name} | ", end='')
            for model in models:
                print(f"{model.name} | ", end='')

                # Fit mean over individuals ###########################################
                group = agent_df.groupby('time')[prop_class.prop_name]
                mean = group.mean()
                std = group.std()
                ts = mean.index
                ts_hat = np.arange(t_ns[0] + dt_hat / 2, t_ns[-1] + dt_hat / 2, dt_hat)
                fish_age = agent_df.index.unique('fish_age')[0]
                fish_genotype = agent_df.index.unique('fish_genotype')[0]
                exp_ID = 0

                # Exclude all time stamps where mean is nan
                ts_hat = ts_hat[~np.isnan(mean)]
                mean = mean[~np.isnan(mean)]

                # Load stimulus
                model.set_stimulus(ts_hat, t_ns, b_left_ns, b_right_ns)

                # Set bounds (can be property specific)
                model.set_bounds(prop_class.prop_name)

                # Fit model
                try:
                    res = curve_fit(
                        f=model.fitfunc,
                        xdata=ts_hat, ydata=mean,
                        p0=model.x0,
                        bounds=model.bounds_curve_fit,
                        nan_policy='omit',
                    )
                    mean_meta_popt = res[0]
                except Exception as e:
                    # Print in red
                    print(f"\t\033[91mfit_spatial_temporal_model(): curve_fit error {e}\033[0m")
                    continue

                # Compute MSE, AIC, BIC
                n_par = len(model.par_names)
                # model.set_stimulus(ts_hat, t_ns, b_left_ns, b_right_ns)
                b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)  # shift by dt to correct for binning
                y_hat = model.eval_cont(b_left, b_right, dt_hat, *mean_meta_popt)
                _error_dict = {
                    'MSE': np.mean((mean.values - y_hat) ** 2),
                    'RMSE': np.sqrt(np.mean((mean.values - y_hat) ** 2)),
                    'AIC': compute_AIC(mean.values, y_hat, n_par),
                    'BIC': compute_BIC(mean.values, y_hat, n_par),
                    'n_par': n_par,
                }

                # Store fit results as dictionary
                _index_dict = dict(zip(meta_fit_index, [
                    prop_class.prop_name, prop_class.dist_name, prop_class.prop_name,
                    fish_age, fish_genotype, exp_ID, model.name, 'median',
                ]))
                _popt_dict = dict(zip(model.par_names, mean_meta_popt))
                mean_meta_fit_results.append({**_index_dict, **_popt_dict, **_error_dict})

                # Inspect result
                if isinstance(path_to_fig_folder, Path):
                    # Get brightness values
                    b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)    # shift by dt to correct for binning
                    y_hat = model.eval_cont(b_left, b_right, dt_hat, *mean_meta_popt)
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(ts, mean, '.-', color=agent.color, label='data')
                    ax.plot(ts_hat, y_hat, '--', color=COLOR_MODEL, label='fit')
                    hide_spines(ax)
                    set_lims(ax, [0, 270], prop_class.par_lims[0])
                    set_labels(ax, 'Time (s)', f'{prop_class.label}\n({prop_class.unit})')
                    set_axlines(ax, axhlines=prop_class.prop_axlines)
                    set_ticks(ax,
                              x_ticks=np.arange(0, 270 + 1, 30), x_ticksize=0,
                              y_ticks=prop_class.par_ticks[0], y_ticksize=5, )
                    add_stimulus_bar(ax, t_ns, b_left_ns, b_right_ns)
                    savefig(fig, path_to_fig_folder.joinpath(f'fit_spatial_temporal_model', f'{prop_class.prop_name}_mean_{agent.name}.pdf'), close_fig=True)

                # Fit individuals #####################################################
                grouped = agent_df.groupby(['fish_age', 'fish_genotype', 'experiment_ID'])
                for idx, exp_df in grouped:
                    fish_age, fish_genotype, exp_ID = idx

                    # Drop NaN values
                    exp_df.dropna(subset=[prop_class.prop_name], inplace=True)
                    exp_df.sort_values('time', inplace=True)
                    y = exp_df[prop_class.prop_name].values
                    ts = exp_df['time'].values
                    ts_hat = ts + dt_hat/2

                    # # Get number of non-nan datapoints from y
                    # if np.count_nonzero(~np.isnan(y)) <= 100:
                    #     print(f"Skipping {agent.name} {exp_ID:03d}: too few data points")
                    #     continue

                    # Exclude all time stamps where mean is nan
                    ts_hat = ts_hat[~np.isnan(y)]
                    y = y[~np.isnan(y)]

                    # Set Spatial-temporal model
                    model.set_stimulus(ts_hat, t_ns, b_left_ns, b_right_ns)

                    # First optimization
                    try:
                        res = curve_fit(
                            f=model.fitfunc,
                            xdata=ts_hat, ydata=y,
                            p0=model.x0,
                            bounds=model.bounds_curve_fit,
                        )
                        ind_meta_popt = res[0]
                    except Exception as e:
                        print(f"\tfit_spatial_temporal_model(): curve_fit error {agent.name} {exp_ID:03d}: {e}")
                        continue

                    # Compute MSE and AIC
                    n_par = len(model.par_names)
                    b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns,
                                                   b_right_ns)  # shift by dt to correct for binning
                    y_hat = model.eval_cont(b_left, b_right, dt_hat, *ind_meta_popt)
                    _error_dict = {
                        'MSE': np.mean((y - y_hat) ** 2),
                        'RMSE': np.sqrt(np.mean((y - y_hat) ** 2)),
                        'AIC': compute_AIC(y, y_hat, n_par),
                        'BIC': compute_BIC(y, y_hat, n_par),
                        'n_par': n_par,
                    }

                    # Store fit results as dictionary
                    _index_dict = dict(zip(meta_fit_index, [
                        prop_class.prop_name, prop_class.dist_name, prop_class.prop_name,
                        fish_age, fish_genotype, exp_ID, model.name, 'median',
                    ]))
                    _popt_dict = dict(zip(model.par_names, ind_meta_popt))
                    ind_meta_fit_results.append({**_index_dict, **_popt_dict, **_error_dict})

                    # # Plot individual results
                    # if isinstance(path_to_fig_folder, Path):
                    #     b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)
                    #     y_hat = model.eval(b_left, b_right, dt_hat, *ind_meta_popt)
                    #     fig, ax = plt.subplots(1, 1)
                    #     ax.plot(ts, y, '.', color=agent.color, label='data')
                    #     ax.plot(ts_hat, y_hat, '--', color=COLOR_MODEL, label='fit')
                    #     hide_spines(ax)
                    #     set_lims(ax, [0, 270], prop_class.par_lims[0])
                    #     set_labels(ax, 'Time (s)', f'{prop_class.label}\n({prop_class.unit})')
                    #     set_axlines(ax, axhlines=prop_class.prop_axlines)
                    #     set_ticks(ax,
                    #               x_ticks=np.arange(0, 270 + 1, 30), x_ticksize=0,
                    #               y_ticks=prop_class.par_ticks[0], y_ticksize=5, )
                    #     savefig(
                    #         fig,
                    #         path_to_fig_folder.joinpath(
                    #             f'fit_spatial_temporal_model',
                    #             f'{prop_class.prop_name}_{agent.name}',
                    #             f'{prop_class.prop_name}_{exp_ID:03d}.pdf'),
                    #         close_fig=True,
                    #     )

            print(f"\033[92mdone\033[0m")

    # Create dataframe of meta fit results ####################################
    # Fit to individual fish
    ind_meta_fit_df = pd.DataFrame(ind_meta_fit_results).set_index(meta_fit_index).sort_index()
    # Fit to mean over fish
    mean_meta_fit_df = pd.DataFrame(mean_meta_fit_results).set_index(meta_fit_index).sort_index()
    # Mean over fits to individual fish
    groupby_labels = ['prop_name', 'dist_name', 'bin_name', 'fish_age', 'fish_genotype', 'model_name', 'par_name', ]
    mean_ind_meta_fit_df = ind_meta_fit_df.groupby(groupby_labels).mean()
    # # Update index to match mean_meta_fit_df
    mean_ind_meta_fit_df['experiment_ID'] = 0
    mean_ind_meta_fit_df.set_index('experiment_ID', append=True, inplace=True)

    return ind_meta_fit_df, mean_ind_meta_fit_df, mean_meta_fit_df,


# Plot functions ##############################################################
def plot_time_series(
        median_df, agents, fit_agents,
        prop_classes, models,
        time_lim, time_ticks,
        t_ns, b_left_ns, b_right_ns,
        fit_df=None,
        row_y_cm=4,     # cm, height of each property row (containing two agents)
        ax_x_cm=7.5,    # cm, width of each axis (single agent)
        ax_y_cm=2,      # cm, height of each axis (single agent)
        pad_y_cm=0.5,   # cm, padding between axes
        offset_y_cm=2,  # cm, offset for first row (for x-labels)
):
    # Ensure prop_classes and models are iterable
    if not isinstance(prop_classes, (list, tuple)):
        prop_classes = [prop_classes]
    if not isinstance(models, (list, tuple)):
        models = [models]

    fig = create_figure(fig_height=25)

    for prop_num, prop_class in enumerate(prop_classes[::-1]):
        for k, (agent, fit_agent) in enumerate(zip(agents[::-1], fit_agents[::-1])):
            # Add axes
            i = 0
            l, b, w, h = (
                pad_x_cm + i * ax_x_cm,
                offset_y_cm + prop_num * row_y_cm + k * ax_y_cm,
                ax_x_cm - pad_x_cm,
                ax_y_cm - pad_y_cm,
            )
            ax = add_axes(fig, l, b, w, h)

            # Plot mean and sem over individuals ##########################
            # Select data for this property and agent
            group = (
                median_df  # Use bootstrapped data
                .query(agent.query)
                .groupby('time')
                [prop_class.prop_name]
            )
            mean = group.mean()
            sem = group.sem()
            std = group.std()
            ax.plot(mean.index, mean, color=agent.color, linestyle='-', label=agent.label)
            ax.fill_between(mean.index, mean - sem, mean + sem, color=agent.color, alpha=ALPHA)

            # Plot fit of models ##########################################
            if not isinstance(fit_df, type(None)):
                # Specify time and brightness for plotting
                dt_hat = 1
                ts_hat = np.arange(dt_hat / 2, t_ns[-1] + dt_hat / 2, dt_hat)
                b_left, b_right = get_b_values(ts_hat, t_ns, b_left_ns, b_right_ns)

                colors = iter([COLOR_MODEL, 'tab:grey'])
                linestyles = iter(['--', '-.'])

                for model in models:
                    color = next(colors)
                    ls = next(linestyles)

                    popt = (
                        fit_df
                        .xs(model.name, level='model_name')
                        .query(fit_agent.query)
                        .xs(prop_class.prop_name, level='prop_name')
                        [model.par_names].values[0]
                    )

                    y_hat = model.eval_cont(b_left, b_right, dt_hat, *popt)
                    ax.plot(ts_hat, y_hat, color=color, linestyle=ls, label=f'Fit ({model.label})')

            # Format ax
            set_ticks(ax,
                      x_ticks=time_ticks,
                      y_ticks=prop_class.par_ticks[0])
            hide_spines(ax, ['top', 'right', 'bottom'], hide_ticks=True)
            set_bounds(ax, y=[prop_class.par_ticks[0][0], prop_class.par_ticks[0][-1]])
            set_lims(ax, time_lim, prop_class.par_lims[0])
            set_spine_position(ax, ['left', 'bottom'])
            set_axlines(ax, axhlines=prop_class.prop_axlines)

            # Add y-label for upper agent
            if k == 1:
                set_labels(ax, y=f'{prop_class.label}\n({prop_class.unit})')

            # Add x-ticks for lowest row
            if (prop_num + k) == 0:
                show_spines(ax, spines=['bottom'])
                set_ticks(ax, x_ticks=time_ticks)
                set_labels(ax, x='Time (s)')

            # Add legend for top two rows
            if prop_num >= len(prop_classes) - 1:
                set_legend(ax)

    # Add stimulus bar on top
    # Add axes
    i = 0
    l, b, w, h = (
        pad_x_cm + i * ax_x_cm,
        offset_y_cm + prop_num * row_y_cm + (k + 1) * ax_y_cm,
        ax_x_cm - pad_x_cm,
        0.25,  # cm
    )
    ax = add_axes(fig, l, b, w, h)
    ax.set_xlim(time_lim[0] + 0.1, time_lim[-1])
    add_stimulus_bar(ax, t_ns, b_left_ns, b_right_ns, axvline=False, y_pos=0.5, height=1)
    set_ticks(ax, x_ticks=[], y_ticks=[])

    return fig


def plot_params(
        ind_fit_df, mean_fit_df, agents,
        prop_classes, model,
        ax_x_cm=2, ax_y_cm=2,
        pad_x_cm=1.5, pad_y_cm=0.5,
        jitter=0.5,
):
    # Ensure prop_classes are iterable
    if not isinstance(prop_classes, (list, tuple)):
        prop_classes = [prop_classes]

    # Set figure width depending on number of parameters
    fig_width_cm = pad_x_cm + model.n_par * ax_x_cm
    fig = create_figure(fig_height=21, fig_width=fig_width_cm)
    fig.suptitle(f'{model.label}')

    for prop_num, prop_class in enumerate(prop_classes[::-1]):
        ind_popt = (
            ind_fit_df
            .xs(prop_class.prop_name, level='prop_name')
            .xs(model.name, level='model_name')
        )
        mean_popt = (
            mean_fit_df
            .xs(prop_class.prop_name, level='prop_name')
            .xs(model.name, level='model_name')
        )
        for i, par_name in enumerate(model.par_names):
            # Add axes
            j = 2 * prop_num
            l, b, w, h = (
                pad_x_cm + i * ax_x_cm,
                3*pad_y_cm + j * ax_y_cm,
                ax_x_cm - pad_x_cm,
                ax_y_cm - pad_y_cm,
            )
            ax = add_axes(fig, l, b, w, h)

            sns.stripplot(  # Individual fits
                data=ind_popt.reset_index(), x='fish_age', y=par_name,
                hue='fish_age', palette=AGE_PALETTE_DICT, alpha=ALPHA, size=MARKER_SIZE,
                marker=MARKER_HOLLOW,
                dodge=True, jitter=jitter,
                label='Bootstrapped', legend=False,
                ax=ax
            )
            sns.stripplot(  # Mean over fits to individual data
                data=mean_popt.reset_index(),
                x='fish_age', y=par_name,
                hue='fish_age', palette=ListedColormap([COLOR_ANNOT]), size=MARKER_SIZE_LARGE, marker='X',
                dodge=True,
                label='Mean', legend=False,
                ax=ax,
            )

            # Format
            # # Set ticks and limits dynamically
            y_min, y_max = ax.get_ylim()
            # # Round to integer
            y_min = np.floor(y_min)
            y_max = np.ceil(y_max)
            y_ticks = np.linspace(y_min, y_max, 3)
            y_ticks = np.round(y_ticks, 0)
            y_ticklabels = [int(tick) for tick in y_ticks]

            set_ticks(ax, x_ticks=[], y_ticks=y_ticks, y_ticklabels=y_ticklabels)
            set_labels(ax, x='', y=model.par_labels_dict[par_name])
            set_spine_position(ax, )
            hide_spines(ax, ['top', 'right', 'bottom'])
    return fig


def plot_fit_errors(
        ind_fit_df, mean_fit_df, agents,
        prop_classes, models,
        metric='MSE',
        ax_x_cm=7.5, ax_y_cm=2,
        pad_y_cm=0.5,
        jitter=0.2,
):
    # Ensure prop_classes are iterable
    if not isinstance(prop_classes, (list, tuple)):
        prop_classes = [prop_classes]

    # Set plot order based on model parameters
    model_names = [model.name for model in models]
    model_npars = [model.n_par for model in models]
    model_labels = [model.label for model in models]
    # Sort models by number of parameters
    order = [x for _, x in sorted(zip(model_npars, model_names))]
    x_ticks = np.arange(len(models))
    x_ticklabels = [f'{x} ({n_par})' for n_par, x in sorted(zip(model_npars, model_labels))]

    # Set figure width depending on number of parameters
    fig = create_figure(fig_height=29, fig_width=(3 * pad_x_cm + ax_x_cm))
    for prop_num, prop_class in enumerate(prop_classes[::-1]):
        for k, agent in enumerate(agents[::-1]):
            # Add axes
            i = 0
            j = 2 * prop_num + k
            l, b, w, h = (
                2*pad_x_cm + i * ax_x_cm,
                10*pad_y_cm + j * ax_y_cm,
                ax_x_cm - pad_x_cm,
                ax_y_cm - pad_y_cm,
            )
            ax = add_axes(fig, l, b, w, h)

            ind_popt = (
                ind_fit_df
                .xs(prop_class.prop_name, level='prop_name')
                .query(agent.query)
                .reset_index()
            )
            mean_popt = (
                mean_fit_df
                .xs(prop_class.prop_name, level='prop_name')
                .query(agent.query)
                .reset_index()
            )

            sns.stripplot(  # Individual fits
                data=ind_popt, x='model_name', y=metric,
                hue='fish_age', palette=AGE_PALETTE_DICT, alpha=ALPHA, size=MARKER_SIZE,
                marker=MARKER_HOLLOW,
                dodge=True, jitter=jitter,
                label='Bootstrapped', legend=False,
                order=order,
                ax=ax
            )
            sns.stripplot(  # Mean over fits to individual data
                data=mean_popt,
                x='model_name', y=metric,
                hue='fish_age', palette=ListedColormap([COLOR_ANNOT]), size=MARKER_SIZE_LARGE, marker='X',
                dodge=True,
                label='Mean', legend=False,
                order=order,
                ax=ax,
            )

            # Format
            # # Set limits and ticks based on prop range
            # prop_range = prop_class.par_ticks[0][-1] - prop_class.par_ticks[0][0]
            # prop_ticks = [0, int(prop_range/2), prop_range]
            # set_ticks(ax, x_ticks=[], y_ticks=prop_ticks)
            hide_spines(ax, ['top', 'right', 'bottom'], hide_ticks=True)
            if metric == 'MSE':
                ticks = prop_class.mse_ticks_dict[agent.name]
                ticklabels = prop_class.mse_ticklabels_dict[agent.name]
                label = prop_class.mse_label
            elif metric == 'RMSE':
                ticks = prop_class.rmse_ticks_dict[agent.name]
                ticklabels = prop_class.rmse_ticklabels_dict[agent.name]
                label = prop_class.rmse_label
            else:
                label = f'{prop_class.label}\n{metric}'
                ticks, ticklabels = None, None

            set_ticks(ax, y_ticks=ticks, y_ticklabels=ticklabels)

            # Add y-label for upper agent
            set_labels(ax, x='', y='')
            if k == 1:
                set_labels(ax, y=label)

            # Add x-ticks for lowest row
            if j == 0:
                show_spines(ax, spines=['bottom'])
                set_spine_position(ax, spines=['bottom'])
                set_ticks(
                    ax,
                    x_ticks=x_ticks, x_ticklabels=x_ticklabels, x_tickrotation=90,
                )

    return fig



def plot_fitted_pars_obs(
        ind_meta_fit_df, mean_ind_meta_fit_df, mean_meta_fit_df,
        prop_class, agents,
        ax_x_cm=2.5, ax_y_cm=4,  # cm
        jitter=0.5
):
    stat_str = ''

    # For statistics, ensure agents is list of length 2
    if not isinstance(agents, list):
        agents = [agents]
    if len(agents) == 1:
        agents = agents * 2

    # Get values
    model = prop_class.model
    ind_meta_popt = ind_meta_fit_df.xs(prop_class.prop_name, level='prop_name')
    mean_ind_meta_popt = mean_ind_meta_fit_df.xs(prop_class.prop_name, level='prop_name')
    # mean_meta_popt = mean_meta_fit_df.xs(prop_class.prop_name, level='prop_name')

    fig = create_figure()
    for i, meta_par_name in enumerate(model.par_names):
        meta_par_label = model.par_labels_dict[meta_par_name]

        # Statistics ##########################################################
        stat_str += f"\n{prop_class.prop_name} | {meta_par_name}\n"
        _stat_str, stat_dict = get_stats_two_groups(
            agents[0].name, agents[1].name,
            ind_meta_popt.query(agents[0].query)[meta_par_name],
            ind_meta_popt.query(agents[1].query)[meta_par_name],
        )
        stat_str += _stat_str

        # Stripplot ###########################################################
        j = 1  # assign to row
        l, b, w, h = (
            pad_x_cm + i * ax_x_cm,
            pad_y_cm + j * ax_y_cm,
            ax_x_cm - pad_x_cm,
            ax_y_cm - pad_y_cm,
        )
        ax = add_axes(fig, l, b, w, h)
        sns.stripplot(  # Individual fits
            data=ind_meta_popt.reset_index(), x='fish_age', y=meta_par_name,
            hue='fish_age', palette=AGE_PALETTE_DICT, alpha=ALPHA, size=MARKER_SIZE,
            marker=MARKER_HOLLOW, dodge=True, legend=False, jitter=jitter,
            ax=ax
        )
        # sns.stripplot(  # Fit to mean data
        #     data=mean_meta_popt.reset_index(), x='fish_age', y=meta_par_name,
        #     hue='fish_age', palette=ListedColormap([COLOR_MODEL]), alpha=ALPHA, size=MARKER_SIZE_LARGE, marker='X',
        #     ax=ax, dodge=True, legend=False,
        # )
        sns.stripplot(    # Mean over fits to individual data
            data=mean_ind_meta_popt.reset_index(),
            x='fish_age', y=meta_par_name,
            hue='fish_age', palette=ListedColormap([COLOR_ANNOT]), size=MARKER_SIZE_LARGE, marker='X',
            ax=ax, dodge=True, legend=False,
        )

        # Add statistics
        # Compare between agents
        # ax.text(0.5, 1.1, p_value_to_stars(stat_dict['Mann-Whitney U test']['p']), ha='center', va='bottom', transform=ax.transAxes)
        # x-coordinates in dataspace, y-coordinates in axes space
        add_stats(ax, 0, 1, ANNOT_Y, p_value_to_stars(stat_dict['Mann-Whitney U test']['p']))
        # # Compare to zero
        # if not 'tau' in meta_par_name:
        #     ax.text(0.2, 1, p_value_to_stars(stat_dict['Bootstrapping']['p0']), ha='center', va='bottom', transform=ax.transAxes)
        #     ax.text(0.8, 1, p_value_to_stars(stat_dict['Bootstrapping']['p1']), ha='center', va='bottom', transform=ax.transAxes)

        # Format
        if meta_par_name == 'offset':
            ticks = prop_class.par_ticks[0]
            bounds = prop_class.par_lims[0]
            axline = prop_class.prop_axlines
        else:
            ticks = model.par_ticks[meta_par_name]
            bounds = [np.min(ticks), np.max(ticks)]
            axline = model.par_axlines[meta_par_name]

        set_ticks(ax, x_ticks=[], y_ticks=ticks)
        set_labels(ax, x='', y=meta_par_label)
        set_bounds(ax, y=bounds)
        set_axlines(ax, axhlines=axline)
        set_spine_position(ax, )
        hide_spines(ax, ['top', 'right', 'bottom'])

        # Histogram ##############################################################
        j = 0  # assign to row
        l, b, w, h = (
            pad_x_cm + i * ax_x_cm,
            pad_y_cm + j * ax_y_cm,
            ax_x_cm - pad_x_cm,
            ax_y_cm - pad_y_cm,
        )
        ax = add_axes(fig, l, b, w, h)
        sns.histplot(
            data=ind_meta_popt.reset_index(), y=meta_par_name, binrange=bounds, bins=21,
            hue='fish_age', palette=AGE_PALETTE_DICT, linestyle='none', alpha=2*ALPHA,
            ax=ax, legend=False,
        )

        set_ticks(ax, y_ticks=ticks)
        set_labels(ax, x='Nr. fish', y=meta_par_label)
        set_bounds(ax, y=bounds)
        set_axlines(ax, axhlines=axline)
        set_spine_position(ax, )
        hide_spines(ax, )

    return fig, stat_str

# #############################################################################
# Integrator
# #############################################################################
def concat_values(
        exp_df, prop_class,
        t_intervals,
):
    p_left_all, ts_all = [], []
    b_left_all, b_right_all = [], []
    for i, stim_query in enumerate(
            ['_LdarkRdark_LdarkRbright', '_LbrightRbright_LdarkRbright']):  # dark vs bright adaptation
        # Define brightness values
        if stim_query == '_LdarkRdark_LdarkRbright':
            b_left_ns = np.asarray([300, 10, 10])
            b_right_ns = np.asarray([300, 10, 300])
        elif stim_query == '_LbrightRbright_LdarkRbright':
            b_left_ns = np.asarray([10, 300, 10])
            b_right_ns = np.asarray([10, 300, 300])
        else:
            raise ValueError(f'Unknown stim_query: {stim_query}')

        for l, t_interval in enumerate(t_intervals):
            # Extract data for this stimulus and interval
            stim_name = f'{t_interval:02.0f}s{stim_query}'
            stim_df = exp_df.xs(stim_name, level='stimulus_name')

            # Get mean and sem over individuals
            group = stim_df.groupby('time')[prop_class.prop_name]
            mean = group.mean()
            sem = group.sem()
            std = group.std()
            ts = mean.index
            # # Fill NaN values with previous value
            mean.ffill(inplace=True)
            p_left_all.append(mean.values)
            ts_all.append(ts)  # depends on whether all p_left vaues are present

            # # Generate brightness values
            t_ns = np.asarray([-20, 0 - t_interval, 0, 100])
            b_left, b_right = get_b_values(ts, t_ns, b_left_ns, b_right_ns)
            b_left_all.append(b_left)
            b_right_all.append(b_right)

    return p_left_all, ts_all, b_left_all, b_right_all


def fit_spatial_temporal_model_integrator(
    median_df, agents, prop_classes,
    t_intervals,
    path_to_fig_folder=None,
):
    # Ensure agents, prop_classes is iterable
    if not isinstance(agents, list):
        agents = [agents]
    if not isinstance(prop_classes, list):
        prop_classes = [prop_classes]

    # rename fish_age 26 to 27 for plotting
    median_df.rename(index={26: 27}, level='fish_age', inplace=True)

    ind_meta_fit_results = []
    mean_meta_fit_results = []
    meta_fit_index = [
        'prop_name', 'dist_name', 'bin_name',
        'fish_age', 'fish_genotype',
        'experiment_ID', 'par_name',
    ]

    # Specify time for plotting
    dt_hat = 1  # s, bin size of data
    ts_hat = np.arange(-20 + dt_hat/2, 10 + dt_hat/2, dt_hat)

    for j, agent in enumerate(agents):
        # Select data for this agent, limit time window
        agent_df = median_df.query(agent.query).query('-20 <= time < 10')

        for prop_class in prop_classes:
            print(f"\tFitting {prop_class.prop_name} | {agent.name} ", end='')
            model = prop_class.model

            # Fit mean over individuals ###########################################
            fish_age = agent_df.index.unique('fish_age')[0]
            fish_genotype = agent_df.index.unique('fish_genotype')[0]
            exp_ID = 0

            # Concatenate values
            p_left_all, ts_all, b_left_all, b_right_all = concat_values(agent_df, prop_class, t_intervals)
            p_left_flat = np.reshape(p_left_all, -1)
            t_flat = np.reshape(ts_all, -1)

            # Set Spatial-temporal model
            def fit_wrapper(
                    x,
                    tau_lpf,
                    w_repulsor_pos, w_repulsor_neg, w_attractor,
            ):
                model = prop_class.model
                y_hat = []
                for b_left, b_right in zip(b_left_all, b_right_all):
                    y_hat.append(model.eval_cont(
                        b_left, b_right, dt_hat,
                        tau_lpf,
                        w_repulsor_pos, w_repulsor_neg, w_attractor,
                    ))

                return np.reshape(y_hat, -1)

            res = curve_fit(
                f=fit_wrapper,
                xdata=t_flat, ydata=p_left_flat,
                p0=model.x0, bounds=model.bounds_curve_fit,
            )
            mean_meta_popt = res[0]

            # Store fit results as dictionary
            _index_dict = dict(zip(meta_fit_index, [
                prop_class.prop_name, prop_class.dist_name, prop_class.prop_name,
                fish_age, fish_genotype, exp_ID, 'median',
            ]))
            _popt_dict = dict(zip(model.par_names, mean_meta_popt))
            mean_meta_fit_results.append({**_index_dict, **_popt_dict})

            # Fit individuals #################################################
            grouped = agent_df.groupby(['fish_age', 'fish_genotype', 'experiment_ID'])
            for idx, exp_df in grouped:
                fish_age, fish_genotype, exp_ID = idx

                # Concatenate values
                p_left_all, ts_all, b_left_all, b_right_all = concat_values(exp_df, prop_class, t_intervals)
                try:
                    p_left_flat = np.reshape(p_left_all, -1)
                    t_flat = np.reshape(ts_all, -1)
                except Exception as e:
                    print(f"\tfit_spatial_temporal_model_integrator(): Error reshaping {agent.name} {exp_ID:03d}: {e}")
                    continue

                try:
                    res = curve_fit(
                        f=fit_wrapper,
                        xdata=t_flat, ydata=p_left_flat,
                        p0=model.x0, bounds=model.bounds_curve_fit,
                    )
                    ind_meta_popt = res[0]
                except Exception as e:
                    print(f"\tfit_spatial_temporal_model_integrator(): Error fitting {agent.name} {exp_ID:03d}: {e}")
                    continue
                except RuntimeWarning as e:
                    print(f"\tfit_spatial_temporal_model_integrator(): RuntimeWarning fitting {agent.name} {exp_ID:03d}: {e}")
                    continue

                # Store fit results as dictionary
                _index_dict = dict(zip(meta_fit_index, [
                    prop_class.prop_name, prop_class.dist_name, prop_class.prop_name,
                    fish_age, fish_genotype, exp_ID, 'median',
                ]))
                _popt_dict = dict(zip(model.par_names, ind_meta_popt))
                ind_meta_fit_results.append({**_index_dict, **_popt_dict})

    # Create dataframe of meta fit results ####################################
    # Fit to mean over fish
    mean_meta_fit_df = pd.DataFrame(mean_meta_fit_results)
    mean_meta_fit_df.set_index(meta_fit_index, inplace=True)
    mean_meta_fit_df.sort_index(inplace=True)

    # Fit to individual fish
    ind_meta_fit_df = pd.DataFrame(ind_meta_fit_results)
    ind_meta_fit_df.set_index(meta_fit_index, inplace=True)
    ind_meta_fit_df.sort_index(inplace=True)

    # Mean over fits to individual fish
    mean_ind_meta_fit_df = (
        ind_meta_fit_df
        .rename(index={26: 27}, level='fish_age')  # rename fish_age 26 to 27 for plotting
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


def fit_spatial_temporal_model_integrator_v2(
        median_df, agents,
        prop_classes, models,
        t_intervals,
        path_to_fig_folder=None,
        dt_hat=1  # s, bin size of data
):
    # Ensure agents, prop_classes, models is iterable
    if not isinstance(agents, list):
        agents = [agents]
    if not isinstance(prop_classes, list):
        prop_classes = [prop_classes]
    if not isinstance(models, list):
        models = [models]

    # rename fish_age 26 to 27 for plotting
    median_df.rename(index={26: 27}, level='fish_age', inplace=True)

    ind_meta_fit_results = []
    mean_meta_fit_results = []
    meta_fit_index = [
        'prop_name', 'dist_name', 'bin_name',
        'fish_age', 'fish_genotype',
        'experiment_ID',
        'model_name', 'par_name',
    ]

    for j, agent in enumerate(agents):
        # Select data for this agent, limit time window
        agent_df = median_df.query(agent.query).query('-5 <= time < 10')
        ts_hat = np.arange(-5 + dt_hat / 2, 10 + dt_hat / 2, dt_hat)
        ts_hat = np.arange(-5 + dt_hat, 10 + dt_hat / 2, dt_hat)

        for prop_class in prop_classes:
            print(f"\tFitting {agent.name} {prop_class.prop_name} | ", end='')
            for model in models:
                print(f"{model.name} | ", end='')

                # Fit mean over individuals ###########################################
                fish_age = agent_df.index.unique('fish_age')[0]
                fish_genotype = agent_df.index.unique('fish_genotype')[0]
                exp_ID = 0

                # Concatenate values
                p_left_all, ts_all, b_left_all, b_right_all = concat_values(agent_df, prop_class, t_intervals)
                p_left_flat = np.reshape(p_left_all, -1)
                t_flat = np.reshape(ts_all, -1)

                # Load stimulus
                model.set_stimulus_integrator(ts_all, b_left_all, b_right_all)
                # Set bounds (can be property specific)
                model.set_bounds(prop_class.prop_name)

                res = curve_fit(
                    f=model.fitfunc_integrator,
                    xdata=t_flat, ydata=p_left_flat,
                    p0=model.x0, bounds=model.bounds_curve_fit,
                )
                mean_meta_popt = res[0]

                # Store fit results as dictionary
                _index_dict = dict(zip(meta_fit_index, [
                    prop_class.prop_name, prop_class.dist_name, prop_class.prop_name,
                    fish_age, fish_genotype, exp_ID, model.name, 'median',
                ]))
                _popt_dict = dict(zip(model.par_names, mean_meta_popt))
                mean_meta_fit_results.append({**_index_dict, **_popt_dict})

                # Inspect result
                if isinstance(path_to_fig_folder, Path):
                    for p_left, ts, b_left, b_right in zip(p_left_all, ts_all, b_left_all, b_right_all):
                        fig, ax = plt.subplots(1, 1)
                        y_hat = model.eval_cont(b_left, b_right, dt_hat, *mean_meta_popt)

                        ax.plot(ts, p_left, '.-', color=agent.color, label='data')
                        ax.plot(ts_hat, y_hat, '--', color=COLOR_MODEL, label='fit')

                        ax.plot(ts, b_left / 10, '-', color='tab:blue', label='b_left')
                        ax.plot(ts, b_right / 10, '-', color='tab:red', label='b_right')

                        hide_spines(ax)
                        # set_lims(ax, [-10, 10], prop_class.par_lims[0])
                        set_labels(ax, 'Time (s)', f'{prop_class.label}\n({prop_class.unit})')
                        set_axlines(ax, axhlines=prop_class.prop_axlines, axvlines=0)
                        # set_ticks(ax,
                        #           x_ticks=np.arange(-10, 10 + 1, 10), x_ticksize=0,
                        #           y_ticks=prop_class.par_ticks[0], y_ticksize=5, )
                    savefig(fig, path_to_fig_folder.joinpath(f'fit_spatial_temporal_model', f'{prop_class.prop_name}_mean_{agent.name}.pdf'), close_fig=True)

                # Fit individuals #################################################
                grouped = agent_df.groupby(['fish_age', 'fish_genotype', 'experiment_ID'])
                for idx, exp_df in grouped:
                    fish_age, fish_genotype, exp_ID = idx

                    # Concatenate values
                    p_left_all, ts_all, b_left_all, b_right_all = concat_values(exp_df, prop_class, t_intervals)
                    try:
                        p_left_flat = np.reshape(p_left_all, -1)
                        t_flat = np.reshape(ts_all, -1)
                    except Exception as e:
                        print(f"\tfit_spatial_temporal_model_integrator(): Error reshaping {agent.name} {exp_ID:03d}: {e}")
                        continue

                    try:
                        res = curve_fit(
                            f=model.fitfunc_integrator,
                            xdata=t_flat, ydata=p_left_flat,
                            p0=model.x0, bounds=model.bounds_curve_fit,
                        )
                        ind_meta_popt = res[0]
                    except Exception as e:
                        print(f"\tfit_spatial_temporal_model_integrator(): Error fitting {agent.name} {exp_ID:03d}: {e}")
                        continue
                    except RuntimeWarning as e:
                        print(f"\tfit_spatial_temporal_model_integrator(): RuntimeWarning fitting {agent.name} {exp_ID:03d}: {e}")
                        continue

                    # Store fit results as dictionary
                    _index_dict = dict(zip(meta_fit_index, [
                        prop_class.prop_name, prop_class.dist_name, prop_class.prop_name,
                        fish_age, fish_genotype, exp_ID, model.name, 'median',
                    ]))
                    _popt_dict = dict(zip(model.par_names, ind_meta_popt))
                    ind_meta_fit_results.append({**_index_dict, **_popt_dict})
            print(f"\033[92mdone\033[0m")

    # Create dataframe of meta fit results ####################################
    # Fit to individual fish
    ind_meta_fit_df = pd.DataFrame(ind_meta_fit_results).set_index(meta_fit_index).sort_index()
    # Fit to mean over fish
    mean_meta_fit_df = pd.DataFrame(mean_meta_fit_results).set_index(meta_fit_index).sort_index()
    # Mean over fits to individual fish
    groupby_labels = ['prop_name', 'dist_name', 'bin_name', 'fish_age', 'fish_genotype', 'model_name', 'par_name', ]
    mean_ind_meta_fit_df = ind_meta_fit_df.groupby(groupby_labels).mean()
    # # Update index to match mean_meta_fit_df
    mean_ind_meta_fit_df['experiment_ID'] = 0
    mean_ind_meta_fit_df.set_index('experiment_ID', append=True, inplace=True)

    return ind_meta_fit_df, mean_ind_meta_fit_df, mean_meta_fit_df,


def get_peaks(median_df, fit_df, agents, model, t_intervals, dt=1/60):
    prop_class = PercentageLeft()

    # Create list to collect results
    peak_results = []

    # Loop over agents
    for k, agent in enumerate(agents):
        # Select data for this agent
        agent_df = median_df.query(agent.query)
        fish_age = agent_df.index.unique('fish_age')[0]
        fish_genotype = agent_df.index.unique('fish_genotype')[0]

        # Get fitted parameters for this agent
        meta_popt = (
            fit_df
            .query(agent.query)
            .xs(prop_class.prop_name, level='prop_name')
            [model.par_names].values[0]
        )

        # Loop over different adaptation conditions
        for j, stim_query in enumerate(['_LbrightRbright_LdarkRbright', '_LdarkRdark_LdarkRbright']):  # dark vs bright adaptation
            # Define brightness values
            if stim_query == '_LdarkRdark_LdarkRbright':
                b_left_ns = np.asarray([300, 10, 10])
                b_right_ns = np.asarray([300, 10, 300])
            elif stim_query == '_LbrightRbright_LdarkRbright':
                b_left_ns = np.asarray([10, 300, 10])
                b_right_ns = np.asarray([10, 300, 300])
            else:
                raise ValueError(f'Unknown stim_query: {stim_query}')

            # Loop over intervals to get data peaks
            for t_interval in t_intervals:
                # Extract data for this stimulus and interval
                stim_name = f'{t_interval:02.0f}s{stim_query}'
                stim_df = agent_df.query(agent.query).xs(stim_name, level='stimulus_name')

                # Plot mean and sem over individuals
                group = stim_df.groupby('time')[prop_class.prop_name]
                mean = group.mean()
                std = group.std()

                # Find position of peak
                mean[mean.index.get_level_values('time') < 0] = np.nan  # Exclude values before stimulus
                peak_loc = (mean - 50).abs().idxmax()
                peak_mean = mean.loc[peak_loc]
                peak_std = std.loc[peak_loc]

                # Append result
                peak_results.append({
                    'type': 'data',
                    'fish_age': fish_age,
                    'fish_genotype': fish_genotype,
                    'stim_query': stim_query,
                    't_interval': t_interval,
                    'peak_mean': peak_mean,
                    'peak_std': peak_std,
                })

            # Compute fit prediction
            ts = np.arange(-20, 30, dt)
            t_intervals_model = np.arange(0, 10 + dt, dt)
            peak_loc = np.searchsorted(ts, 0, side='right')
            # peaks = []
            for l, t_interval in enumerate(t_intervals_model):
                # Set color
                color = agent.cmap((l + 1) / (len(t_intervals_model) + 1))

                # # Generate brightness values
                t_ns = np.asarray([-20, 0 - t_interval, 0, 100])
                b_left, b_right = get_b_values(ts, t_ns, b_left_ns, b_right_ns)

                # # Evaluate model
                y_hat = model.eval_cont(b_left, b_right, dt, *meta_popt)

                # # Find position of peak
                peak_mean = y_hat[peak_loc]
                # peaks.append(y_hat[peak_loc])

                # fig2 = plt.figure()
                # plt.plot(ts, y_hat)
                # plt.plot(ts[peak_loc], peak_mean, 'o')

                # Append result
                peak_results.append({
                    'type': 'fit',
                    'fish_age': fish_age,
                    'fish_genotype': fish_genotype,
                    'stim_query': stim_query,
                    't_interval': t_interval,
                    'peak_mean': peak_mean,
                })

    # Convert list of dicts to DataFrame
    peak_df = pd.DataFrame(peak_results)
    peak_df.set_index(['type', 'fish_age', 'fish_genotype', 'stim_query'], inplace=True)
    return peak_df

